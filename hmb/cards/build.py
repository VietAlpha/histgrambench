"""`build_card` orchestration (§7.3-§8.8, asc-2.0.0): per crop, two staged generation calls per writer, then chunked
verification (each writer is asked about every union fact, in chunks of at most `card_verify_chunk_size`
statements; its own facts count as supported unless it votes false), then deterministic alignment -> closure/MGS -> consensus -> frozen card,
with immutable artifacts.

Bounded by construction (2026-09-26 streamlining): a generation keeps at most `card_max_entities` entity lines
and `card_max_facts` fact lines; a verification call carries at most `card_verify_chunk_size` statements; every
reply is read line by line, so a reply cut at `max_tokens` (recorded from the provider's stop reason) keeps its
complete lines. Calls per crop <= 2 * 4 + 2 * ceil(|union MGS| / chunk), |union MGS| <= 2 * (2 * card_max_entities + card_max_facts).

Staged generation (`GENERATION_PROTOCOL`, 2026-09-26 repair): per writer an entity call (entity lines only), then a
fact call whose user text lists that writer's own parsed entity lines (fact lines only). A stage whose reply ends at
the output limit with no complete line of its kind is asked once more, with `prompts.retry_note` (a different
request asking for at most half the cap); so at most 4 generation calls per writer, and a writer with no entity
line gets no fact call. Every call's stop reason and usable-line count is stored. One writer's failure never
discards the other's output, and a failed writer still verifies.

Artifacts (hmb.store.Layout):
  semantic_cards/<item_id>.json          the Card (union MGS with eligibility per fact)
  semantic_cards/votes/<item_id>.json    raw writer replies, stop reasons, parse notes, OOV, alignment log,
                                         per-writer MGS, union, verification statements and votes per chunk
  manifests/cards/<card_uid>/<item_id>.json  (C29) asc_version and every hashed input (§8.8), writer order,
                                         family map, OOV rates, truncated calls, call counts, sha256s

Model calls go through `hmb.providers.call_model`. `call` may inject a stand-in with the same signature
(tests); a fake response is refused under the production profile. builder_id = the writer's model_id.
Cache identity: every writer call uses the scope `CARD_CALL_SCOPE`; the request itself (spec, system prompt,
text, image, max_tokens) completes the key, so a change of gate levels, thresholds or post-processing never
re-bills a call, and one writer's settings never re-bill the other writer's calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from hmb.hashing import sha256_file, sha256_json
from hmb.ontology import ASC_MAJOR_MINOR_PATCH, CARD_PREDICATES
from hmb.providers import BatchDeferred
from hmb.roster import builder_rule_problems
from hmb.schemas import Card, ModelRequest, ModelSpec
from hmb.store import write_json_immutable
from hmb.vocab import Profile, Role

from .align import AlignParams, align
from .closure import closure_spec
from .consensus import consensus, consensus_rule_hash, fact_label
from .normalize import normalization_spec
from .parse import ParseResult, parse_builder_output, stage_line_count
from .prompts import (fact_stage_user_text, generation_user_text, prompt_hashes, retry_note, system_prompt,
                      verification_user_text)
from .verify import build_union, parse_votes, split_votes, verification_chunks

CARD_CALL_SCOPE = f"{ASC_MAJOR_MINOR_PATCH}|writer-call"
GENERATION_PROTOCOL = "staged-entities-facts-1"     # card identity (asc_hash_inputs); not a cache-key input
CAP_PARAMS = ("card_max_entities", "card_max_facts", "card_verify_chunk_size")
TRUNCATION_STOP_REASONS = ("max_tokens", "length", "model_length")


class RosterError(ValueError):
    """The builder roster violates §7.3 / §10.9."""


class FakeResponseRefused(RuntimeError):
    """A fake provider response reached a production path."""


def check_builder_roster(builders: list[ModelSpec]) -> None:
    """§7.3: the `roster-check` builder rule (exactly 2 builders in 2 families, owner 2026-09-26; D17); ids unique;
    role builder."""
    ids = [b.model_id for b in builders]
    if len(set(ids)) != len(ids):
        raise RosterError(f"duplicate builder model ids {ids}")
    bad = [b.model_id for b in builders if b.role != Role.builder.value]
    if bad:
        raise RosterError(f"not builder role: {bad}")
    probs = builder_rule_problems(builders)
    if probs:
        raise RosterError(probs[0])


def builder_judge_overlap(builders: list[ModelSpec], judges: list[ModelSpec]) -> set[str]:
    """Families shared by builders and judges (must be empty in the primary configuration, §10.9)."""
    return {b.family for b in builders} & {j.family for j in judges}


def check_primary_disjoint(builders: list[ModelSpec], judges: list[ModelSpec]) -> None:
    shared = builder_judge_overlap(builders, judges)
    if shared:
        raise RosterError(f"builder and judge families overlap in the primary configuration: {sorted(shared)}")


def ontology_spec() -> list:
    return [[p.name, list(p.args), p.fclass.value, p.strictness.value, p.anchor.value, p.decidability.value,
             [list(x) for x in p.symmetric_pairs]] for p in CARD_PREDICATES.values()]


def asc_hash_inputs(builders: list[ModelSpec], th, max_tokens: int) -> dict:
    """Everything that determines a card's content (§8.8): vocabulary, rules, normalization, alignment, writers,
    prompts, caps and the consensus rule. Gate levels are not inputs: the gates read the card, so changing a
    level never changes a card. Reads calibrated parameters (production raises CalibrationMissing until the
    level freeze, by design)."""
    return {
        "ontology": ASC_MAJOR_MINOR_PATCH, "predicate_table": ontology_spec(),
        "closure": closure_spec(), "normalization": normalization_spec(),
        "alignment": AlignParams.from_thresholds(th).as_dict(),
        "builder_order": [b.model_id for b in builders],
        "family_map": {b.model_id: b.family for b in builders},
        "builders": [{"model_id": b.model_id, "snapshot": b.snapshot, "provider": b.provider,
                      "decoding": b.decoding, "effort": b.effort} for b in builders],
        "max_tokens": max_tokens, "caps": {k: th.value(k) for k in CAP_PARAMS}, "prompts": prompt_hashes(),
        "generation_protocol": GENERATION_PROTOCOL,
        "consensus_rule": consensus_rule_hash(),
    }


def asc_version(inputs: dict) -> str:
    return f"{ASC_MAJOR_MINOR_PATCH}+{sha256_json(inputs)[:16]}"


def generation_requests(crop_canonical: Path, ocr_words: list[dict], context_text: str | None, *,
                        builders: list[ModelSpec], th, max_tokens: int | None = None,
                        phase: str = "TEST") -> list[ModelRequest]:
    """The exact first-stage (entity) generation requests `build_card` sends (one per writer, frozen order). Used
    to enumerate cache entries before any call (resumable batch runs); a fact-stage request depends on its entity
    reply, so it can only be named once that reply exists."""
    mt = int(max_tokens if max_tokens is not None else th.value("card_builder_max_tokens"))
    crop_sha = sha256_file(crop_canonical)
    sys_, text = system_prompt("builder_generation_entities"), generation_user_text(ocr_words, context_text)
    return [ModelRequest(spec=s, system=sys_, text=text, image_paths=[str(crop_canonical)], image_sha256=[crop_sha],
                         max_tokens=mt, purpose="card_generation", phase=phase, cache_scope=CARD_CALL_SCOPE)
            for s in builders]


def _dump(out) -> dict | None:
    return None if out is None else out.model_dump(mode="json")


def _default_call():
    from hmb.providers import call_model  # pyright: ignore[reportMissingImports]  (owner Opus 1)
    return call_model


def _call_record(resp) -> dict:
    return {"cache_key": resp.cache_key, "fake": resp.fake, "stop_reason": resp.stop_reason,
            "truncated": resp.stop_reason in TRUNCATION_STOP_REASONS, "output_tokens": resp.output_tokens,
            "model_version_string": resp.model_version_string}


def build_card(item_id: str, crop_canonical: Path, ocr_words: list[dict], context_text: str | None, *,
               builders: list[ModelSpec], layout, th, ledger, allow_live: bool, provider_override=None,
               max_tokens: int | None = None, phase: str = "TEST", call: Callable | None = None) -> Card:
    check_builder_roster(builders)
    if th.is_production and any(b.provider == "fake" for b in builders):
        raise FakeResponseRefused("fake builder spec under the production profile")
    call = call or _default_call()
    mt = int(max_tokens if max_tokens is not None else th.value("card_builder_max_tokens"))
    inputs = asc_hash_inputs(builders, th, mt)
    asc = asc_version(inputs)
    crop_sha = sha256_file(crop_canonical)
    order = [b.model_id for b in builders]
    fam = {b.model_id: b.family for b in builders}
    max_e, max_f = int(th.value("card_max_entities")), int(th.value("card_max_facts"))
    chunk = int(th.value("card_verify_chunk_size"))

    def ask(spec: ModelSpec, system: str, text: str, purpose: str):
        req = ModelRequest(spec=spec, system=system, text=text, image_paths=[str(crop_canonical)],
                           image_sha256=[crop_sha], max_tokens=mt, purpose=purpose, phase=phase,
                           cache_scope=CARD_CALL_SCOPE)
        resp = call(req, layout, ledger, allow_live=allow_live, provider_override=provider_override)
        if resp.fake and th.is_production:
            raise FakeResponseRefused(f"fake response for {spec.model_id} under the production profile")
        return resp

    def stage(spec: ModelSpec, name: str, text: str, cap: int, log: list[dict]) -> str | None:
        """One bounded generation stage: the call, and at most one non-identical retry when the reply ended at
        the output limit with no complete line of its kind. Returns the text to parse (None: no usable reply)."""
        sys_ = system_prompt(f"builder_generation_{name}")
        for attempt, t in enumerate((text, text + retry_note(-(-cap // 2)))):
            resp = ask(spec, sys_, t, "card_generation")
            n = stage_line_count(resp.text, name)
            rec = {"stage": name, "attempt": attempt, "text": resp.text, **_call_record(resp), "usable_lines": n}
            log.append(rec)
            if n or not rec["truncated"]:
                return resp.text
        return None

    parsed: dict[str, ParseResult] = {}
    raw_gen: dict[str, dict] = {}
    deferred: list[BatchDeferred] = []        # hybrid batch runs: the writers' calls are independent
    for greq in generation_requests(crop_canonical, ocr_words, context_text, builders=builders, th=th,
                                    max_tokens=mt, phase=phase):
        spec = greq.spec
        log: list[dict] = []
        try:
            ent_text = stage(spec, "entities", greq.text, max_e, log)
            ents = parse_builder_output(ent_text or "", spec.model_id, spec.family, max_entities=max_e,
                                        max_facts=max_f, fact_raw="")
            fact_text = None
            if ents.output is not None:
                pv = {f.args[0]: f.args[1] for f in ents.output.facts if len(f.args) == 2
                      and CARD_PREDICATES[f.predicate].fclass.value == "entity"}
                el = [{**e.model_dump(mode="json"), "param": pv.get(e.local_id)} for e in ents.output.entities]
                fact_text = stage(spec, "facts", fact_stage_user_text(ocr_words, context_text, el), max_f, log)
        except BatchDeferred as d:
            deferred.append(d)
            continue
        parsed[spec.model_id] = parse_builder_output(ent_text or "", spec.model_id, spec.family, max_entities=max_e,
                                                     max_facts=max_f, fact_raw=fact_text or "")
        raw_gen[spec.model_id] = {"calls": log, "fact_stage": ("skipped: no entity line" if ents.output is None
                                                               else "ok" if fact_text is not None
                                                               else "no usable reply")}

    if deferred:
        raise BatchDeferred.merge(deferred)
    al = align({b: r.output for b, r in parsed.items()}, order, AlignParams.from_thresholds(th))
    U = build_union(al)

    ver_sys = system_prompt("builder_verification")
    votes: dict = {b: {} for b in order}
    ver_log: dict[str, list[dict]] = {b: [] for b in order}
    for spec in builders:
        for ci, items in enumerate(verification_chunks(U, al, item_id, spec.model_id, chunk)):
            try:
                resp = ask(spec, ver_sys, verification_user_text(ocr_words, [(i.sid, i.text) for i in items]),
                           "card_verification")
            except BatchDeferred as d:
                deferred.append(d)
                continue
            vp = parse_votes(resp.text, items)
            votes[spec.model_id].update(split_votes(items, vp))
            ver_log[spec.model_id].append({
                "chunk": ci, "statements": [{"sid": i.sid, "text": i.text, "fact": fact_label(i.fact)} for i in items],
                "raw": resp.text, **_call_record(resp),
                "votes": {k: v.value for k, v in vp.votes.items()}, "missing": vp.missing, "invalid": vp.invalid})

    if deferred:
        raise BatchDeferred.merge(deferred)
    abst = []
    for b in order:
        out = parsed[b].output
        if out is None:
            continue
        for a in out.abstentions:
            abst.append({"builder_id": b, "scope": a.scope,
                         "entity": al.local_to_canon.get((b, a.entity)) if a.entity else None,
                         "reason": a.reason})
    card = consensus(item_id, al, U, votes, fam, asc_version=asc, profile=Profile(th.profile).value,
                     abstentions=abst)

    card_sha = write_json_immutable(layout.card(item_id), card.model_dump(mode="json", by_alias=True))
    votes_doc = {
        "item_id": item_id, "asc_version": asc, "crop_sha256": crop_sha,
        "generation": {b: {**raw_gen[b], "malformed": parsed[b].malformed, "notes": parsed[b].notes,
                           "oov_rate": parsed[b].oov_rate, "bad_lines": parsed[b].bad_lines,
                           "over_cap_entities": parsed[b].over_cap_entities,
                           "over_cap_facts": parsed[b].over_cap_facts, "parsed": _dump(parsed[b].output)}
                       for b in order},
        "alignment_log": al.log,
        "per_builder_mgs": {b: [fact_label(f) for f in fs] for b, fs in U.per_builder_mgs.items()},
        "union_mgs": [fact_label(f) for f in U.facts],
        "verification": ver_log,
    }
    votes_sha = write_json_immutable(layout.card_votes(item_id), votes_doc)
    truncated = ([{"builder_id": b, "purpose": "card_generation", "stage": c["stage"], "attempt": c["attempt"],
                   "usable_lines": c["usable_lines"]} for b in order for c in raw_gen[b]["calls"] if c["truncated"]]
                 + [{"builder_id": b, "purpose": "card_verification", "chunk": c["chunk"]}
                    for b in order for c in ver_log[b] if c["truncated"]])
    manifest = {
        "item_id": item_id, "card_uid": card.card_uid, "asc_version": asc, "asc_hash_inputs": inputs,
        "builder_order": order, "family_map": fam,
        "oov_rate": {b: parsed[b].oov_rate for b in order},
        "malformed_builders": [b for b in order if parsed[b].output is None],
        "truncated_calls": truncated,
        "calls": {b: {"card_generation": len(raw_gen[b]["calls"]), "card_verification": len(ver_log[b])}
                  for b in order},
        # the repair's acceptance count: generation calls cut at the limit with no complete line (2026-09-26)
        "empty_truncated_generation_calls": [t for t in truncated if t["purpose"] == "card_generation"
                                             and not t["usable_lines"]],
        "verification_statements": {b: sum(len(c["statements"]) for c in ver_log[b]) for b in order},
        "card_sha256": card_sha, "votes_sha256": votes_sha, "profile": card.profile,
        "context_supplied": bool(context_text),
    }
    write_json_immutable(layout.card_manifest(card.card_uid, item_id), manifest)   # C29: per item
    return card
