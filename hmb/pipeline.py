"""Integration layer (lead-owned): one cell = submission -> sandbox -> CGX -> adjudication -> score.

`run_cell` is the single production path used for historical Tracks A/B and for the generated
known-geometry renders of the offline tests. It composes the component contracts only; it has no scoring logic of its own.

Resume and cache behaviour come from the components: every stage writes immutably and returns the
stored artifact on an identical re-run (process_submission, adjudicate, write_score); model calls are
content-addressed in `hmb.providers`, so a repeated run makes no duplicate paid call. A changed
evaluator input changes the relevant hash (judge_config_hash, cache_scope, harness_version) and the
stale artifact is refused rather than silently reused or overwritten.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cpu_pool
from .adjudication import adjudicate
from .cgx import extract_cgx, write_cgx
from .grading import score_cell
from .grading.cell import metric_hash, write_score
from .schemas import Card, ModelSpec
from .store import Layout, write_json_immutable
from .submission import finalize_submission, process_submission
from .submission.process import harness_version, model_dir_name
from .vocab import Outcome, Track


@dataclass
class CellResult:
    item_id: str
    model_id: str
    track: str
    outcome: str
    score: dict | None
    verdicts: dict | None
    submission: dict
    cgx_fact_count: int | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        v = self.verdicts or {}
        counts: dict[str, int] = {}
        for f in v.get("facts", []):
            counts[f["verdict"]] = counts.get(f["verdict"], 0) + 1
        return {"item_id": self.item_id, "model_id": self.model_id, "track": self.track,
                "outcome": self.outcome, "S": (self.score or {}).get("S"),
                "evaluator_failure": (self.score or {}).get("evaluator_failure"),
                "verdict_counts": counts, "cgx_fact_count": self.cgx_fact_count,
                "notes": self.notes}


def compile_paths(layout: Layout, model_id: str, track: str, item_id: str) -> tuple[Path, Path]:
    d = layout.compile_dir(model_dir_name(model_id), track, item_id)
    return d / "cand.pdf", d / "render.png"


def run_cell(card: Card, raw: str, stop_reason: str | None, *, item_id: str, contestant: ModelSpec,
             track: str, layout: Layout, th, judges: list[ModelSpec], ledger, allow_live: bool,
             crop_png: Path | None, meta: dict, provider_override=None, phase: str = "FINAL") -> CellResult:
    """Score one (item, model, track) cell end to end. Only outcome OK reaches adjudication (§9.5)."""
    t = Track(track).value
    if card.item_id != item_id:
        raise ValueError(f"card {card.item_id} does not belong to cell item {item_id}")
    stage = process_submission(raw, stop_reason, item_id=item_id, model=contestant, track=t,
                               layout=layout, th=th, meta=meta)
    outcome = Outcome(stage.outcome_code)
    cgx = vf = None
    cgx_path = v_path = None
    if outcome == Outcome.OK:
        pdf, render = compile_paths(layout, contestant.model_id, t, item_id)
        cgx = cpu_pool.run(extract_cgx, pdf, item_id=item_id, model_id=contestant.model_id, track=t, th=th)
        cgx_path, _ = write_cgx(cgx, layout, contestant.model_id, t, item_id)
        vf = adjudicate(card, cgx, render, crop_png, judges=judges, layout=layout, th=th, ledger=ledger,
                        allow_live=allow_live, provider_override=provider_override, phase=phase,
                        contestant_family=contestant.family)
        v_path = layout.verdicts(cgx.model_id, t, item_id)
    sc = score_cell(card, stage, outcome, vf, th, model_id=contestant.model_id, track=t)
    s_path, _ = write_score(layout, sc)
    final = finalize_submission(stage, layout, cgx_path=cgx_path, verdicts_path=v_path, scores_path=s_path,
                                judge_config_id=vf.judge_config_hash if vf else None)
    return CellResult(item_id=item_id, model_id=contestant.model_id, track=t, outcome=outcome.value,
                      score=sc.model_dump(mode="json"),
                      verdicts=vf.model_dump(mode="json", by_alias=True) if vf else None,
                      submission=final.model_dump(mode="json"),
                      cgx_fact_count=len(cgx.facts) if cgx else None)


def contestant_cell(layout: Layout, th, *, rec: dict, card: Card, crop: Path, ctx: str, track: str,
                    contestant: ModelSpec, judges: list[ModelSpec], ledger, allow_live: bool, provider_override,
                    judge_override, phase: str, access_date: str, provider_terms_version: str, output_release: str,
                    release_hash: str | None) -> CellResult:
    """One final-run (MS5) cell: the §9.1 contestant call, then `run_cell` (extraction, compile, CGX, adjudication,
    score). Shared by the sequential `run_model` and the hybrid scheduler (hmb.batch), so both run the
    same code; a cache miss raises LiveCallsDisabled (BatchDeferred under a batch router) before anything
    of the cell is written."""
    from .providers import call_model
    from .submission.prompts import build_contestant_request, contestant_prompt, decoding_params, image_encoding
    req = build_contestant_request(track, contestant, crop, rec["assets"]["crop_canonical_sha256"],
                                   ctx, th, phase=phase, cache_scope="contestant-v1")
    resp = call_model(req, layout, ledger, allow_live=allow_live, provider_override=provider_override)
    pid, _txt, psha = contestant_prompt(track, contestant)
    meta = {"dataset_version": rec["dataset_version"], "release_hash": release_hash,
            "model_version_string": resp.model_version_string, "access_date": access_date,
            "provider_terms_version": provider_terms_version, "output_release": output_release,
            "image_encoding": image_encoding(contestant, th, rec["assets"]["crop_canonical_sha256"]),
            "attempts": [a.model_dump() for a in resp.attempts], "thinking": resp.thinking,
            "decoding_params": decoding_params(contestant, th), "prompt_id": pid,
            "prompt_sha256": psha}
    return run_cell(card, resp.text, resp.stop_reason, item_id=rec["id"], contestant=contestant,
                    track=track, layout=layout, th=th, judges=judges, ledger=ledger,
                    allow_live=allow_live, crop_png=crop, meta=meta,
                    provider_override=judge_override, phase=phase)


def write_run_report(layout: Layout, name: str, th, cells: list[CellResult], extra: dict[str, Any]) -> Path:
    """Versioned report: every number is tagged with the hashes that produced it and the profile."""
    rep = {"report": name, "profile": th.profile.value, "thresholds_hash": th.hash,
           "harness_version": harness_version(th), "metric_hash": metric_hash(th),
           "judge_config_ids": sorted({c.submission.get("judge_config_id") for c in cells
                                       if c.submission.get("judge_config_id")}),
           "claim_status": ("engineering evidence only (fixture profile); not a benchmark score"
                            if not th.is_production else "production"),
           "cells": [c.summary() for c in cells], **extra}
    path = layout.reports / f"{name}.json"
    write_json_immutable(path, json.loads(json.dumps(rep, default=str)))
    return path


# --------------------------------------------------------------------------- MS5 model run
class RunRefused(RuntimeError):
    """A precondition of an official model run is not met (freeze, release, scoring window)."""


def eligible_items(layout: Layout, split: str) -> list[dict]:
    from .store import read_json
    d = layout.p("data", split)
    out = []
    for p in sorted(d.glob("hmb_*.json")) if d.exists() else []:
        rec = read_json(p)
        if rec.get("quality", {}).get("eligible"):
            out.append(rec)
    return out


def window_open_check(layout: Layout, th, judges: list[ModelSpec]) -> dict:
    """§10.9: the scoring window is open only if `drift-probe --window open` ran for every roster judge
    on the frozen triple set and each judge's agreement with the reference verdicts (recorded on the
    development set) is >= kappa_min."""
    from .hashing import sha256_file
    from .store import read_json
    if layout.p("manifests", "drift_probe", "close.json").exists():
        return {"open": False, "reason": "scoring window already closed (§10.9)"}
    op = layout.p("manifests", "drift_probe", "open.json")
    tp = layout.p("manifests", "drift_probe", "triples.json")
    if not op.exists() or not tp.exists():
        return {"open": False, "reason": "no drift-probe open record or triple set"}
    rec = read_json(op)
    if rec.get("profile") != th.profile.value:
        return {"open": False, "reason": f"window was opened under profile {rec.get('profile')!r}"}
    if rec.get("triples_sha256") != sha256_file(tp):
        return {"open": False, "reason": "open record was made on a different triple set"}
    refs = rec.get("references") or {}           # own development Q1 answers (replay probe, owner 2026-09-30)
    kmin = float(th.value("kappa_min"))
    per = {}
    for j in judges:
        ans = rec.get("answers", {}).get(j.model_id)
        if ans is None:
            return {"open": False, "reason": f"judge {j.model_id} was not probed at window open"}
        ref = refs.get(j.model_id)
        if not ref:
            return {"open": False, "reason": f"judge {j.model_id} has no recorded development Q1 vote on the triples"}
        per[j.model_id] = sum(ans.get(i) == v for i, v in ref.items()) / len(ref)
    ok = all(a >= kmin for a in per.values())
    return {"open": ok, "agreement_open": per, "kappa_min": kmin,
            "reason": None if ok else "agreement below kappa_min at window open"}


def development_probe_items(layout: Layout) -> list[dict]:
    """The items a pre-freeze development run scores (MS2 metering, MS3 variance probe; C42 fix 2026-09-26):
    the candidate records of declared development books whose card gates pass with at least one eligible fact.
    They have no final item record yet; `items-finalize` later writes them to the development split."""
    from .acquisition.drivers import development_books
    from .cards.batch import candidates_dir, card_gates_path
    from .store import read_json
    dev = development_books(layout)
    out = []
    d = candidates_dir(layout)
    for p in sorted(d.glob("hmb_*.json")) if d.exists() else []:
        rec = read_json(p)
        gp = card_gates_path(layout, rec["id"])
        if (rec.get("source") or {}).get("book_id") not in dev or not gp.exists() or not layout.card(rec["id"]).exists():
            continue
        g = read_json(gp)
        if not g.get("exclusion_reasons") and sum((g.get("eligible_counts") or {}).values()) > 0:
            out.append(rec)
    return out


def require_official_run(layout: Layout, th, contestant: ModelSpec, judges: list[ModelSpec], *, fake: bool,
                         split: str = "test") -> None:
    """Production preconditions of every contestant run (§15.5, §17.3, §10.9): the locked roster entry,
    every level frozen, a releasable production release, an open scoring window, no fake provider."""
    from .roster import locked_roster_problems
    from .store import read_json
    if not th.is_production:
        return
    lock = [p for p in locked_roster_problems([contestant]) if contestant.model_id in p]
    if lock:
        raise RunRefused("contestant is not the locked roster entry: " + "; ".join(lock))
    if th.is_prefreeze:           # MS2/MS3 development run: development books only, provisional levels
        if split != "development":
            raise RunRefused("a pre-freeze context runs contestants only on development books (--split development)")
        if fake:
            raise RunRefused("fake providers are refused under the production profile")
        return
    missing = th.missing_for_production()
    if missing:
        raise RunRefused(f"levels not frozen: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if not layout.release_manifest.exists():
        raise RunRefused("no release manifest (release-build) — cards and thresholds must be frozen first")
    rel = read_json(layout.release_manifest)
    if rel.get("profile") != "production" or rel.get("releasable") is not True:
        raise RunRefused("release manifest is not a releasable production release")
    w = window_open_check(layout, th, judges)
    if not w["open"]:
        raise RunRefused(f"scoring window not open (§10.9): {w['reason']}")
    if fake:
        raise RunRefused("fake providers are refused under the production profile")


def run_t3_probe_model(layout: Layout, th, *, contestant: ModelSpec, judges: list[ModelSpec], split: str,
                       allow_live: bool, access_date: str, provider_terms_version: str,
                       output_release: str = "public", provider_override=None, judge_override=None,
                       phase: str = "FINAL", limit: int | None = None) -> dict:
    """§13.6 T3 text-only recall probe for one contestant over the eligible items: the call and its
    submission stage are `submission.process.run_t3_probe` (context text, no image, prompt
    reconstruction-T3-v2, item id `<item>__t3`, track B), and the cell is then judged against the
    item's card and scored through `run_cell`. A diagnostic condition, never an A/B table: score-report
    reports it only in `t3_probe_s13_6`. Resumable; same preconditions as `run_model`."""
    from .ledger import Ledger
    from .providers import LiveCallsDisabled
    from .schemas import Card
    from .store import read_json
    from .submission.process import run_t3_probe
    from .submission.prompts import t3_item_id
    require_official_run(layout, th, contestant, judges, fake=provider_override is not None or judge_override is not None,
                         split=split)
    release_hash = read_json(layout.release_manifest)["release_hash"] if layout.release_manifest.exists() else None
    ledger = Ledger(layout)
    results, stopped = [], None
    for rec in eligible_items(layout, split)[: limit or None]:
        iid, tid = rec["id"], t3_item_id(rec["id"])
        if layout.submission_record(model_dir_name(contestant.model_id), "B", tid).exists():
            results.append({"item_id": tid, "resumed": True})
            continue
        ctx_path = layout.context_text(iid)
        ctx = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else ""
        meta = {"dataset_version": rec["dataset_version"], "release_hash": release_hash, "access_date": access_date,
                "provider_terms_version": provider_terms_version, "output_release": output_release}
        try:
            stage = run_t3_probe(item_id=iid, model=contestant, context_text=ctx, layout=layout, th=th, ledger=ledger,
                                 allow_live=allow_live, phase=phase, cache_scope="contestant-v1", meta=meta,
                                 provider_override=provider_override)
        except LiveCallsDisabled as e:
            stopped = f"{type(e).__name__}: {e}"
            break
        card = Card.model_validate(read_json(layout.card(iid))).model_copy(update={"item_id": tid})
        raw = (layout.root / stage.raw_output).read_bytes().decode("utf-8")
        res = run_cell(card, raw, stage.stop_reason, item_id=tid, contestant=contestant, track="B", layout=layout,
                       th=th, judges=judges, ledger=ledger, allow_live=allow_live, crop_png=layout.crop_canonical(iid),
                       meta={**meta, "model_version_string": stage.model_version_string,
                             "image_encoding": stage.image_encoding, "prompt_id": stage.prompt_id,
                             "prompt_sha256": stage.prompt_sha256},
                       provider_override=judge_override, phase=phase)
        results.append(res.summary())
    return {"model_id": contestant.model_id, "condition": "T3 text-only probe (§13.6)", "split": split,
            "profile": th.profile.value, "cells": results, "stopped": stopped}


def run_model(layout: Layout, th, *, contestant: ModelSpec, judges: list[ModelSpec], tracks: list[str],
              split: str, allow_live: bool, provider_override=None, judge_override=None,
              phase: str = "FINAL", access_date: str, provider_terms_version: str,
              output_release: str = "public", limit: int | None = None) -> dict:
    """One output per item per track (§9.1). Resumable: a finalized cell is never re-run.

    Production preconditions (§15.5, §17.3, §10.9): every level frozen; a release manifest exists;
    the model's scoring window was opened with a passing drift probe. Spending is metered, not capped
    (owner decision 2026-09-23); LiveCallsDisabled stops the run with the completed cells intact.
    """
    from .ledger import Ledger
    from .providers import LiveCallsDisabled
    from .store import read_json
    from .roster import accepts_images
    if "A" in tracks and not accepts_images(contestant):
        raise RunRefused(f"{contestant.model_id} is text-only: the direct-figure track (A) is unavailable to it")
    require_official_run(layout, th, contestant, judges, fake=provider_override is not None or judge_override is not None,
                         split=split)
    if th.is_prefreeze:
        phase = th.context or phase
    cap = int(th.value("common_output_cap_tokens"))
    if int(th.value("contestant_max_tokens")) > cap:
        raise RunRefused(f"contestant_max_tokens {th.value('contestant_max_tokens')} exceeds the common output cap {cap}")
    release_hash = read_json(layout.release_manifest)["release_hash"] if layout.release_manifest.exists() else None
    ledger = Ledger(layout)
    done, results, stopped = 0, [], None
    items = (development_probe_items(layout) if th.is_prefreeze else eligible_items(layout, split))[: limit or None]
    for rec in items:
        iid = rec["id"]
        from .schemas import Card
        card = Card.model_validate(read_json(layout.card(iid)))
        crop = layout.crop_canonical(iid)
        ctx_path = layout.context_text(iid)
        ctx = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else ""
        for t in tracks:
            srec = layout.submission_record(model_dir_name(contestant.model_id), t, iid)
            if srec.exists():                                 # finalized: resume skips it, report keeps it
                prev = read_json(srec)
                sc = read_json(layout.root / prev["scores"]) if prev.get("scores") else {}
                results.append({"item_id": iid, "model_id": contestant.model_id, "track": t,
                                "outcome": prev["outcome_code"], "S": sc.get("S"),
                                "evaluator_failure": sc.get("evaluator_failure"), "resumed": True})
                continue
            try:
                res = contestant_cell(layout, th, rec=rec, card=card, crop=crop, ctx=ctx, track=t,
                                      contestant=contestant, judges=judges, ledger=ledger, allow_live=allow_live,
                                      provider_override=provider_override, judge_override=judge_override,
                                      phase=phase, access_date=access_date,
                                      provider_terms_version=provider_terms_version,
                                      output_release=output_release, release_hash=release_hash)
            except LiveCallsDisabled as e:
                stopped = f"{type(e).__name__}: {e}"
                break
            results.append(res.summary())
            done += 1
        if stopped:
            break
    summary = {"model_id": contestant.model_id, "tracks": tracks, "split": split, "profile": th.profile.value,
               "release_hash": release_hash, "cells_run": done, "cells_total": len(results), "stopped": stopped,
               "ledger_spent_usd_estimate": ledger.spent(phase), "cells": results,
               "note": "ledger figures are estimates from the dated rate card, not billing guarantees"}
    from .store import write_mutable
    out = layout.reports / f"model_run_{model_dir_name(contestant.model_id)}.json"
    write_mutable(out, (json.dumps(summary, indent=1, default=str) + "\n").encode())
    from .config import prefreeze_record
    prefreeze_record(th, layout, f"model-run {contestant.model_id}", [out], inputs={"n_items": len(items)})
    return summary


def close_model_run(layout: Layout, th, *, contestant: ModelSpec, tracks: list[str],
                    output_release: str = "public") -> Path:
    """§15.3 run manifest scores/<model>/run.json, written once after the scoring window closed.

    Requires the drift-probe close result (manifests/drift_probe/result.json). Under the production
    profile the window must have passed (§10.9); a failed window refuses (re-score or bump version).
    """
    from datetime import datetime, timezone
    from .hashing import H, sha256_file
    from .manifests import build_run_manifest, write_run_manifest
    from .providers import effective_inference
    from .roster import route_record
    from .store import read_json
    from .submission.prompts import contestant_prompt
    res_p = layout.p("manifests", "drift_probe", "result.json")
    if not res_p.exists():
        raise RunRefused("scoring window not closed: run drift-probe --window close first")
    res = read_json(res_p)
    if th.is_production and res.get("pass") is not True:
        raise RunRefused("drift probe failed at window close: the window is invalid (§10.9)")
    if th.is_production:
        if not layout.release_manifest.exists():
            raise RunRefused("missing release manifest")
        rel_doc = read_json(layout.release_manifest)
        if rel_doc.get("profile") != "production" or rel_doc.get("releasable") is not True:
            raise RunRefused("release manifest is not a releasable production release")
    mdir = model_dir_name(contestant.model_id)
    vfiles = sorted(v for t in tracks for v in layout.p("verdicts", mdir, t).glob("*.json"))
    verdicts_hash = H([[str(v.relative_to(layout.root)).replace("\\", "/"), sha256_file(v)] for v in vfiles])
    subs = sorted(x for t in tracks for x in layout.p("submissions", mdir, t).glob("*.json"))
    recs = [read_json(x) for x in subs]
    if not recs:
        raise RunRefused(f"no finalized submissions for {contestant.model_id}")
    versions = sorted({r["model_version_string"] for r in recs})
    model_record = {"model_id": contestant.model_id, "provider_version": versions, "family": contestant.family,
                    "snapshot": contestant.snapshot, "access_date": sorted({r["access_date"] for r in recs}),
                    "decoding": recs[0]["decoding_params"],
                    "attempts": {"cells": len(recs),
                                 "max_attempts": max(len(r.get("attempts") or []) for r in recs)},
                    "effort": contestant.effort, "input_mode": contestant.input_mode,
                    "prompts": {t: contestant_prompt(t, contestant)[2] for t in tracks},
                    "route": route_record(contestant),
                    "effective_inference": effective_inference(contestant, int(th.value("contestant_max_tokens")))}
    funnel_rate = None
    fr = layout.reports / "funnel_report.json"
    if fr.exists():
        stages = (read_json(fr).get("stages") or {})
        s4, s8 = stages.get("S4"), stages.get("S8")
        funnel_rate = (s8 / s4) if isinstance(s4, (int, float)) and s4 and isinstance(s8, (int, float)) else None

    def mtime(name: str) -> str:
        f = layout.p("manifests", "drift_probe", name)
        return datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    window = {"open": mtime("open.json"), "close": mtime("close.json"),
              "source": "drift-probe record times"}
    drift = {"open_agreement": res.get("open_agreement"), "close_agreement": res.get("close_agreement"),
             "kappa_min": res.get("kappa_min"), "pass": res.get("pass"),
             "engineering_valid": res.get("engineering_valid"), "profile": th.profile.value}
    rel = read_json(layout.release_manifest)["release_hash"] if layout.release_manifest.exists() else "0" * 64
    run = build_run_manifest(release_hash_value=rel, model_record=model_record,
                             prompt_sha256=H(sorted(model_record["prompts"].values())), scoring_window=window,
                             drift_probe=drift, verdicts_hash=verdicts_hash, funnel_eligibility_rate=funnel_rate,
                             output_release=output_release)
    run["profile"] = th.profile.value
    return write_run_manifest(layout, contestant.model_id, run)


# --------------------------------------------------------------------------- CLI
def _cmd_roster_check(args) -> int:
    from .roster import check_roster, load_roster
    probs = check_roster(load_roster(args.roster))
    print(json.dumps({"roster": args.roster, "problems": probs, "ok": not probs}, indent=1))
    return 0 if not probs else 2


def _cmd_model_run(args) -> int:
    from .grading.cli import ReportRefused, resolve_thresholds
    from .roster import load_roster
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds(args.thresholds, layout, allow_prefreeze=True)
    except ReportRefused as e:
        print(f"REFUSED: {e}")
        return 2
    r = load_roster(args.roster)
    contestant = next((s for s in r["contestant"] if s.model_id == args.model_id), None)
    if contestant is None:
        print(f"{args.model_id} is not a roster contestant")
        return 2
    try:
        out = run_model(layout, th, contestant=contestant, judges=r["judge"], tracks=args.track, split=args.split,
                        allow_live=args.allow_live, access_date=args.access_date,
                        provider_terms_version=args.provider_terms_version, output_release=args.output_release,
                        limit=args.limit)
    except RunRefused as e:
        print(f"REFUSED: {e}")
        return 3
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1, default=str))
    return 4 if out["stopped"] else 0


def _cmd_t3_probe(args) -> int:
    from .grading.cli import ReportRefused, resolve_thresholds
    from .roster import load_roster
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds(args.thresholds, layout)
    except ReportRefused as e:
        print(f"REFUSED: {e}")
        return 2
    r = load_roster(args.roster)
    contestant = next((s for s in r["contestant"] if s.model_id == args.model_id), None)
    if contestant is None:
        print(f"{args.model_id} is not a roster contestant")
        return 2
    try:
        out = run_t3_probe_model(layout, th, contestant=contestant, judges=r["judge"], split=args.split,
                                 allow_live=args.allow_live, access_date=args.access_date,
                                 provider_terms_version=args.provider_terms_version,
                                 output_release=args.output_release, limit=args.limit)
    except RunRefused as e:
        print(f"REFUSED: {e}")
        return 3
    print(json.dumps({k: v for k, v in out.items() if k != "cells"} | {"n_cells": len(out["cells"])}, indent=1))
    return 4 if out["stopped"] else 0


def _cmd_model_close(args) -> int:
    from .grading.cli import ReportRefused, resolve_thresholds
    from .roster import load_roster
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds(args.thresholds, layout)
    except ReportRefused as e:
        print(f"REFUSED: {e}")
        return 2
    c = next((s for s in load_roster(args.roster)["contestant"] if s.model_id == args.model_id), None)
    if c is None:
        print(f"{args.model_id} is not a roster contestant")
        return 2
    try:
        p = close_model_run(layout, th, contestant=c, tracks=args.track, output_release=args.output_release)
    except RunRefused as e:
        print(f"REFUSED: {e}")
        return 3
    print(f"wrote {p}")
    return 0


def register_cli(sub) -> None:
    p = sub.add_parser("roster-check", help="validate configs/roster.yaml against §2.7/§7.3/§10.9")
    p.add_argument("roster")
    p.set_defaults(func=_cmd_roster_check)
    p = sub.add_parser("model-run", help="MS5: one contestant over eligible items (resumable; ledger-gated)")
    p.add_argument("--roster", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--track", action="append", choices=["A", "B"], required=True)
    p.add_argument("--split", default="test", choices=["development", "test"])
    p.add_argument("--thresholds", default=None)
    p.add_argument("--allow-live", action="store_true")
    p.add_argument("--access-date", required=True)
    p.add_argument("--provider-terms-version", required=True)
    p.add_argument("--output-release", default="public", choices=["public", "withheld"])
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_cmd_model_run)
    p = sub.add_parser("t3-probe", help="§13.6 T3 text-only probe for one contestant (context text, no image; "
                                        "diagnostic, never an A/B table; same preconditions as model-run)")
    p.add_argument("--roster", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--split", default="test", choices=["development", "test"])
    p.add_argument("--thresholds", default=None)
    p.add_argument("--allow-live", action="store_true")
    p.add_argument("--access-date", required=True)
    p.add_argument("--provider-terms-version", required=True)
    p.add_argument("--output-release", default="public", choices=["public", "withheld"])
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_cmd_t3_probe)
    p = sub.add_parser("model-close", help="§15.3 run manifest after the scoring window closed")
    p.add_argument("--roster", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--track", action="append", choices=["A", "B"], required=True)
    p.add_argument("--thresholds", default=None)
    p.add_argument("--output-release", default="public", choices=["public", "withheld"])
    p.set_defaults(func=_cmd_model_close)
