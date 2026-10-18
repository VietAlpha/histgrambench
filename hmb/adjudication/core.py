"""§10 adjudication of one valid cell: `adjudicate(card, cgx, render_png, crop_png, ...)`.

Flow: alignment (+ Q3 on ambiguous entities) -> deterministic verdicts for geo facts and EXEMPT ->
Q1 two judges on model facts, a third when the two do not decide -> immutable verdict file. Every cell
gets a verdict for every eligible fact: a fact the evaluator could not decide (a tie, or a judge failure)
is INDETERMINATE and the cell records `evaluator_failure` when a judge call failed. Cells with only geo
facts and no ambiguous entity make no judge call. A stored verdict file is returned as-is (no re-judging
after freeze). The extras channel (ladder and Q2) was removed 2026-09-26: it never entered the score.
"""
from __future__ import annotations

from pathlib import Path

from hmb.ontology import PREDICATES, entity_args
from hmb.cards.normalize import normalize_label
from hmb.schemas import Card, CgxRecord, FactVerdict, ModelSpec, VerdictFile
from hmb.store import read_json, safe_model_dir, write_json_immutable
from hmb.vocab import EvaluatorFailure, JudgeQuestion, Profile, Verdict

from hmb import cpu_pool

from . import geom as G
from .aggregate import decide, recall_verdict
from .alignment import Alignment, align, card_scope
from .geo import CgxIndex, map_args, route
from .judges import BItem, CellFailure, JudgeCtx, Panel, Q3PartialFailure, run_binary, run_q3
from .serialize import check_unique, fact_refs, fact_statement, names_for_card, object_lines, q3_question
from .protocol import JudgeConfigError, judge_cache_scope, judge_config_hash, select_judges


class AdjudicationError(RuntimeError):
    pass


def _letter(pred: str) -> str:
    return PREDICATES[pred].category


def _profile(card: Card, th) -> str:
    if th.is_production and card.profile != Profile.production.value:
        raise AdjudicationError("production adjudication refuses a fixture card")
    return Profile.production.value if th.is_production else Profile.fixture.value


def _contestant_family(layout, cgx: CgxRecord) -> str | None:
    """From the compile-stage record (the final submission record is written only after scoring)."""
    from hmb.submission.process import model_dir_name
    p = layout.compile_dir(model_dir_name(cgx.model_id), str(cgx.track), cgx.item_id) / "submission_stage.json"
    if p.exists():
        fam = read_json(p).get("model_family")
        if fam:
            return fam
    if str(cgx.track) == "C":
        return None          # generated fixture render (track C): not a contestant
    raise AdjudicationError(f"no model_family in {p}; pass contestant_family= (§10.9)")


# ------------------------------------------------------------------ deterministic verdicts
def _label_attachments(index: CgxIndex, th) -> list[tuple[str, str | None, str]]:
    """(normalized text, owning primitive, raw text) for every extracted NAME label run. The owner
    is the extractor's `bound_prim` (shared §10.3 rule); a run without one is bound here with the
    same shared function."""
    runs = [o for o in index.cgx.labels if o.get("text") and o.get("kind", "name") == "name"]
    need = [k for k, o in enumerate(runs) if o.get("bound_prim") is None and o.get("bbox")]
    D = float(index.cgx.diagonal_D)
    R, eps = float(th.value("label_bind_frac")) * D, float(th.value("eps_on_frac")) * D
    got = index.geometry.label_owners([G.bbox_center(runs[k]["bbox"]) for k in need], R, eps) if need else []
    computed = dict(zip(need, got))
    return [(normalize_label(o["text"]), o.get("bound_prim") or computed.get(k), o["text"])
            for k, o in enumerate(runs)]


def _label_verdict(f, al: Alignment, attach) -> tuple[Verdict, dict]:
    t, e = f.args
    nt = normalize_label(t)
    p = al.binding[e]
    here = [(n, raw) for n, own, raw in attach if own == p]
    if any(n == nt for n, _ in here):
        return Verdict.PRESENT, {"label_match": {"strict": True, "error": None}}
    if any(n == nt for n, _, _ in attach):
        return Verdict.ABSENT, {"label_match": {"strict": False, "error": "attachment"}}
    if here:
        return Verdict.CONTRADICTED, {"label_match": {"strict": False, "error": "transcription",
                                                      "extracted": sorted(raw for _, raw in here)}}
    return Verdict.ABSENT, {"label_match": {"strict": False, "error": "missing"}}


def _entity_verdict(f, al: Alignment, index: CgxIndex) -> tuple[Verdict, str]:
    e = f.args[0]
    if e in al.binding:
        if PREDICATES[f.predicate].arity == 1:
            return Verdict.PRESENT, "bound"
        m = map_args(f.predicate, f.args, al.binding)
        ok, basis = index.holds(f.predicate, m)
        if ok:
            return Verdict.PRESENT, basis
        if not any(k[0] == f.predicate and k[1][0] == m[0] for k in index.closed):
            g = index.prims[m[0]].geom or {}
            if f.predicate == "axis":
                return Verdict.PRESENT, "bound(type only)"
            if f.predicate == "polygon" and len(g.get("vertices", [])) == f.args[1]:
                return Verdict.PRESENT, "geom.vertices"
        return Verdict.ABSENT, basis
    if e in al.absent:
        return Verdict.ABSENT, "entity absent"
    return Verdict.INDETERMINATE, "alignment ambiguous"


def deterministic_verdicts(card: Card, al: Alignment, index: CgxIndex, th
                           ) -> tuple[dict[str, FactVerdict], list, dict[str, bool]]:
    """Returns (verdicts decided without a judge, facts routed to Q1, e_incompatible per routed fact)."""
    _, facts = card_scope(card)
    attach = _label_attachments(index, th)
    # EXEMPT (b) needs an argument entity whose own eligible entity fact is decided without a judge (it is then
    # ABSENT from the extractor), the condition `grading.cell._check_exempt_is_deterministic` enforces. A
    # model-class entity (`curve`, …) goes to Q1 even when alignment finds no primitive for it, so its
    # absence never exempts: those facts stay in the denominator (2026-09-26; one live cell aborted on
    # on(p, curve) before this).
    ent_fact_elig = {f.args[0] for f in facts if PREDICATES[f.predicate].fclass.value == "entity"
                     and route(f) != "judges"}
    out: dict[str, FactVerdict] = {}
    to_judge, e_inc = [], {}
    for f in sorted(facts, key=lambda f: f.fact_id):
        ea = entity_args(f.predicate, f.args)
        bind = {x: al.binding[x] for x in ea if x in al.binding}
        is_entity = PREDICATES[f.predicate].fclass.value == "entity"
        if not is_entity:
            counted = sorted(x for x in ea if x in al.absent and x in ent_fact_elig)   # deterministic absence only
            if counted:
                out[f.fact_id] = FactVerdict(fact_id=f.fact_id, verdict=Verdict.EXEMPT, source="deterministic",
                                             binding=bind, evidence=f"argument entity absent: {counted}")
                continue
            det_bound = al.deterministically_bound()
            if f.anchor == "B" and any(x not in det_bound and x not in al.not_aligned for x in ea):
                out[f.fact_id] = FactVerdict(fact_id=f.fact_id, verdict=Verdict.EXEMPT, source="deterministic",
                                             binding=bind, evidence="anchor B without unique binding")
                continue
        if route(f) == "judges" or any(x in al.not_aligned for x in ea):
            to_judge.append(f)
            m = map_args(f.predicate, f.args, al.binding)
            e_inc[f.fact_id] = bool(m is not None and index.e_incompatible(f.predicate, m))
            continue
        q3_abs = sorted(x for x in ea if x in al.q3_absent)
        q3_und = sorted(x for x in ea if x in al.q3_asked and x in al.ambiguous)
        if q3_abs:
            out[f.fact_id] = FactVerdict(fact_id=f.fact_id, verdict=Verdict.ABSENT, source="judges", binding=bind,
                                         evidence=f"Q3 judged absent: {q3_abs}")
            continue
        if q3_und:                  # forced decisions (2026-09-27): no usable Q3 binding -> the fact goes to Q1
            to_judge.append(f)
            e_inc[f.fact_id] = False
            continue
        if is_entity:
            v, basis = _entity_verdict(f, al, index)
            out[f.fact_id] = FactVerdict(fact_id=f.fact_id, verdict=v, source="extractor", binding=bind,
                                         evidence=f"basis={basis}")
            continue
        if f.predicate == "label":
            e = f.args[1]
            if e not in al.binding:
                v = Verdict.ABSENT if e in al.absent else Verdict.INDETERMINATE
                out[f.fact_id] = FactVerdict(fact_id=f.fact_id, verdict=v, source="extractor", binding=bind,
                                             evidence="label entity unbound")
                continue
            v, info = _label_verdict(f, al, attach)
            out[f.fact_id] = FactVerdict(fact_id=f.fact_id, verdict=v, source="extractor",
                                         binding={**bind, **info}, evidence="basis=labels")
            continue
        m = map_args(f.predicate, f.args, al.binding)
        if m is None:
            v = Verdict.ABSENT if any(x in al.absent for x in ea) else Verdict.INDETERMINATE
            out[f.fact_id] = FactVerdict(fact_id=f.fact_id, verdict=v, source="extractor", binding=bind,
                                         evidence="argument unbound")
            continue
        ok, basis = index.holds(f.predicate, m)
        if ok:
            v = Verdict.PRESENT
        else:
            v = Verdict.CONTRADICTED if index.e_incompatible(f.predicate, m) else Verdict.ABSENT
        out[f.fact_id] = FactVerdict(fact_id=f.fact_id, verdict=v, source="extractor", binding=bind,
                                     evidence=f"basis={basis}")
    return out, to_judge, e_inc


# ------------------------------------------------------------------ Q3 inputs
def _to_canvas(index: CgxIndex, th):
    ext = index.geometry.extent()                     # internal frame, y up
    k = index.geometry.px_per_pt(int(th.value("canvas_px")), float(th.value("max_upscale")),
                                 int(th.value("render_dpi_candidate")))
    if ext is None or k is None:
        return None
    C = int(th.value("canvas_px"))
    x0, y0, x1, y1 = ext
    ox, oy = (C - (x1 - x0) * k) / 2.0, (C - (y1 - y0) * k) / 2.0
    return lambda x, y: (ox + (x - x0) * k, oy + (y1 - y) * k)


def _object_lines(prims: list[str], index: CgxIndex, th) -> tuple[str, dict[str, str]]:
    """Q3 candidate objects o1.. in the given (sorted) primitive order; the text is `serialize.object_lines`."""
    f = _to_canvas(index, th)
    C = float(th.value("canvas_px"))
    rows, ids = [], {}
    for n, pid in enumerate(prims, 1):
        oid = f"o{n}"
        ids[oid] = pid
        pos = None
        if f is not None:
            cx, cy = f(*index.geometry.anchor(pid))
            pos = (cx / C, cy / C)
        rows.append((oid, index.prims[pid].type, pos))
    return object_lines(rows), ids


# ------------------------------------------------------------------ main entry
def _align_cell(card: Card, cgx: CgxRecord, th) -> Alignment:
    return align(card, CgxIndex(cgx, th), th)


def _deterministic_cell(card: Card, al: Alignment, cgx: CgxRecord, th):
    return deterministic_verdicts(card, al, CgxIndex(cgx, th), th)


def adjudicate(card: Card, cgx: CgxRecord, render_png: Path, crop_png: Path | None, *,
               judges: list[ModelSpec], layout, th, ledger, allow_live: bool, provider_override=None,
               phase: str = "FINAL", configuration: str = "primary", contestant_family: str | None = None) -> VerdictFile:
    profile = _profile(card, th)
    if card.item_id != cgx.item_id:
        raise AdjudicationError("card and CGX record are for different items")
    cfg = judge_config_hash(judges, cgx.extractor_version, th)
    track = str(cgx.track)
    vpath = layout.verdicts(cgx.model_id, track, cgx.item_id)
    if vpath.exists():
        vf = VerdictFile.model_validate(read_json(vpath))
        if vf.judge_config_hash != cfg:
            raise AdjudicationError("stored verdicts use another judge configuration; bump the version")
        if vf.card_uid is None:
            if th.is_production:
                raise AdjudicationError(f"{vpath}: stored verdict file has no card_uid; refused in production (B6 #10)")
            import warnings
            warnings.warn(f"{vpath}: stored verdict file has no card_uid; reused under the fixture profile only")
        elif vf.card_uid != card.card_uid:
            raise AdjudicationError(f"{vpath}: stored verdicts judged card {vf.card_uid}, not {card.card_uid}")
        return vf
    base = dict(item_id=cgx.item_id, card_uid=card.card_uid, model_id=cgx.model_id, track=track,
                judge_config_hash=cfg, profile=profile)

    def finish(failure, facts, ambiguous=()) -> VerdictFile:
        vf = VerdictFile(**base, evaluator_failure=failure, facts=sorted(facts, key=lambda v: v.fact_id),
                         alignment_ambiguous=sorted(ambiguous))
        write_json_immutable(vpath, vf.model_dump(mode="json", by_alias=True))
        return vf

    # With a CPU pool (hmb.cpu_pool) the pure geometric steps run in a worker process, which rebuilds the index
    # from the CGX record; neither step mutates the alignment it is given or the index.
    index = None if cpu_pool.active() else CgxIndex(cgx, th)
    ents, facts = card_scope(card)
    if not facts:
        raise AdjudicationError("card has no eligible facts (item not in I, §11.12)")
    if contestant_family is None:
        contestant_family = _contestant_family(layout, cgx)
    panel = Panel(select_judges(judges, builder_families=card.family_map.values(),
                                contestant_family=contestant_family, configuration=configuration))

    def ctx() -> JudgeCtx:
        if len(panel.judges) < 2:
            raise JudgeConfigError("fewer than two eligible judge families for this cell (§10.6, §10.9)")
        return JudgeCtx(layout, th, ledger, allow_live, provider_override, phase,
                        judge_cache_scope(judges, cgx.extractor_version, th))

    names = names_for_card(card)                  # one deterministic identity map for Q1 and Q3
    al = cpu_pool.run(_align_cell, card, cgx, th) if index is None else align(card, index, th)
    al.det_bound = set(al.binding)
    ambiguous_q3 = sorted(al.ambiguous)
    failures: list[str] = []
    # Q3 on ALIGNMENT_AMBIGUOUS entities (a failed Q3 leaves them ambiguous: their facts are INDETERMINATE)
    try:
        if al.ambiguous:
            al.q3_asked = set(al.ambiguous)
            cand = sorted({p for e in al.ambiguous for p in al.candidates.get(e, [])})
            objects_text, oid = _object_lines(cand, index or CgxIndex(cgx, th), th)
            back = {v: k for k, v in oid.items()}
            ent_text = names.context(ents)
            qs = [(e, q3_question(names, e, [back[p] for p in al.candidates.get(e, [])]))
                  for e in sorted(al.ambiguous)]
            try:
                answers, _ = run_q3(qs, entities_text=ent_text, objects_text=objects_text, render=Path(render_png),
                                    ctx=ctx(), panel=panel,
                                    options={e: [back[p] for p in al.candidates.get(e, [])] for e in al.ambiguous})
            except Q3PartialFailure as qf:          # a failed slot keeps every answer two judges agreed on
                failures.append(qf.reason)
                answers = {e: a for e, a in qf.answers.items() if a is not None}
            claims: dict[str, list[str]] = {}
            for e, a in answers.items():
                if a == "none":                       # judge-derived: never feeds EXEMPT
                    al.ambiguous.discard(e)
                    al.q3_absent.add(e)
                elif a in oid and oid[a] in al.candidates.get(e, []):
                    claims.setdefault(oid[a], []).append(e)
            for p, es in claims.items():
                if len(es) == 1 and p not in al.binding.values():
                    al.binding[es[0]] = p
                    al.method[es[0]] = "q3"
                    al.ambiguous.discard(es[0])
    except CellFailure as cf:
        failures.append(cf.reason)
    write_json_immutable(layout.p("adjudication", "alignment", safe_model_dir(cgx.model_id), track, f"{cgx.item_id}.json"),
                         al.to_json())

    verdicts, to_judge, e_inc = (cpu_pool.run(_deterministic_cell, card, al, cgx, th) if index is None
                                 else deterministic_verdicts(card, al, index, th))
    # Q1 on model-class facts (a failed Q1 leaves them INDETERMINATE)
    try:
        if to_judge:
            check_unique(to_judge, names)          # distinct facts never share one wording
            items = [BItem(key=f.fact_id, text=fact_statement(f, names), refs=fact_refs(f)) for f in to_judge]
            res = run_binary(JudgeQuestion.Q1, items, images=[Path(render_png)], ctx=ctx(), panel=panel,
                             seed=f"{cgx.item_id}|{cgx.model_id}|{track}|Q1", names=names)
            failures += [r["reason"] for r in res.failures]   # decided facts stand; only undecided ones are open
            for f in to_judge:
                v = recall_verdict(res.decision[f.fact_id], e_inc[f.fact_id])
                fc = res.failed_calls.get(f.fact_id)
                verdicts[f.fact_id] = FactVerdict(
                    fact_id=f.fact_id, verdict=v, source="judges", votes=res.votes[f.fact_id],
                    evidence="forced: earliest panel judge's answer" if f.fact_id in res.forced else None,
                    binding={"args": {x: al.binding[x] for x in entity_args(f.predicate, f.args) if x in al.binding},
                             "e_incompatible": e_inc[f.fact_id],
                             **({"failed_judge_calls": fc} if fc else {})})
    except CellFailure as cf:
        failures.append(cf.reason)
    for f in facts:                                 # every eligible fact gets exactly one verdict
        if f.fact_id not in verdicts:
            verdicts[f.fact_id] = FactVerdict(fact_id=f.fact_id, verdict=Verdict.INDETERMINATE, source="judges",
                                              evidence="not decided: judge call failed")
    failure = None                                  # §10.8: refusal only if every failed call refused
    if failures:
        failure = EvaluatorFailure.judge_refusal if all(f == EvaluatorFailure.judge_refusal.value
                                                        for f in failures) else EvaluatorFailure.judge_failure
    return finish(failure, [verdicts[f.fact_id] for f in facts], ambiguous_q3)


# ------------------------------------------------------------------ published re-derivations
def recompute_without_family(vf: VerdictFile, card: Card, family: str) -> dict[str, str]:
    """Leave-one-judge-family-out verdicts from stored votes (§10.9, §13.5, G6); no new calls. Judged
    facts lose that family's votes and are re-decided by a majority of the remaining votes
    (`min_concurring = 1`, since at most two votes remain); a tie or no vote is INDETERMINATE."""
    out = {}
    for fv in vf.facts:
        if fv.source != "judges" or not fv.votes:
            out[fv.fact_id] = str(fv.verdict)
            continue
        d = decide((v.positive for v in fv.votes if v.judge_family != family), min_concurring=1)
        e_inc = bool((fv.binding or {}).get("e_incompatible"))
        out[fv.fact_id] = recall_verdict(d, e_inc).value
    return out
