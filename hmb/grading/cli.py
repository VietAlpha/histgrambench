"""Scaffold 4 reporting commands (owner: Opus 4).

  hmb score-report    --models M.. --track A [--track B] [--split test]  -> reports/scores_<split>_<release>.json

It reads the stored score / verdict / card files under the root and make NO model calls. Every
report carries the profile and the hashes that produced it (thresholds, metric, judge configs,
release). `score-report` also writes the `gate_evidence` block (G1 N, G2, G6, G7, G8) that
`gates-evaluate --scores` reads, so no gate input is assembled by hand; the report itself never emits a
gate result. One split is reported at a time (default test; development is the MS3 variance probe). Only
historical tracks A/B are accepted; a Track C record found among them is refused.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from hmb import stats
from hmb.config import CalibrationMissing, fixture_thresholds, load_thresholds
from hmb.schemas import Card, FactVerdict, ScoreRecord, VerdictFile
from hmb.store import Layout, read_json, safe_model_dir, write_json_immutable
from hmb.vocab import Outcome

from .cell import GradingError, metric_hash, score_cell
from .tables import (ceiling_flag, complete_items, context_effect, headline, model_track_table,
                     separation_evidence)

REPORTED = ("A", "B")
# §13.6 text-only probe prompt ids (hmb.submission.prompts.T3_PROMPT_ID); v1 records predate 2026-09-26
T3_PROMPT_IDS = ("reconstruction-T3-v1", "reconstruction-T3-v2")
T3_SUFFIX = "__t3"


class ReportRefused(RuntimeError):
    pass


# ------------------------------------------------------------------ loading
FIXTURE_BANNER = "*** FIXTURE PROFILE (explicit --thresholds fixture): engineering evidence only, not a benchmark score ***"


def resolve_thresholds(arg: str | None, layout: Layout, *, allow_prefreeze: bool = False):
    """--thresholds: a path, or 'fixture' only when passed explicitly (banner printed). Default:
    <root>/thresholds.json; if it does not exist the command refuses (never a silent fixture). A pre-freeze
    context is accepted only by the development-run commands (`model-run --split development`, `dev-sizing`;
    C42 fix 2026-09-26); a reported score always reads the level-frozen thresholds."""
    if arg == "fixture":
        print(FIXTURE_BANNER)
        return fixture_thresholds()
    if arg and arg.startswith("prefreeze:"):
        if not allow_prefreeze:
            raise ReportRefused("a pre-freeze context never produces a reported score; scoring reads the "
                                "level-frozen thresholds")
        from hmb.config import ThresholdsError, resolve_thresholds_arg
        from hmb.store import FreezeViolation
        try:
            return resolve_thresholds_arg(arg, layout)
        except (ThresholdsError, FreezeViolation, FileNotFoundError) as e:
            raise ReportRefused(str(e)) from e
    if arg is None:
        if not layout.thresholds.exists():
            raise ReportRefused(f"{layout.thresholds} does not exist; pass --thresholds <path> "
                                "(or --thresholds fixture for engineering runs)")
        return load_thresholds(layout.thresholds)
    return load_thresholds(arg)


def check_profile(th, profile: str, what: str) -> None:
    """B6 #11: never mix profiles (e.g. production records rescored with fixture caps)."""
    if str(profile) != th.profile.value:
        raise ReportRefused(f"profile mismatch: thresholds are {th.profile.value!r} but {what} is {profile!r}")


def release_id(layout: Layout) -> tuple[str, str | None]:
    """(release label for the file name, release_hash or None)."""
    if layout.release_manifest.exists():
        rel = read_json(layout.release_manifest)
        return str(rel.get("benchmark_version", "unversioned")), rel.get("release_hash")
    return "unreleased", None


def load_cells(layout: Layout, model: str, track: str, th=None) -> dict[str, ScoreRecord]:
    if track not in REPORTED:
        raise ReportRefused(f"track {track} is not a historical track; Track C never enters A/B tables")
    d = layout.p("scores", safe_model_dir(model), track)      # the same directory write_score used (':' in ids)
    out: dict[str, ScoreRecord] = {}
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        if p.name == "run.json":
            continue
        r = ScoreRecord.model_validate(read_json(p))
        if str(r.track) != track:
            raise ReportRefused(f"{p}: record track {r.track} found under track {track}; refused")
        if th is not None:
            check_profile(th, r.profile, f"score record {p}")
        out[r.item_id] = r
    if not out:
        raise ReportRefused(f"no score files under {d}")
    return out


def is_t3_cell(layout: Layout, model: str, track: str, item: str) -> tuple[bool, str]:
    """§13.6 T3 probe cell? Keyed on the submission record's prompt_id; the derived item-id suffix
    is used only when no submission record exists. Returns (is_t3, basis)."""
    from hmb.submission.process import model_dir_name
    p = layout.submission_record(model_dir_name(model), track, item)
    if p.exists():
        return read_json(p).get("prompt_id") in T3_PROMPT_IDS, "prompt_id"
    return item.endswith(T3_SUFFIX), "item_id_suffix"


def load_track_cells(layout: Layout, models, track: str, th=None) -> tuple[dict, dict, dict]:
    """(A/B cells per model with T3 probe cells removed, T3 cells per model, classification basis counts)."""
    main_, t3, basis = {}, {}, {"prompt_id": 0, "item_id_suffix": 0}
    for m in models:
        allc = load_cells(layout, m, track, th)
        main_[m], t3[m] = {}, {}
        for i, r in allc.items():
            flag, how = is_t3_cell(layout, m, track, i)
            basis[how] += 1
            (t3 if flag else main_)[m][i] = r
        if not main_[m]:
            raise ReportRefused(f"no non-probe score files for {m} track {track}")
    return main_, t3, basis


def t3_probe_table(layout: Layout, t3_by_model: dict) -> dict:
    """§13.6 T3 text-only recall probe, reported separately from every A/B table: per book prominence
    tercile when the prominence exists for every probed book, else counts only."""
    from hmb.rights import load_manifests
    from hmb.sampling import prominence_terciles
    terc = prominence_terciles(load_manifests(layout.rights_manifest))
    out: dict[str, Any] = {}
    for m, cells in t3_by_model.items():
        if not cells:
            continue
        book = {}
        for i in cells:
            base = i[:-len(T3_SUFFIX)] if i.endswith(T3_SUFFIX) else i
            hits = [p for sp in ("development", "test") if (p := layout.item_record(sp, base)).exists()]
            book[i] = ((read_json(hits[0]).get("source") or {}).get("book_id")) if hits else None
        counts = {"n": len(cells), "n_ok": sum(str(r.outcome_code) == "OK" for r in cells.values())}
        if all(book[i] in terc for i in cells):
            by: dict[int, list] = {}
            for i, r in cells.items():
                by.setdefault(terc[book[i]], []).append(r)
            out[m] = {**counts, "mode": "per_prominence_tercile", "terciles": {
                str(t): {"n": len(rs), "n_scored": sum(r.S is not None for r in rs),
                         "mean_S": (sum(r.S for r in rs if r.S is not None) / max(1, sum(r.S is not None for r in rs)))
                         if any(r.S is not None for r in rs) else None,
                         "Valid@1": sum(str(r.outcome_code) == "OK" for r in rs) / len(rs)}
                for t, rs in sorted(by.items())}}
        else:
            out[m] = {**counts, "mode": "counts_only", "reason": "book prominence unavailable"}
    return out


def item_meta(layout: Layout, items) -> tuple[dict[str, str], dict[str, int], dict[str, bool | None], dict[str, str]]:
    """split_group, context_char_count, the G7a `touching_cc` border measure (D10) and the split per
    item from the item records (data/<split>/<item>.json)."""
    groups, ctx, touching, split = {}, {}, {}, {}
    for i in items:
        hits = [(sp, p) for sp in ("development", "test") if (p := layout.item_record(sp, i)).exists()]
        if not hits:
            raise ReportRefused(f"item record for {i} not found under data/")
        split[i], path = hits[0]
        rec = read_json(path)
        if not rec.get("split_group"):
            raise ReportRefused(f"item {i} has no split_group (split groups are required for §14 inference)")
        groups[i] = rec["split_group"]
        ctx[i] = int((rec.get("extraction") or {}).get("context_char_count", 0))
        touching[i] = ((rec.get("automatic_metadata") or {}).get("crop_border") or {}).get("touching_cc")
    return groups, ctx, touching, split


def _restrict(cells: dict, keep: set[str]) -> dict:
    return {m: {i: r for i, r in c.items() if i in keep} for m, c in cells.items()}


def touching_cc_table(cells_by_model: dict, items, groups, touching: dict[str, bool | None], th) -> dict:
    """D10: the headline over the items, stratified by the crop's G7a `touching_cc` (a recorded border
    measure, never an exclusion). Descriptive only."""
    out = {}
    for v in (True, False, None):
        sub = [i for i in items if touching.get(i) is v]
        if sub:
            out[str(v).lower()] = {"n": len(sub), "H": {m: headline(c, sub, groups, th)["estimate"]
                                                        for m, c in cells_by_model.items()}}
    return out


def item_diagram_types(layout: Layout, items) -> dict[str, str | None]:
    """The predicted descriptive diagram type of each item (automatic_metadata.diagram_type.label, one
    of vocab.DiagramType, or None when the gatekeeper's label was invalid)."""
    out = {}
    for i in items:
        hits = [p for sp in ("development", "test") if (p := layout.item_record(sp, i)).exists()]
        am = (read_json(hits[0]).get("automatic_metadata") or {}) if hits else {}
        out[i] = (am.get("diagram_type") or {}).get("label")
    return out


def diagram_type_table(cells_by_model: dict, items, groups, types: dict[str, str | None], th) -> dict:
    """Secondary, descriptive: n and the headline per model for each of the six diagram types (plus the
    items whose label was invalid). Never an input to the primary score or a gate."""
    from hmb.vocab import DiagramType
    out: dict[str, Any] = {"role": "secondary_descriptive"}
    for label in [t.value for t in DiagramType] + [None]:
        sub = [i for i in items if types.get(i) == label]
        if sub:
            out[label or "unlabelled_invalid_reply"] = {
                "n": len(sub), "H": {m: headline(c, sub, groups, th)["estimate"] for m, c in cells_by_model.items()}}
    return out


def context_rescore(layout, th, models, track, cells, cards, groups) -> dict:
    """§7.3 sensitivity re-score excluding `context_assisted` facts (stored verdicts, no calls): the
    headline per model over each card's eligible facts that are not context-assisted."""
    from hmb.cards.rederive import context_excluded

    def tr(card, vf):
        return _restricted(card, vf, context_excluded(card))
    out = _ordering(_rescore(layout, th, models, track, cells, cards, tr), groups, th, None)
    out["n_context_assisted_facts"] = sum(1 for c in cards.values() for f in c.facts if f.eligible and f.context_assisted)
    out["role"] = "sensitivity (published, §7.3)"
    return out


def load_verdicts(layout: Layout, model: str, track: str, item: str, th=None) -> VerdictFile | None:
    p = layout.verdicts(model, track, item)
    if not p.exists():
        return None
    vf = VerdictFile.model_validate(read_json(p))
    if th is not None:
        check_profile(th, vf.profile, f"verdict file {p}")
    return vf


def _judge_hashes(layout, models, tracks, items, th=None) -> list[str]:
    out = set()
    for m in models:
        for t in tracks:
            for i in items:
                v = load_verdicts(layout, m, t, i, th)
                if v is not None:
                    out.add(v.judge_config_hash)
    return sorted(out)


def _header(name: str, layout: Layout, th, models, tracks, items) -> dict:
    rel, rel_hash = release_id(layout)
    return {"report": name, "release": rel, "release_hash": rel_hash, "profile": th.profile.value,
            "claim_status": ("engineering evidence only (fixture profile); not a benchmark score"
                             if not th.is_production else "production"),
            "gate_results": "none emitted by this report",
            "thresholds_hash": th.hash, "metric_hash": metric_hash(th),
            "judge_config_hashes": _judge_hashes(layout, models, tracks, items, th),
            "models": list(models), "tracks": list(tracks)}


def _write(layout: Layout, name: str, rep: dict) -> Path:
    path = layout.reports / f"{name}_{rep['release']}.json"
    write_json_immutable(path, json.loads(json.dumps(rep, default=_json_default)))
    return path


def _json_default(o: Any):
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _clean(o):
    """inf -> 'inf', tuples -> lists, recursively (canonical-JSON safe)."""
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


# ------------------------------------------------------------------ score-report
def _rescore(layout, th, models, track, cells, cards, transform) -> dict[str, dict[str, ScoreRecord]]:
    """Re-score every stored cell with transform(card, vf) -> (card', vf') (None card' = drop item)."""
    out: dict[str, dict[str, ScoreRecord]] = {}
    for m in models:
        out[m] = {}
        for i, rec in cells[m].items():
            vf = load_verdicts(layout, m, track, i, th)
            c2, v2 = transform(cards[i], vf)
            if c2 is None or not any(f.eligible for f in c2.facts):
                continue
            out[m][i] = score_cell(c2, None, Outcome(str(rec.outcome_code)), v2 if str(rec.outcome_code) == "OK"
                                   else None, th, model_id=rec.model_id, track=track)
    common = set.intersection(*[set(v) for v in out.values()]) if out else set()
    return {m: {i: r for i, r in v.items() if i in common} for m, v in out.items()}


def _ordering(cells_by_model, groups, th, base_H: dict[str, float] | None) -> dict:
    its = complete_items(cells_by_model)
    H = {m: headline(c, its, groups, th)["estimate"] for m, c in cells_by_model.items()}
    res: dict[str, Any] = {"N": len(its), "H": H}
    if base_H is not None:
        ms = sorted(H)
        res["kendall_tau_b_vs_full"] = (stats.kendall_tau_b([base_H[m] for m in ms], [H[m] for m in ms])
                                        if None not in [H[m] for m in ms] else {"status": "insufficient_items"})
    return res


def lojfo(layout, th, models, track, cells, cards, groups) -> dict:
    """G6 ranking stability: the headline ordering with each judge family's votes removed from the
    stored verdicts (§10.9, §13.5), Kendall τ_b against the full ordering; no new calls."""
    from hmb.adjudication.core import recompute_without_family
    items = complete_items(cells)
    base = {m: headline(cells[m], items, groups, th)["estimate"] for m in models}
    fams = sorted({v.judge_family for m in models for i in items
                   for vf in [load_verdicts(layout, m, track, i, th)] if vf for fv in vf.facts for v in fv.votes})
    by = {}
    for fam in fams:
        def tr(card, vf, fam=fam):
            if vf is None:
                return card, None
            new = recompute_without_family(vf, card, fam)
            return card, vf.model_copy(update={"facts": [fv.model_copy(update={"verdict": new[fv.fact_id]})
                                                         for fv in vf.facts]})
        by[fam] = _ordering(_rescore(layout, th, models, track, cells, cards, tr), groups, th, base)
    taus = [r["kendall_tau_b_vs_full"].get("tau_b") for r in by.values()]
    taus = [t for t in taus if t is not None]
    return {"judge_families": fams, "by_family": by, "min_tau_b": min(taus) if taus else None}


def score_report(layout: Layout, th, models: list[str], tracks: list[str], split: str = "test") -> dict:
    loaded = {t: load_track_cells(layout, models, t, th) for t in tracks}
    all_items = sorted(set().union(*[set(c) for t in tracks for c in loaded[t][0].values()]))
    groups, _, touching, split_of = item_meta(layout, all_items)
    keep = {i for i in all_items if split_of[i] == split}
    if not keep:
        raise ReportRefused(f"no scored items in the {split} split")
    cells = {t: _restrict(loaded[t][0], keep) for t in tracks}
    items = sorted(keep)
    rep = _header("scores", layout, th, models, tracks, items)
    rep["split"] = split
    rep["tables"] = {t: model_track_table(cells[t], groups, th, items=items) for t in tracks}
    rep["by_touching_cc"] = {t: touching_cc_table(cells[t], items, groups, touching, th) for t in tracks}
    dtypes = item_diagram_types(layout, items)
    rep["by_diagram_type"] = {t: diagram_type_table(cells[t], items, groups, dtypes, th) for t in tracks}
    rep["secondary_descriptive"] = ["tables.B.partial_order", "by_touching_cc", "by_diagram_type",
                                    "context_rescore"]
    rep["t3_probe_s13_6"] = {t: t3_probe_table(layout, loaded[t][1]) for t in tracks}
    rep["t3_classification_basis"] = {t: loaded[t][2] for t in tracks}
    prof = th.profile.value
    ev: dict[str, Any] = {"profile": prof, "split": split, "N": len(items)}
    cards = {i: Card.model_validate(read_json(layout.card(i))) for i in items}
    rep["context_rescore"] = {t: context_rescore(layout, th, models, t, cells[t], cards, groups) for t in tracks}
    n_g: dict[str, int] = {}
    for i in items:
        n_g[groups[i]] = n_g.get(groups[i], 0) + 1
    ev["G2"] = {"N": len(items), "F_bar": sum(sum(f.eligible for f in cards[i].facts) for i in items) / len(items),
                "group_sizes": n_g, "profile": prof}
    if "A" in tracks:
        a_cells = cells["A"]
        sep = separation_evidence(a_cells, rep["tables"]["A"]["partial_order"], items, groups, th)
        cf = ceiling_flag(a_cells, items, groups, th)
        ev["G7"] = {**sep, "ceiling_flag": cf.get("flag"), "ceiling": cf, "profile": prof}
        lo = lojfo(layout, th, models, "A", a_cells, cards, groups) if len(models) > 1 else None
        rep["lojfo"] = lo
        ev["G6"] = {"M": len(models), "n_judge_families": len(lo["judge_families"]) if lo else 0,
                    "min_tau_b": lo["min_tau_b"] if lo else None, "profile": prof}
    if "A" in tracks and "B" in tracks:
        ce = context_effect(cells["A"], cells["B"], items, groups, th)
        rep["context_effect"] = ce
        ev["G8"] = ({"ci": ce["pooled"]["ci"], "estimate": ce["pooled"]["estimate"], "profile": prof}
                    if ce.get("status") == "ok" else {"status": ce.get("status")})
    rep["gate_evidence"] = ev
    return _clean(rep)


# ------------------------------------------------------------------ context re-score helper
def _restricted(card: Card, vf: VerdictFile | None, keep: set[str]) -> tuple[Card, VerdictFile | None]:
    """Card with eligibility restricted to `keep` (subset of the frozen eligible set) and the verdict
    file restricted accordingly. An anchor-A EXEMPT whose absent argument entity lost its eligible
    entity fact is no longer 'already counted as an entity miss' and becomes ABSENT."""
    facts = [f.model_copy(update={"eligible": f.fact_id in keep}) for f in card.facts]
    c2 = card.model_copy(update={"facts": facts})
    if vf is None:
        return c2, None
    from hmb.ontology import PREDICATES, entity_args
    ent_ok = {f.args[0] for f in facts if f.eligible and PREDICATES[f.predicate].fclass.value == "entity"}
    fmap = {f.fact_id: f for f in facts}
    out = []
    for fv in vf.facts:
        if fv.fact_id not in keep:
            continue
        f = fmap[fv.fact_id]
        if str(fv.verdict) == "EXEMPT" and f.anchor == "A" and not any(
                x in ent_ok for x in entity_args(f.predicate, f.args)):
            fv = FactVerdict(fact_id=fv.fact_id, verdict="ABSENT", source="extractor", binding=fv.binding,
                             evidence="re-derivation: EXEMPT justification no longer eligible")
        out.append(fv)
    return c2, vf.model_copy(update={"facts": out})


def dev_sizing(layout: Layout, th, models: list[str], G_values: list[int], seed: int) -> Any:
    """MS3 development variance probe (§13.7): the paired Track A differences of two probe models on the
    development split -> σ̂_d, ρ̂, ρ_plan, the simulation-based N*(G) (`stats.sizing_simulation` with the
    registered sizing_grid, sizing_reps, alpha, power_target and delta_star) and the text decision rule.
    The owner commits N* and G* from this report; nothing here sets a level."""
    if len(models) != 2:
        raise ReportRefused("dev-sizing needs exactly two probe models")
    cells, _, _ = load_track_cells(layout, models, "A", th)
    items = sorted(set(cells[models[0]]) & set(cells[models[1]]))
    if getattr(th, "is_prefreeze", False):       # MS3 probe before the freeze: development candidates, their split groups
        groups = development_groups(layout, th, items)
        items = [i for i in items if i in groups]
    else:
        groups, _, _, split_of = item_meta(layout, items)
        items = [i for i in items if split_of[i] == "development"]
    if len(items) < 2:
        raise ReportRefused("dev-sizing reads development-split items only; fewer than two were found")
    d = [cells[models[0]][i].S - cells[models[1]][i].S for i in items]
    g = [groups[i] for i in items]
    import numpy as np
    sigma = float(np.std(d, ddof=1))
    icc = stats.icc_oneway(d, g)
    rp = stats.rho_plan(d, g, reps=int(th.value("sizing_reps")), seed=seed)
    sim = stats.sizing_simulation(d, g, float(th.value("delta_star")), G_values, grid=list(th.value("sizing_grid")),
                                  reps=int(th.value("sizing_reps")), alpha=float(th.value("alpha")),
                                  power_target=float(th.value("power_target")), seed=seed)
    return _clean({"report": "dev_sizing", "profile": th.profile.value, "models": models, "split": "development",
                   "n_items": len(items), "n_groups": len(set(g)), "sigma_d": sigma, "icc": icc, "rho_plan": rp,
                   "simulation": sim, "seed": seed,
                   "decision": {str(G): stats.sizing_decision(rp.get("rho_plan"), sigma, G,
                                                              G_min=int(th.value("G_min"))) for G in G_values},
                   "thresholds_hash": th.hash})


def development_groups(layout: Layout, th, items) -> dict[str, str]:
    """Split group of each development-book candidate among `items`, from the §13.1 split graph over the
    declared development books (the same grouping `items-finalize` later assigns)."""
    from hmb.acquisition.drivers import development_books
    from hmb.cards.batch import candidates_dir
    from hmb.sampling import apply_collapse_guard, split_inputs
    dev = development_books(layout)
    books, its, elig = split_inputs(layout)
    books = [b for b in books if b["book_id"] in dev]
    graph = apply_collapse_guard(books, [i for i in its if i["book_id"] in dev], {b: n for b, n in elig.items() if b in dev}, th)
    out = {}
    for i in items:
        p = candidates_dir(layout) / f"{i}.json"
        if p.exists():
            g = graph["book_group"].get((read_json(p).get("source") or {}).get("book_id"))
            if g:
                out[i] = g
    return out


def _cmd_dev_sizing(args) -> int:
    from hmb.config import execution_context, prefreeze_record
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds(args.thresholds, layout, allow_prefreeze=True)
        rep = dev_sizing(layout, th, args.models, args.G, args.seed)
    except (ReportRefused, GradingError, CalibrationMissing) as e:
        print(f"REFUSED: {e}")
        return 2
    rel = release_id(layout)[0]
    path = layout.reports / f"dev_sizing_{rel}_seed{args.seed}.json"
    rep["execution_context"] = execution_context(th, "dev-sizing", {"models": args.models, "seed": args.seed})
    write_json_immutable(path, json.loads(json.dumps(rep, default=_json_default)))
    prefreeze_record(th, layout, "dev-sizing", [path])
    print(path)
    return 0


# ------------------------------------------------------------------ CLI
def _cmd_score_report(args) -> int:
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds(args.thresholds, layout)
        rep = score_report(layout, th, args.models, sorted(set(args.track)), args.split)
    except (ReportRefused, GradingError, CalibrationMissing) as e:
        print(f"REFUSED: {e}")
        return 2
    print(_write(layout, f"scores_{args.split}", rep))
    return 0


def register_cli(sub) -> None:
    p = sub.add_parser("dev-sizing", help="MS3 development variance probe: σ̂_d, ρ̂, ρ_plan and the §13.7 "
                                          "simulation N*(G) from two probe models' development scores")
    p.add_argument("--models", nargs=2, required=True, help="the two probe models' score directories")
    p.add_argument("--G", nargs="+", type=int, required=True, help="candidate numbers of test groups")
    p.add_argument("--seed", type=int, default=0, help="simulation seed (recorded in the report)")
    p.add_argument("--thresholds", default=None,
                   help="prefreeze:DEV (MS3, before the level freeze), a thresholds.json path, or 'fixture'")
    p.set_defaults(func=_cmd_dev_sizing)
    for name, fn, hlp in (("score-report", _cmd_score_report, "headline + interval, diagnostics, partial order and "
                                                              "the G2/G6/G7/G8 gate evidence (no model calls)"),):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--models", nargs="+", required=True, help="model directory names under scores/")
        p.add_argument("--track", action="append", choices=list(REPORTED), required=True)
        p.add_argument("--thresholds", default=None,
                       help="thresholds.json path (default <root>/thresholds.json); 'fixture' only explicitly")
        p.add_argument("--split", choices=["test", "development"], default="test",
                       help="items reported (default test; development = the MS3 variance probe)")
        p.set_defaults(func=fn)
