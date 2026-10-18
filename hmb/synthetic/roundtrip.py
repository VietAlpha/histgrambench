"""Extractor round trip (engineering check, formerly "V1x DEV"): generated TikZ -> production render ->
CGX -> the geometry that produced it.

The CGX extractor decides every `geo`-class fact deterministically (§10.1-§10.2), so its correctness is a
software property of the compiled PDF. This module measures it against known geometry with no model call.
It is an offline engineering check: it sets no level, is not a gate and supports no benchmark claim
(checkpoint simplification 2026-09-26). The CGX-primitive ↔ Γ-entity correspondence is established
GEOMETRICALLY, not by the judge-path alignment (lead decision 2026-09-22):

  1. Frame. Γ is in TikZ cm (y up); CGX primitives are in PDF points (page frame, y down; converted
     to y up by `geo_predicates.to_internal`). With no scaling inside the picture the map is
     x' = s·x + t with s = 72/2.54 pt per cm (exact unit conversion) and a translation t fitted as the
     median offset of label-matched point pairs (a Γ point with name label L ↔ the CGX point bound to
     L). With no such pair, t aligns the lower-left corners of the two content boxes.
  2. Assignment. Per geometric type group (points; straight strokes; circles/arcs/ellipses;
     polygons; curves; angle markers; right-angle marks; length ticks) CGX primitives are assigned to
     Γ entities by the Hungarian method on the symmetric Hausdorff distance between the transformed,
     resampled drawn geometries (sample-to-polyline in both
     directions; markers and ticks: the distance between their vertex / centre). A pair is accepted only when that distance is <= ε_on of §10.2
     (`eps_on_frac` x the CGX render diagonal): the extractor's own incidence tolerance, not a new
     number. Unassigned primitives are unmapped; model-only entities (region, solid) are excluded.
  3. Scores over geo facts under closure semantics (`hmb.cards.closure.entails`), so an MGS
     representation choice is never an error:
       recall    = Γ geo facts entailed by the closure of the mapped CGX facts / Γ geo facts
       precision = mapped CGX geo facts entailed by the closure of Γ / all CGX geo facts
     A type disagreement inside a group (e.g. line vs segment) shows up as existence-fact errors.

The §10.3 alignment (`hmb.adjudication.alignment.align`) is reported SEPARATELY: the P/R it would
give (`alignment_precision/recall`, secondary) and `alignment_ambiguity_rate`, an adjudication
property.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from hmb.adjudication.alignment import align, card_scope
from hmb.adjudication.geo import CgxIndex, map_args
from hmb.cards.closure import canon, entails, forward_chain
from hmb.cards.normalize import normalize_label
from hmb.cards.oracle import oracle_card, oracle_facts
from hmb.ontology import CARD_PREDICATES, PREDICATES, OntologyError, entity_args

from hmb.hashing import canonical_json

from .geo_predicates import to_internal
from .spec import Geo, polyline_dist

PT_PER_CM = 72.0 / 2.54
SEED_BASE = 9_000_000          # the seed block of the recorded 2026-09-22 round-trip evidence (reports/v1x_dev_10seeds.txt)
MODEL_ONLY = frozenset({"region", "solid"})
GROUPS = {"point": "point", "segment": "straight", "line": "straight", "ray": "straight", "axis": "straight",
          "arrow": "straight", "circle": "round", "arc": "round", "ellipse": "round", "polygon": "polygon",
          "curve": "curve", "angle_marker": "angle_marker", "right_angle_marked": "right_angle",
          "length_tick": "tick"}
RESAMPLE = 48



def fkey(pred: str, args) -> str:
    return canonical_json([pred, [list(a) if isinstance(a, tuple) else a for a in args]]).decode()

def _geo(pred: str) -> bool:
    return pred in PREDICATES and PREDICATES[pred].decidability.value == "geo"


def _cgx_internal(cgx_record) -> list[dict]:
    out = []
    for p in cgx_record.primitives:
        try:
            ents, _ = to_internal({"y_up": False, "entities": {p.prim_id: {"type": p.type, "g": p.geom}},
                                   "texts": []})
            out.append(dict(ents[0], label=p.label))
        except (KeyError, ValueError, TypeError):
            continue                                   # a primitive type outside the shared format
    return out


def _transform(ents: list[dict], s: float, t: np.ndarray) -> list[dict]:
    """Similarity x -> s x + t over internal entity geometry (no rotation, no reflection)."""
    def P(v):
        return (s * np.asarray(v, float) + t).tolist()
    out = []
    for e in ents:
        g = dict(e["geom"])
        if e["type"] == "point":
            x, y = P([g["x"], g["y"]])
            g["x"], g["y"] = x, y
        for k in ("a", "b", "c"):
            if k in g and isinstance(g[k], list):
                g[k] = P(g[k])
        if e["type"] in ("angle_marker", "right_angle_marked") and "v" in g:
            g["v"] = P(g["v"])
        for k in ("v", "pts", "vertices"):
            if k in g and isinstance(g[k], list) and g[k] and isinstance(g[k][0], (list, tuple)):
                g[k] = [P(q) for q in g[k]]
        for k in ("r", "rx", "ry", "s", "h"):
            if k in g and isinstance(g[k], (int, float)):
                g[k] = s * g[k]
        out.append(dict(e, geom=g))
    return out


ANCHOR_TYPES = {"angle_marker": "v", "right_angle_marked": "v", "length_tick": "c"}


def _samples(G: Geo, eid: str) -> np.ndarray:
    """Resampled drawn geometry; markers and ticks use their anchor (vertex / centre) only, since the
    shared format carries no marker orientation (vertex + size / arc ends / tick stroke)."""
    t = G.type(eid)
    if t in ANCHOR_TYPES:
        return np.asarray([G.E[eid]["geom"][ANCHOR_TYPES[t]]], float)
    pts = []
    for pl in G.polylines(eid):
        pl = np.asarray(pl, float)
        if len(pl) == 1:
            pts.append(pl)
            continue
        seg = np.hypot(*np.diff(pl, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] == 0:
            pts.append(pl[:1])
            continue
        u = np.linspace(0, cum[-1], RESAMPLE)
        pts.append(np.stack([np.interp(u, cum, pl[:, 0]), np.interp(u, cum, pl[:, 1])], axis=1))
    return np.vstack(pts)


def _directed(a_pts: np.ndarray, b_lines: list[np.ndarray]) -> float:
    """max over the samples of A of the distance to B's drawn polylines (not to B's samples, so the
    resampling phase never adds error)."""
    worst = 0.0
    for q in a_pts:
        d = min((polyline_dist(q, pl) if len(pl) > 1 else float(np.hypot(*(q - pl[0])))) for pl in b_lines)
        worst = max(worst, d)
    return worst


def _hausdorff_geo(Ga: Geo, ea: str, Gb: Geo, eb: str) -> float:
    ta, tb = Ga.type(ea), Gb.type(eb)
    if ta in ANCHOR_TYPES or tb in ANCHOR_TYPES:
        return float(cdist(_samples(Ga, ea), _samples(Gb, eb)).max())
    return max(_directed(_samples(Ga, ea), [np.asarray(x, float) for x in Gb.polylines(eb)]),
               _directed(_samples(Gb, eb), [np.asarray(x, float) for x in Ga.polylines(ea)]))


def geometric_mapping(oracle_graph: dict, cgx_record, th) -> dict:
    """{"binding": {entity: prim}, "method", "translation", "unassigned_prims", "unassigned_entities"}."""
    gam = [e for e in oracle_graph["entities"] if e["type"] not in MODEL_ONLY and e["type"] in GROUPS]
    cgx = _cgx_internal(cgx_record)
    s = PT_PER_CM
    by_label = {normalize_label(p["label"]): p for p in cgx if p["type"] == "point" and p.get("label")}
    offs = []
    for e in gam:
        if e["type"] == "point" and e.get("label") and normalize_label(e["label"]) in by_label:
            q = by_label[normalize_label(e["label"])]["geom"]
            offs.append([q["x"] - s * e["geom"]["x"], q["y"] - s * e["geom"]["y"]])
    if offs:
        t, method = np.median(np.asarray(offs), axis=0), f"label_points({len(offs)})"
    else:
        gb, cb = Geo(gam).bbox(), Geo(cgx).bbox()
        t, method = np.array([cb[0] - s * gb[0], cb[1] - s * gb[1]]), "content_box"
    gam_t = _transform(gam, s, t)
    Gg, Gc = Geo(gam_t), Geo(cgx)
    eps = float(th.value("eps_on_frac")) * float(cgx_record.diagonal_D)
    binding: dict[str, str] = {}
    groups = sorted({GROUPS[e["type"]] for e in gam_t})
    for grp in groups:
        E = [e["entity_id"] for e in gam_t if GROUPS[e["type"]] == grp]
        Pr = [p["entity_id"] for p in cgx if GROUPS.get(p["type"]) == grp]
        if not E or not Pr:
            continue
        cost = np.array([[_hausdorff_geo(Gg, e, Gc, p) for p in Pr] for e in E])
        big = np.where(cost <= eps, cost, 1e9)
        rows, cols = linear_sum_assignment(big)
        for r, c in zip(rows, cols):
            if cost[r, c] <= eps:
                binding[E[r]] = Pr[c]
    used = set(binding.values())
    return {"binding": binding, "method": method, "translation": [float(t[0]), float(t[1])], "eps_pt": eps,
            "unassigned_prims": sorted(p["entity_id"] for p in cgx if p["entity_id"] not in used),
            "unassigned_entities": sorted(e["entity_id"] for e in gam_t if e["entity_id"] not in binding)}


def _score(gamma_geo, C_gamma, types, cgx_record, inv: dict[str, str]) -> dict:
    mapped, unmapped = [], []
    for f in cgx_record.facts:
        if not _geo(f.predicate):
            continue
        m = map_args(f.predicate, list(f.args), inv)
        k = None
        if m is not None:
            try:
                k = canon((f.predicate, m))
            except (OntologyError, KeyError, TypeError):
                k = None
        (mapped.append(k) if k is not None else unmapped.append(f.predicate))
    C_cgx = forward_chain(mapped, types) if mapped else set()
    per: dict[str, dict[str, int]] = defaultdict(lambda: {"n_gamma": 0, "recovered": 0, "n_cgx": 0, "correct": 0})
    rec = 0
    for f in gamma_geo:
        hit = entails(C_cgx, f)
        rec += hit
        per[f[0]]["n_gamma"] += 1
        per[f[0]]["recovered"] += hit
    cor = 0
    for g in mapped:
        ok = entails(C_gamma, g)
        cor += ok
        per[g[0]]["n_cgx"] += 1
        per[g[0]]["correct"] += ok
    for p in unmapped:
        per[p]["n_cgx"] += 1
    n_cgx = len(mapped) + len(unmapped)
    return {"n_gamma": len(gamma_geo), "n_gamma_recovered": rec, "n_cgx": n_cgx, "n_cgx_correct": cor,
            "n_cgx_unmapped": len(unmapped), "recall": rec / len(gamma_geo) if gamma_geo else None,
            "precision": cor / n_cgx if n_cgx else None,
            "per_predicate": {k: dict(v) for k, v in sorted(per.items())},
            "mapped": sorted(fkey(p, a) for p, a in mapped)}


def roundtrip_cell(oracle_graph: dict, cgx_record, th, *, condition: str = "clean",
                   profile: str | None = None) -> dict:
    # asc-2.0.0: the round trip measures the extractor on the predicates a card can assert (E, R, L); the layout facts
    # CGX still computes serve only the CONTRADICTED diagnostic and are outside the measure on both sides
    cgx_record = cgx_record.model_copy(update={"facts": [f for f in cgx_record.facts
                                                         if f.predicate in CARD_PREDICATES]})
    gamma, types = oracle_facts(oracle_graph)
    model_only = {e["entity_id"] for e in oracle_graph["entities"] if e["type"] in MODEL_ONLY}
    gamma_all = [f for f in gamma if _geo(f[0])]
    gamma_geo = [f for f in gamma_all if not set(entity_args(f[0], list(f[1]))) & model_only]
    C_gamma = forward_chain(gamma, types)
    gm = geometric_mapping(oracle_graph, cgx_record, th)
    primary = _score(gamma_geo, C_gamma, types, cgx_record, {p: e for e, p in gm["binding"].items()})
    # secondary: the §10.3 judge-path alignment (an adjudication property)
    card = oracle_card(oracle_graph, profile=profile or "production")
    al = align(card, CgxIndex(cgx_record), th)
    secondary = _score(gamma_geo, C_gamma, types, cgx_record, al.inverse)
    scope, _ = card_scope(card)
    n_scope = len([e for e in scope if e not in al.not_aligned])
    return {"spec_id": oracle_graph.get("spec_id"), "diagram_id": oracle_graph.get("spec_id"),
            "family": oracle_graph.get("family"), "condition": condition, "profile": profile or "production",
            **{k: v for k, v in primary.items() if k != "mapped"},
            "n_gamma_model_only_excluded": len(gamma_all) - len(gamma_geo),
            "mapping": {k: v for k, v in gm.items() if k != "binding"} | {"bound": len(gm["binding"])},
            "cgx_mapped": primary["mapped"],
            "alignment_precision": secondary["precision"], "alignment_recall": secondary["recall"],
            "alignment_ambiguity_rate": (len(al.ambiguous) / n_scope) if n_scope else None,
            "alignment": {"bound": len(al.binding), "ambiguous": sorted(al.ambiguous),
                          "absent": sorted(al.absent)}}


def run_roundtrip(layout, th, *, families=None, seeds: int = 10, seed_scan: int = 400) -> list[dict]:
    """Driver: the first `seeds` admitted specs per family (from SEED_BASE) -> production render
    (process_submission, track C) -> extract_cgx -> roundtrip_cell. Returns the cells
    (a render that is not OK is returned as {"family", "spec_id", "status"}). Needs the sandbox; no
    model calls."""
    from hmb.cgx.extract import extract_cgx
    from hmb.submission.process import model_dir_name

    from .admission import admission_reasons
    from .evaluator import oracle_facts
    from .families import FAMILIES, generate
    from .generate import oracle_graph
    from .render import CONTROL_MODEL, render_candidate
    from .tikz import wrap
    out = []
    for fam in families or FAMILIES:
        specs = []
        for seed in range(SEED_BASE, SEED_BASE + seed_scan):
            s = generate(fam, seed)
            if not admission_reasons(s, th):
                specs.append(s)
                if len(specs) == seeds:
                    break
        if not specs:
            out.append({"family": fam, "status": "no admitted spec"})
        for spec in specs:
            og = oracle_graph(spec, oracle_facts(spec, th))
            raw, stop = wrap(og["tikz"], "none")
            cid = spec.spec_id + "__c00"
            r = render_candidate(raw, stop, candidate_id=cid, layout=layout, th=th)
            if r["status"] != "OK":
                out.append({"family": fam, "spec_id": spec.spec_id, "status": r["status"],
                            "reason": r.get("reason")})
                continue
            pdf = layout.compile_dir(model_dir_name(CONTROL_MODEL.model_id), "C", cid) / "cand.pdf"
            cgx = extract_cgx(pdf, item_id=cid, model_id=CONTROL_MODEL.model_id, track="C", th=th)
            cell = roundtrip_cell(og, cgx, th, profile=getattr(th.profile, "value", None))
            out.append(dict(cell, status="OK", extractor_version=cgx.extractor_version))
    return out


def roundtrip_summary(cells: list[dict], worst: int = 3) -> str:
    """Per-family pooled round-trip P/R (sums of counts over cells) and the worst cells (lowest min(P, R))."""
    from collections import OrderedDict
    fams: "OrderedDict[str, list]" = OrderedDict()
    for c in cells:
        fams.setdefault(c["family"], []).append(c)

    def f(x):
        return "-" if x is None else f"{x:.3f}"
    rows = ["family | cells | pooled P | pooled R | cells at 1/1 | align ambiguity (mean) | worst cells"]
    for fam, cs in fams.items():
        ok = [c for c in cs if c.get("status") == "OK"]
        bad = [c for c in cs if c.get("status") != "OK"]
        nc, kc = sum(c["n_cgx"] for c in ok), sum(c["n_cgx_correct"] for c in ok)
        ng, rg = sum(c["n_gamma"] for c in ok), sum(c["n_gamma_recovered"] for c in ok)
        perfect = sum(1 for c in ok if c["precision"] == 1.0 and c["recall"] == 1.0)
        amb = [c["alignment_ambiguity_rate"] for c in ok if c["alignment_ambiguity_rate"] is not None]
        ws = sorted((c for c in ok if not (c["precision"] == 1.0 and c["recall"] == 1.0)),
                    key=lambda c: min(c["precision"] or 0, c["recall"] or 0))[:worst]
        wdesc = []
        for c in ws:
            miss = sorted(((v["correct"] - v["n_cgx"]) + (v["recovered"] - v["n_gamma"]), p)
                          for p, v in c["per_predicate"].items())[:3]
            wdesc.append(f"{c['spec_id']} P={f(c['precision'])} R={f(c['recall'])} "
                         f"[{', '.join(f'{p}({d})' for d, p in miss if d < 0)}]")
        wdesc += [f"{c.get('spec_id')} {c['status']}" for c in bad]
        rows.append(f"{fam} | {len(cs)} | {f(kc / nc if nc else None)} | {f(rg / ng if ng else None)} | "
                    f"{perfect}/{len(ok)} | {f(sum(amb) / len(amb) if amb else None)} | " + "; ".join(wdesc))
    return "\n".join(rows)


def register_cli(sub) -> None:
    def _cmd(args) -> int:
        import json
        import sys

        from hmb.config import fixture_thresholds, load_thresholds
        from hmb.store import Layout, write_mutable
        th = fixture_thresholds() if args.thresholds == "fixture" else load_thresholds(args.thresholds)
        cells = run_roundtrip(Layout.from_env(args.root), th, seeds=args.seeds,
                              families=args.families.split(",") if args.families else None)
        sys.stdout.write(roundtrip_summary(cells) + "\n")
        if args.out:
            write_mutable(args.out, (json.dumps(cells, indent=1, sort_keys=True, default=str) + "\n").encode())
        return 0
    p = sub.add_parser("cgx-roundtrip", help="engineering check: generated TikZ -> sandbox -> CGX vs the known "
                                             "geometry, per family (no model call; not a gate)")
    p.add_argument("--families", default=None)
    p.add_argument("--seeds", type=int, default=10, help="admitted specs per family (default 10)")
    p.add_argument("--out", default=None, help="write the cells as JSON")
    p.add_argument("--thresholds", default="fixture")
    p.set_defaults(func=_cmd)
