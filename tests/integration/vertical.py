"""Scaffold 2 vertical path (offline, fixture profile, EXPLICITLY FAKE builders/judges).

Runs a generated known-geometry specification through the production cell path:
  oracle graph -> card (A: direct oracle card; B: 3 fake builders through build_card/consensus)
  -> three candidates {valid, relabelled (a semantic change with a derived diff), compile failure}
  -> process_submission (real sandbox compile + render) -> CGX -> adjudicate (fake judges) -> score
  -> versioned report.
Expected answers come from the generator (oracle graphs, derived diffs Φ) and from §9.5/§11.3
arithmetic written in this file, never from the code under test.

Usage (runbook): .venv/Scripts/python.exe tests/integration/vertical.py --root <empty dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from oracle_providers import OracleBuilder, OracleJudge  # noqa: E402

from hmb.adjudication.serialize import names_for_card, verbalize  # noqa: E402
from hmb.cards import build_card, oracle_card  # noqa: E402
from hmb.cards.align import AlignParams, align  # noqa: E402
from hmb.cards.closure import canon, forward_chain  # noqa: E402
from hmb.cards.parse import parse_builder_output  # noqa: E402
from hmb.cards.verify import build_union, fact_text  # noqa: E402
from hmb.config import fixture_thresholds  # noqa: E402
from hmb.synthetic.evaluator import flip_set, oracle_facts  # noqa: E402
from hmb.synthetic.families import generate  # noqa: E402
from hmb.synthetic.generate import oracle_graph  # noqa: E402
from hmb.synthetic.tikz import emit_tikz  # noqa: E402
from hmb.ledger import Ledger  # noqa: E402
from hmb.pipeline import compile_paths, run_cell, write_run_report  # noqa: E402
from hmb.schemas import ModelSpec  # noqa: E402
from hmb.store import Layout, write_mutable  # noqa: E402

PRICE_ID = "fixture-zero-cost"
JUDGES = [ModelSpec(model_id=f"fake-judge-{i}", provider="fake", family=f"fam_judge_{i}", role="judge",
                    unit_price_id=PRICE_ID) for i in (1, 2, 3)]
BUILDERS = [ModelSpec(model_id=f"fake-builder-{i}", provider="fake", family=f, role="builder",
                      unit_price_id=PRICE_ID)
            for i, f in ((1, "fam_b_x"), (2, "fam_b_y"))]
CONTESTANT = ModelSpec(model_id="trackc-oracle-generator", provider="fake", family="trackc_generator",
                       role="contestant")
META = {"dataset_version": "trackc-vertical", "model_version_string": CONTESTANT.model_id,
        "access_date": "1970-01-01", "provider_terms_version": "n/a (generated control)",
        "output_release": "public", "image_encoding": {"source": "generated"}}
SEED_BASE = 9_000_000          # the seed block the recorded vertical-path evidence used


def relabel(spec, th, base):
    """One semantic change with a known answer: a labelled point gets a fresh label. Φ (the removed facts)
    is derived by re-running the evaluator on the edited specification, never declared."""
    used = {e.label for e in spec.entities if e.label}
    for e in spec.entities:
        if e.type == "point" and e.label:
            mut = spec.copy()
            mut.ent(e.id).label = next(c for c in "QRSTUVWXYZ" if c not in used)
            removed, _ = flip_set(base, oracle_facts(mut, th))
            if removed:
                return mut, removed
    return None


def thresholds():
    # judging and consensus need no calibrated input (majority rule, fixed §8.4 rule; 2026-09-25)
    return fixture_thresholds()


def write_fixture_pricing(layout: Layout) -> None:
    doc = ("version: pricing-1\nentries:\n"
           f"  - unit_price_id: {PRICE_ID}\n    model_id: fixture\n    currency: USD\n"
           "    usd_per_mtok_input: 0\n    usd_per_mtok_output: 0\n    usd_per_image: 0\n"
           "    pricing_url: https://invalid.example/fixture-zero-cost\n    retrieved_utc: '2026-09-22'\n")
    write_mutable(layout.pricing, doc.encode())


def fenced(tikz: str) -> str:
    return "Here is the reconstruction.\n\n```latex\n" + tikz.strip() + "\n```\n"


def truth_texts(og: dict, closed: set) -> set[str]:
    # judge wording is content-derived (serialize.EntityNames), so the oracle card's names are the names
    # any card with the same entities, labels, incidences and anchors receives
    d = names_for_card(oracle_card(og, profile="fixture")).describe
    out = set()
    for pred, args in closed:
        try:
            out.add(verbalize(pred, [list(a) if isinstance(a, tuple) else a for a in args], d))
        except (KeyError, IndexError):
            pass
    return out


def _coords(g) -> list[tuple[float, float]]:
    """All (x, y) pairs in an oracle geometry dict (points, endpoints, vertices, centres)."""
    out: list[tuple[float, float]] = []
    if isinstance(g, dict):
        if isinstance(g.get("x"), (int, float)) and isinstance(g.get("y"), (int, float)):
            out.append((float(g["x"]), float(g["y"])))
        if isinstance(g.get("c"), list) and isinstance(g.get("r"), (int, float)):
            cx, cy, r = float(g["c"][0]), float(g["c"][1]), float(g["r"])
            out += [(cx - r, cy - r), (cx + r, cy + r)]
        for v in g.values():
            if isinstance(v, (dict, list)):
                out += _coords(v)
    elif isinstance(g, list):
        if len(g) == 2 and all(isinstance(v, (int, float)) for v in g):
            out.append((float(g[0]), float(g[1])))
        else:
            for v in g:
                out += _coords(v)
    return out


def _bbox_anchor(e: dict, canvas: dict) -> dict:
    """What a real builder reports: the entity's crop-relative bounding box (y down)."""
    pts = _coords(e.get("geom"))
    if not pts:
        return {"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0}
    W, H = float(canvas["w"]), float(canvas["h"])
    xs = [x / W for x, _ in pts]
    ys = [1.0 - y / H for _, y in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "w": x1 - x0, "h": y1 - y0}


def builder_generation_json(og: dict) -> str:
    """The oracle graph as an asc-2.0.0 writer reply (JSON Lines): entity lines carry type, label and
    parameter; fact lines carry the card relations and measure labels (layout is not a card class)."""
    from hmb.cards.prompts import fact_predicates
    from hmb.ontology import PREDICATES
    params = {f["args"][0]: f["args"][1] for f in og["facts"]
              if PREDICATES[f["predicate"]].fclass.value == "entity" and len(f["args"]) == 2}
    lines = []
    for e in og["entities"]:
        a = _bbox_anchor(e, og["canvas"])
        rec = {"id": e["entity_id"], "type": e["type"], "anchor": [a["x"], a["y"], a["w"], a["h"]]}
        if e.get("label"):
            rec["label"] = e["label"]
        if e["entity_id"] in params:
            rec["param"] = params[e["entity_id"]]
        lines.append(json.dumps(rec, ensure_ascii=False))
    fp = set(fact_predicates())
    lines += [json.dumps({"fact": f["predicate"], "args": f["args"]}, ensure_ascii=False)
              for f in og["facts"] if f["predicate"] in fp]
    return "\n".join(lines)


def run(root: Path, *, family: str = "triangles_polygons", seed: int | None = None) -> dict:
    layout = Layout(root)
    th = thresholds()
    write_fixture_pricing(layout)
    ledger = Ledger(layout)
    seeds = [seed] if seed is not None else [SEED_BASE + k for k in range(1, 40)]
    mut = spec = base = None
    for sd in seeds:
        spec = generate(family, sd)
        base = oracle_facts(spec, th)
        mut = relabel(spec, th, base)
        if mut is not None:
            seed = sd
            break
    if mut is None:
        raise RuntimeError(f"no relabel with a non-empty derived diff for {family} in {seeds[:3]}...")
    og = oracle_graph(spec, base)
    clean_tikz = emit_tikz(spec)
    broken_tikz = clean_tikz.replace("\\draw", "\\drawHMBundefined", 1)
    cands = {
        "valid": dict(raw=fenced(clean_tikz), spec=spec, expected_outcome="OK"),
        "mutation": dict(raw=fenced(emit_tikz(mut[0])), spec=mut[0], expected_outcome="OK",
                         phi_removed=[canon(f) for f in mut[1]], operators=["relabel"]),
        "compile_failure": dict(raw=fenced(broken_tikz), spec=spec, expected_outcome="C_ERROR"),
    }
    for c in cands.values():
        c["truth"] = truth_texts(og, oracle_facts(c["spec"], th).closed)

    def judge_truth_for(key):
        def truth(model_id, question, stmt):
            if question == "Q3":
                return None
            return stmt in cands[key]["truth"]
        return truth

    cells, results = [], {}
    # ---- card A: direct oracle card
    card_a = oracle_card(og, profile="fixture")
    crop = None
    for key in ("valid", "mutation", "compile_failure"):
        cid = f"{spec.spec_id}__{key}"
        judge = OracleJudge(judge_truth_for(key))
        res = run_cell(card_a.model_copy(update={"item_id": cid}), cands[key]["raw"], "end_turn",
                       item_id=cid, contestant=CONTESTANT, track="C", layout=layout, th=th, judges=JUDGES,
                       ledger=ledger, allow_live=True, crop_png=crop, meta=META, provider_override=judge)
        if key == "valid":
            crop = compile_paths(layout, CONTESTANT.model_id, "C", cid)[1]
        results[("oracle_card", key)] = (res, judge.sends)
        cells.append(res)

    # ---- card B: built by 3 fake builders through the production consensus path
    gen = builder_generation_json(og)
    outs = {b.model_id: parse_builder_output(gen, b.model_id, b.family, max_entities=int(th.value("card_max_entities")),
                                             max_facts=int(th.value("card_max_facts"))).output for b in BUILDERS}
    al = align(outs, [b.model_id for b in BUILDERS], AlignParams.from_thresholds(th))
    U = build_union(al)
    types = {s.cid: s.type for s in al.slots}
    ver_truth = {fact_text(f, al) for f in forward_chain(U.facts, types)}
    builder = OracleBuilder(gen, lambda t: t in ver_truth)
    card_b = build_card(spec.spec_id, crop, [], None, builders=BUILDERS, layout=layout, th=th, ledger=ledger,
                        allow_live=True, provider_override=builder)
    for key in ("valid", "mutation"):
        cid = f"{spec.spec_id}__{key}"
        judge = OracleJudge(judge_truth_for(key))
        # distinct contestant id so the built-card cells do not collide with the oracle-card cells
        contestant_b = CONTESTANT.model_copy(update={"model_id": CONTESTANT.model_id + "-cardB"})
        res = run_cell(card_b.model_copy(update={"item_id": cid}), cands[key]["raw"], "end_turn",
                       item_id=cid, contestant=contestant_b, track="C", layout=layout, th=th,
                       judges=JUDGES, ledger=ledger, allow_live=True, crop_png=crop,
                       meta=dict(META, model_version_string=contestant_b.model_id), provider_override=judge)
        results[("built_card", key)] = (res, judge.sends)
        cells.append(res)

    extra = {"spec_id": spec.spec_id, "family": family, "seed": seed, "gamma_size": len(base.mgs),
             "mutation_operators": cands["mutation"]["operators"],
             "phi_removed": [list(map(str, f)) for f in cands["mutation"]["phi_removed"]],
             "card_b": {"card_uid": card_b.card_uid, "n_facts": len(card_b.facts),
                        "n_eligible": sum(f.eligible for f in card_b.facts)},
             "fake_builder_sends": builder.sends,
             "ledger_spent_usd": ledger.spent()}
    report = write_run_report(layout, f"vertical_{spec.spec_id}", th, cells, extra)
    return {"layout": layout, "th": th, "ledger": ledger, "spec": spec, "og": og, "base": base,
            "cands": cands, "results": results, "card_a": card_a, "card_b": card_b, "report": report,
            "builder": builder}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--family", default="triangles_polygons")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")
    out = run(Path(a.root), family=a.family, seed=a.seed)
    print(out["report"].read_text(encoding="utf-8"))
