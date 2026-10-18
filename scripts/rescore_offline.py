"""Reproduction level 2: recompute item scores from the stored answers, geometry and verdicts, with no model call.

For every stored (model, track, item) cell under a data root (see scripts/materialize_root.py) this script
  1. re-derives the geometry-decided verdicts with the frozen code (hmb.adjudication.core.deterministic_verdicts)
     from the stored CGX record and the stored alignment, and checks them against the stored verdict file;
  2. recomputes the item score with the frozen scorer (hmb.grading.cell.score_cell) from the stored submission
     record, outcome code and verdict file, and checks it against the stored score file;
  3. optionally (--realign N) recomputes the geometric alignment from scratch for N sampled valid cells and checks
     it against the stored alignment before its judge-chosen (Q3) bindings. Alignment is CPU-heavy (seconds to
     hours per cell), which is why the default re-derives verdicts from the stored alignment.
Judge-decided verdicts (Q1 votes, Q3 choices) are inputs here: re-deciding them would need new model calls.

The stored alignment file omits `not_aligned` (entities of model-only types, or every entity when the drawing
exceeded the primitive cap); it is restored here exactly as hmb.adjudication.alignment.align derives it.

Usage (pinned run environment):
    HMB_ROOT=ROOT python scripts/rescore_offline.py --split test [--workers 6] [--realign 50] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hmb.adjudication.alignment import MODEL_ONLY_TYPES, Alignment, align, card_scope  # noqa: E402
from hmb.adjudication.core import deterministic_verdicts  # noqa: E402
from hmb.adjudication.geo import CgxIndex  # noqa: E402
from hmb.config import load_thresholds  # noqa: E402
from hmb.grading.cell import score_cell  # noqa: E402
from hmb.schemas import Card, CgxRecord, ScoreRecord, SubmissionRecord, VerdictFile  # noqa: E402
from hmb.store import Layout, read_json  # noqa: E402
from hmb.vocab import Outcome  # noqa: E402


def stored_alignment(card: Card, cgx: CgxRecord, d: dict) -> Alignment:
    """Post-Q3 alignment as adjudicate() held it when it computed the geometry verdicts."""
    ents, _ = card_scope(card)
    not_aligned = {e for e, v in ents.items() if v.type in MODEL_ONLY_TYPES}
    if cgx.primitives_over_cap is not None:
        not_aligned |= set(ents)
    pre_q3 = {e for e, how in d["method"].items() if how != "q3"}
    return Alignment(binding=dict(d["binding"]), method=dict(d["method"]), ambiguous=set(d["ambiguous"]),
                     absent=set(d["absent"]), candidates={k: list(v) for k, v in d["candidates"].items()},
                     not_aligned=not_aligned, q3_asked=set(d["q3_asked"]), q3_absent=set(d["q3_absent"]),
                     det_bound=pre_q3)


def check_group(root: str, m: str, t: str, items: list[str]) -> dict:
    L = Layout.from_env(root)
    th = load_thresholds(L.thresholds)
    r = {"cells": 0, "score_equal": 0, "score_mismatch": [], "geometry_verdicts_checked": 0,
         "geometry_verdict_mismatch": [], "judged_set_mismatch": [], "outcomes": Counter(), "valid_cells": []}
    for iid in items:
        sp = L.p("scores", m, t, f"{iid}.json")
        if not sp.exists():
            continue
        card = Card.model_validate(read_json(L.card(iid)))
        sub = SubmissionRecord.model_validate(read_json(L.p("submissions", m, t, f"{iid}.json")))
        outcome = Outcome(sub.outcome_code)
        r["outcomes"][outcome.value] += 1
        vf = None
        if outcome == Outcome.OK:
            vf = VerdictFile.model_validate(read_json(L.p("verdicts", m, t, f"{iid}.json")))
            cgx = CgxRecord.model_validate(read_json(L.p("cgx", m, t, f"{iid}.json")))
            al = stored_alignment(card, cgx, read_json(L.p("adjudication", "alignment", m, t, f"{iid}.json")))
            det, to_judge, _ = deterministic_verdicts(card, al, CgxIndex(cgx, th), th)
            stored = {fv.fact_id: fv for fv in vf.facts}
            judged = {fid for fid, fv in stored.items() if fv.source == "judges"}
            if {f.fact_id for f in to_judge} != judged - set(det):
                r["judged_set_mismatch"].append(f"{m}/{t}/{iid}")
            for fid, fv in det.items():
                r["geometry_verdicts_checked"] += 1
                if (str(fv.verdict), fv.source) != (str(stored[fid].verdict), stored[fid].source):
                    r["geometry_verdict_mismatch"].append(f"{m}/{t}/{iid}/{fid}")
            r["valid_cells"].append((m, t, iid))
        sc = score_cell(card, sub, outcome, vf, th, model_id=sub.model_id, track=t)
        ref = ScoreRecord.model_validate(read_json(sp))
        r["cells"] += 1
        if sc.model_dump(mode="json") == ref.model_dump(mode="json"):
            r["score_equal"] += 1
        else:
            r["score_mismatch"].append({"cell": f"{m}/{t}/{iid}", "stored_S": ref.S, "recomputed_S": sc.S})
    return r


def realign_one(root: str, m: str, t: str, iid: str) -> tuple[str, bool, float]:
    L = Layout.from_env(root)
    th = load_thresholds(L.thresholds)
    card = Card.model_validate(read_json(L.card(iid)))
    cgx = CgxRecord.model_validate(read_json(L.p("cgx", m, t, f"{iid}.json")))
    d = read_json(L.p("adjudication", "alignment", m, t, f"{iid}.json"))
    s = time.time()
    fresh = align(card, CgxIndex(cgx, th), th)
    pre = {e: p for e, p in d["binding"].items() if d["method"][e] != "q3"}
    ok = (fresh.binding == pre and set(fresh.ambiguous) == set(d["q3_asked"]) and sorted(fresh.absent) == d["absent"]
          and dict(fresh.candidates) == d["candidates"])
    return f"{m}/{t}/{iid}", ok, round(time.time() - s, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "development"])
    ap.add_argument("--models", nargs="*", default=None, help="model directories under scores/ (default: all)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--realign", type=int, default=0, help="recompute the alignment for this many sampled valid cells")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = os.environ.get("HMB_ROOT")
    L = Layout.from_env(root)
    root = str(L.root)
    items = sorted(p.stem for p in L.p("data", a.split).glob("*.json") if read_json(p)["quality"]["eligible"])
    models = a.models or sorted(p.name for p in L.p("scores").iterdir() if p.is_dir())
    t0 = time.time()
    # chunks of one model-track, split further so that workers stay busy
    chunks = [(m, t, items[i::4]) for m in models for t in ("A", "B") for i in range(4)]
    res = {"split": a.split, "cells": 0, "score_equal": 0, "score_mismatch": [], "geometry_verdicts_checked": 0,
           "geometry_verdict_mismatch": [], "judged_set_mismatch": [], "outcomes": Counter()}
    valid = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(check_group, *zip(*[(root, m, t, its) for m, t, its in chunks])):
            for k in ("cells", "score_equal", "geometry_verdicts_checked"):
                res[k] += r[k]
            for k in ("score_mismatch", "geometry_verdict_mismatch", "judged_set_mismatch"):
                res[k] += r[k]
            res["outcomes"].update(r["outcomes"])
            valid += r["valid_cells"]
            print(f"{res['cells']} cells, {time.time() - t0:.0f}s", flush=True)
        rl = {"checked": 0, "equal": 0, "mismatch": [], "seconds": []}
        if a.realign:
            sample = random.Random(a.seed).sample(sorted(valid), min(a.realign, len(valid)))
            for cell, ok, sec in ex.map(realign_one, *zip(*[(root, m, t, i) for m, t, i in sample])):
                rl["checked"] += 1
                rl["equal"] += ok
                rl["seconds"].append(sec)
                if not ok:
                    rl["mismatch"].append(cell)
    res["outcomes"] = dict(res["outcomes"])
    res["realign"] = rl
    res["seconds"] = round(time.time() - t0, 1)
    summary = {k: (len(v) if isinstance(v, list) else v) for k, v in res.items() if k != "realign"}
    summary["realign"] = {k: (len(v) if isinstance(v, list) else v) for k, v in rl.items()}
    print(json.dumps(summary, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    bad = res["score_mismatch"] or res["geometry_verdict_mismatch"] or res["judged_set_mismatch"] or rl["mismatch"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
