"""Reproduction level 2 (compilation): re-run extraction, normalisation, the policy check, the sandboxed LuaLaTeX
compile and the vector-geometry (CGX) extraction on a stratified sample of stored answers, and compare with the
stored records. Needs Docker and the pinned sandbox image (docs/SANDBOX.md); makes no model call.

    HMB_ROOT=ROOT python scripts/recompile_sample.py --per-outcome 3 --work SCRATCH_DIR [--out report.json]

The sample takes up to --per-outcome cells for every (outcome code, model) pair found in the stored answers of
the test split, so invalid outcomes (truncation, policy rejection, compile error, timeout, ...) are exercised as
well as valid drawings. Compiles run one at a time, as in the original run (CPU-time limits are load-dependent).
Answers without a stored raw text are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hmb.cgx.extract import extract_cgx  # noqa: E402
from hmb.config import load_thresholds  # noqa: E402
from hmb.roster import load_roster  # noqa: E402
from hmb.store import Layout, read_json  # noqa: E402
from hmb.submission.process import process_submission  # noqa: E402

META = ("dataset_version", "model_version_string", "access_date", "provider_terms_version", "output_release",
        "image_encoding", "release_hash", "prompt_id", "prompt_sha256", "decoding_params", "attempts")
COMPARE = ("outcome_code", "extraction_path", "n_blocks", "wrapper_class", "body_sha256", "normalized_tex_sha256",
           "page_count")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-outcome", type=int, default=3)
    ap.add_argument("--work", type=Path, required=True, help="empty scratch directory for the recompiled records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    L = Layout.from_env(os.environ.get("HMB_ROOT"))
    th = load_thresholds(L.thresholds)
    roster = {m.model_id: m for m in load_roster(REPO / "configs" / "roster.yaml")["contestant"]}
    if a.work.exists() and any(a.work.iterdir()):
        raise SystemExit(f"{a.work} is not empty")
    W = Layout.from_env(a.work)
    items = sorted(p.stem for p in L.p("data", "test").glob("*.json") if read_json(p)["quality"]["eligible"])
    strata = defaultdict(list)
    for mdir in sorted(p.name for p in L.p("submissions").iterdir() if p.is_dir()):
        for t in ("A", "B"):
            for iid in items:
                sp = L.p("submissions", mdir, t, f"{iid}.json")
                if sp.exists() and L.p("submissions", mdir, t, f"{iid}.txt").exists():
                    strata[(read_json(sp)["outcome_code"], mdir)].append((mdir, t, iid))
    rng = random.Random(a.seed)
    sample = [c for k in sorted(strata) for c in rng.sample(strata[k], min(a.per_outcome, len(strata[k])))]
    rows, t0 = [], time.time()
    for mdir, t, iid in sample:
        stored = read_json(L.p("submissions", mdir, t, f"{iid}.json"))
        raw = L.p("submissions", mdir, t, f"{iid}.txt").read_text(encoding="utf-8")
        meta = {k: stored.get(k) for k in META}
        s = time.time()
        new = process_submission(raw, stored.get("stop_reason"), item_id=iid, model=roster[stored["model_id"]],
                                 track=t, layout=W, th=th, meta=meta).model_dump(mode="json")
        row = {"cell": f"{mdir}/{t}/{iid}", "stored_outcome": stored["outcome_code"], "new_outcome": new["outcome_code"],
               "fields_equal": all(new.get(k) == stored.get(k) for k in COMPARE),
               "field_diffs": {k: [stored.get(k), new.get(k)] for k in COMPARE if new.get(k) != stored.get(k)},
               "pdf_bytes_equal": new.get("pdf_sha256") == stored.get("pdf_sha256") if stored.get("pdf_sha256") else None,
               "cpu_seconds": [stored.get("cpu_seconds"), new.get("cpu_seconds")], "seconds": None, "cgx_equal": None}
        if new["outcome_code"] == "OK" and stored["outcome_code"] == "OK":
            pdf = W.compile_dir(mdir, t, iid) / "cand.pdf"
            cgx = extract_cgx(pdf, item_id=iid, model_id=stored["model_id"], track=t, th=th).model_dump(mode="json")
            ref = read_json(L.p("cgx", mdir, t, f"{iid}.json"))
            row["cgx_equal"] = cgx == ref
        row["seconds"] = round(time.time() - s, 1)
        rows.append(row)
        print(f"{row['cell']}: {row['stored_outcome']} -> {row['new_outcome']} fields_equal={row['fields_equal']} "
              f"cgx_equal={row['cgx_equal']} ({row['seconds']}s)", flush=True)
    summ = {"cells": len(rows), "outcome_equal": sum(r["stored_outcome"] == r["new_outcome"] for r in rows),
            "fields_equal": sum(r["fields_equal"] for r in rows),
            "cgx_checked": sum(r["cgx_equal"] is not None for r in rows),
            "cgx_equal": sum(bool(r["cgx_equal"]) for r in rows),
            "pdf_bytes_equal": sum(bool(r["pdf_bytes_equal"]) for r in rows),
            "pdf_compared": sum(r["pdf_bytes_equal"] is not None for r in rows),
            "strata": len(strata), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summ, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1), encoding="utf-8")
    return 0 if summ["outcome_equal"] == summ["cells"] and summ["cgx_equal"] == summ["cgx_checked"] else 1


if __name__ == "__main__":
    sys.exit(main())
