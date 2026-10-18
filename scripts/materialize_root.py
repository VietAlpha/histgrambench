"""Rebuild the benchmark's data-root layout from the released files, offline.

The evaluation code (`python -m hmb ...`) and the paper analysis scripts read a data root laid out as in the
original run (data/<split>/<item>.json, semantic_cards/, submissions/, compiled/, cgx/, verdicts/, scores/, ...).
This script writes that layout from
  * this repository's frozen/ folder and results/score_cells.jsonl.gz, and
  * a local copy of the Hugging Face dataset (e.g. `hf download <repo> --repo-type dataset --local-dir DIR`).
Every native record is written byte for byte and checked against the sha256 stored beside it.


Usage (release environment, see scripts/requirements-release.txt):
    python scripts/materialize_root.py --dataset DIR --out ROOT [--skip cgx compiled_pdfs]
Then point the evaluation code at ROOT with the environment variable HMB_ROOT=ROOT.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]


class Writer:
    def __init__(self, root: Path):
        self.root, self.n, self.bad = root, 0, []

    def put(self, rel: str, data: bytes | str | None, sha: str | None) -> None:
        if data is None:
            return
        b = data.encode("utf-8") if isinstance(data, str) else data
        if sha is not None and hashlib.sha256(b).hexdigest() != sha:
            self.bad.append(rel)
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        self.n += 1


def batches(dataset: Path, config: str, columns: list[str] | None):
    files = sorted((dataset / "data" / config).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no Parquet files for config {config!r} under {dataset}/data/{config}")
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=columns):
            yield from b.to_pylist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="local copy of the Hugging Face dataset")
    ap.add_argument("--out", required=True, type=Path, help="data root to create (must not exist or be empty)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["cgx", "compiled_pdfs"],
                    help="large configs not needed for re-scoring or the paper analyses")
    ap.add_argument("--without-stored-reports", action="store_true",
                    help="leave out frozen/reports/scores_*.json so that `hmb score-report` regenerates it "
                         "(reproduction level 1); the stored copy stays in frozen/ for comparison")
    a = ap.parse_args()
    out: Path = a.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; choose a fresh directory")
    w = Writer(out)

    # frozen run configuration, manifests and reports (canonical in this repository)
    for p in sorted((REPO / "frozen").rglob("*")):
        if p.is_file():
            rel = p.relative_to(REPO / "frozen").as_posix()
            if a.without_stored_reports and rel.startswith("reports/scores_"):
                continue
            w.put(rel, p.read_bytes(), None)
    w.put("releases/1.0.0/release.json", (REPO / "frozen" / "manifests" / "release.json").read_bytes(), None)
    # per-work rights records (repository, year, language) are read by the analyses; canonical in the dataset
    w.put("sources/rights_manifest.yaml", (a.dataset / "metadata" / "rights_manifest.yaml").read_bytes(), None)

    # item-level scores (canonical in this repository)
    with gzip.open(REPO / "results" / "score_cells.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            w.put(d["path"], d["content"], d["sha256"])

    split_dir = {"test": "test", "development": "development"}
    for r in batches(a.dataset, "items", ["item_id", "split", "item_record_json", "item_record_sha256", "image",
                                          "original_crop", "page_text", "page_text_sha256"]):
        rec = json.loads(r["item_record_json"])
        w.put(f"data/{split_dir[r['split']]}/{r['item_id']}.json", r["item_record_json"], r["item_record_sha256"])
        assets = rec["assets"]
        if r["image"]:
            w.put(f"images/{assets['crop_canonical_image']}", r["image"]["bytes"], assets["crop_canonical_sha256"])
        if r["original_crop"]:
            w.put(f"images/{assets['crop_image']}", r["original_crop"]["bytes"], assets["crop_image_sha256"])
        if r["page_text"] is not None:
            w.put(f"images/{assets['context_text']}", r["page_text"], r["page_text_sha256"])

    for r in batches(a.dataset, "answer_key_records", ["item_id", "card_json", "card_sha256", "generation_manifest_json",
                                                       "generation_manifest_sha256", "card_gate_json", "card_gate_sha256"]):
        iid = r["item_id"]
        uid = json.loads(r["card_json"])["card_uid"]
        w.put(f"semantic_cards/{iid}.json", r["card_json"], r["card_sha256"])
        w.put(f"manifests/cards/{uid}/{iid}.json", r["generation_manifest_json"], r["generation_manifest_sha256"])
        w.put(f"data/card_gates/{iid}.json", r["card_gate_json"], r["card_gate_sha256"])

    for r in batches(a.dataset, "answers", None):
        base = f"{r['model_dir']}/{r['track']}/{r['item_id']}"
        w.put(f"submissions/{base}.json", r["submission_record_json"], r["submission_record_sha256"])
        w.put(f"submissions/{base}.txt", r["raw_output"], r["raw_output_sha256"])
        w.put(f"submissions/{base}.tex", r["normalized_tex"], r["normalized_tex_sha256"])
        for name in ("submission_stage", "extraction", "compile"):
            w.put(f"compiled/{base}/{name}.json", r[f"{name}_json"], r[f"{name}_sha256"])

    for r in batches(a.dataset, "verdicts", None):
        base = f"{r['model_dir']}/{r['track']}/{r['item_id']}"
        w.put(f"verdicts/{base}.json", r["verdict_json"], r["verdict_sha256"])
        w.put(f"adjudication/alignment/{base}.json", r["alignment_json"], r["alignment_sha256"])

    if "cgx" not in a.skip:
        for r in batches(a.dataset, "cgx", None):
            w.put(f"cgx/{r['model_dir']}/{r['track']}/{r['item_id']}.json", r["cgx_json"], r["cgx_sha256"])
    if "compiled_pdfs" not in a.skip:
        for r in batches(a.dataset, "compiled_pdfs", None):
            w.put(f"compiled/{r['model_dir']}/{r['track']}/{r['item_id']}/cand.pdf", r["pdf"], r["pdf_sha256"])

    print(json.dumps({"root": str(out), "files_written": w.n, "sha256_mismatches": w.bad[:20],
                      "n_sha256_mismatches": len(w.bad)}, indent=1))
    return 1 if w.bad else 0


if __name__ == "__main__":
    sys.exit(main())
