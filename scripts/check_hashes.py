"""Recompute the protocol identities from this checkout and compare them with the paper run (offline).

    python scripts/check_hashes.py            # pinned run environment

* code_hash: content hash of the evaluation code (hmb/submission, hmb/cgx and their dependencies). The run used
  93e9e015... harness versions for 25,060 answers (code f4637512..., before a wrapper fix) and d9a0fa81... for 5,024
  answers (code 213b01d6..., all Claude Haiku 4.5 answers and 10 Gemma 4 26B answers); this repository is the latter.
* form_freeze_sha256: hash of the frozen forms (registry, ontology, prompts, metric and estimator code); frozen as
  255a5c7b... before the test run.
* harness_version: code_hash plus sandbox image, wrapper, extraction rules and rasterizer; it equals the run's value
  only with the original sandbox image (sha256:e902dd5a...).
* release_hash: recomputed from frozen/manifests/release.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EXPECT = {
    "code_hash": "213b01d698c615ceb5edda548eec097937694236ca2400c88e3e08ca8ad605ab",
    "form_freeze_sha256": "255a5c7b3c4cca8d52ecc3d588e66b781cbc37f33ed9e3fd3bd73436d52af84e",
    "harness_version": "d9a0fa817f9ebbd2f9bf143ba854b7933eb5412b0f62274315fb2fb96c700644",
    "release_hash": "sha256:a1bfbd4f1938236aa657dc10fc4e3f51f168b4324197d59d128087021e20283e",
}


def main() -> int:
    from hmb.config import load_thresholds
    from hmb.manifests import _form_freeze_sha, _release_hash_of, form_freeze_components
    from hmb.submission.process import code_hash, harness_version
    th = load_thresholds(REPO / "frozen" / "thresholds.json")
    rel = json.loads((REPO / "frozen" / "manifests" / "release.json").read_text(encoding="utf-8"))
    got = {"code_hash": code_hash(), "form_freeze_sha256": _form_freeze_sha(form_freeze_components()),
           "release_hash": "sha256:" + _release_hash_of(rel)}
    try:
        got["harness_version"] = harness_version(th)
    except Exception as e:  # noqa: BLE001 - needs sandbox/image.lock.json only, not Docker
        got["harness_version"] = f"unavailable ({e})"
    bad = 0
    for k, v in EXPECT.items():
        ok = got[k] == v
        bad += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {k}: {got[k]}" + ("" if ok else f" (expected {v})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
