"""Minimal offline example: score stored contestant answers with the real pipeline, no model call.

    python examples/minimal_example.py [--root examples/sample_root] [--item ITEM] [--no-docker]

For every stored answer to one figure (6 models x 2 tracks) the script
  1. loads the figure record and its answer key;
  2. re-runs extraction, normalisation, the policy check and - when Docker and the pinned sandbox image are
     available - the sandboxed LuaLaTeX compile on the stored raw answer, in a temporary directory, and compares the
     outcome code with the stored one (without Docker it uses the stored outcome and compiled PDF);
  3. for valid drawings, reads the compiled PDF back as vector geometry (CGX), aligns the answer key to it, decides
     every geometry-checkable fact, takes the judge-decided facts and the judges' object choices from the stored
     verdict file (re-deciding them would need model calls), and computes the item score;
  4. prints the recomputed and stored scores side by side.
The sample root holds two figures of one work (Ramus, The Way to Geometry); any root made by scripts/materialize_root.py works too.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hmb.adjudication.alignment import align  # noqa: E402
from hmb.adjudication.core import deterministic_verdicts  # noqa: E402
from hmb.adjudication.geo import CgxIndex  # noqa: E402
from hmb.cgx.extract import extract_cgx  # noqa: E402
from hmb.config import load_thresholds  # noqa: E402
from hmb.grading.cell import score_cell  # noqa: E402
from hmb.roster import load_roster  # noqa: E402
from hmb.schemas import Card, ScoreRecord, SubmissionRecord, VerdictFile  # noqa: E402
from hmb.store import Layout, read_json  # noqa: E402
from hmb.submission.process import process_submission  # noqa: E402
from hmb.vocab import Outcome  # noqa: E402

META = ("dataset_version", "model_version_string", "access_date", "provider_terms_version", "output_release",
        "image_encoding", "release_hash", "prompt_id", "prompt_sha256", "decoding_params", "attempts")


def docker_ok() -> bool:
    try:
        from hmb.submission import sandbox
        return bool(sandbox.sandbox_available())
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "examples" / "sample_root")
    ap.add_argument("--item", default=None)
    ap.add_argument("--no-docker", action="store_true")
    a = ap.parse_args()
    L = Layout.from_env(a.root)
    th = load_thresholds(L.thresholds)
    roster = {m.model_id: m for m in load_roster(REPO / "configs" / "roster.yaml")["contestant"]}
    item = a.item or sorted(p.stem for p in L.p("data", "test").glob("*.json"))[0]
    rec = read_json(L.p("data", "test", f"{item}.json"))
    card = Card.model_validate(read_json(L.card(item)))
    key = [f for f in card.facts if f.eligible]
    print(f"figure {item}: {rec['source']['work_id']}, scan page {rec['source']['page_index']}, "
          f"{len(key)} key facts {dict(Counter(str(f.fact_class) for f in key))}")
    use_docker = not a.no_docker and docker_ok()
    print(f"compile: {'sandbox (Docker)' if use_docker else 'stored outcome and PDF (no Docker)'}\n")
    print(f"{'model':48} {'trk':3} {'outcome':16} {'S recomputed':>12} {'S stored':>9}  match")
    ok_all = True
    with tempfile.TemporaryDirectory(prefix="hgb_example_") as tmp:
        W = Layout.from_env(tmp)
        for mdir in sorted(p.name for p in L.p("scores").iterdir() if p.is_dir()):
            for t in ("A", "B"):
                sp = L.p("submissions", mdir, t, f"{item}.json")
                if not sp.exists():
                    continue
                stored = read_json(sp)
                sub = SubmissionRecord.model_validate(stored)
                pdf = L.compile_dir(mdir, t, item) / "cand.pdf"
                if use_docker:
                    raw = L.p("submissions", mdir, t, f"{item}.txt").read_text(encoding="utf-8")
                    new = process_submission(raw, stored.get("stop_reason"), item_id=item,
                                             model=roster[stored["model_id"]], track=t, layout=W, th=th,
                                             meta={k: stored.get(k) for k in META})
                    if new.outcome_code != sub.outcome_code:
                        print(f"  recompiled outcome {new.outcome_code} differs from stored {sub.outcome_code}")
                        ok_all = False
                    pdf = W.compile_dir(mdir, t, item) / "cand.pdf"
                outcome = Outcome(sub.outcome_code)
                vf = None
                if outcome == Outcome.OK:
                    cgx = extract_cgx(pdf, item_id=item, model_id=sub.model_id, track=t, th=th)
                    index = CgxIndex(cgx, th)
                    al = align(card, index, th)                       # geometric alignment, recomputed
                    stored_vf = VerdictFile.model_validate(read_json(L.p("verdicts", mdir, t, f"{item}.json")))
                    stored_al = read_json(L.p("adjudication", "alignment", mdir, t, f"{item}.json"))
                    al.det_bound = set(al.binding)
                    for e, how in stored_al["method"].items():       # the judges' object choices (Q3), stored
                        if how == "q3":
                            al.binding[e], al.method[e] = stored_al["binding"][e], "q3"
                    al.ambiguous = set(stored_al["ambiguous"])
                    al.q3_asked, al.q3_absent = set(stored_al["q3_asked"]), set(stored_al["q3_absent"])
                    det, _to_judge, _ = deterministic_verdicts(card, al, index, th)
                    facts = [det.get(fv.fact_id, fv) for fv in stored_vf.facts]   # judged facts: stored votes
                    vf = stored_vf.model_copy(update={"facts": facts})
                sc = score_cell(card, sub, outcome, vf, th, model_id=sub.model_id, track=t)
                ref = ScoreRecord.model_validate(read_json(L.p("scores", mdir, t, f"{item}.json")))
                match = abs(sc.S - ref.S) < 1e-12
                ok_all &= match
                print(f"{mdir:48} {t:3} {outcome.value:16} {sc.S:12.4f} {ref.S:9.4f}  {'yes' if match else 'NO'}")
    print("\nall recomputed scores equal the stored scores" if ok_all else "\nsome results differ (see above)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
