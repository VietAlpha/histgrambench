"""C24 CPU-time check (`sandbox-cputime-check`): sampling, serial vs concurrent comparison and outcome-change
detection over stored submissions, with a fake container runner (no Docker). The real compiles run on the
target host; see BUILD_STATUS.md."""
import json
import threading

from hmb.schemas import SubmissionRecord
from hmb.store import Layout
from hmb.submission.cli import cputime_check_stored
from hmb.submission.sandbox import RunArtifacts


def _store(lay: Layout, n: int) -> None:
    for i in range(n):
        item = f"hmb_{i:016x}"
        tex = lay.root / "submissions" / "fake_c" / "A" / f"{item}.tex"
        tex.parent.mkdir(parents=True, exist_ok=True)
        tex.write_bytes(f"doc {i}".encode())
        rec = SubmissionRecord(
            item_id=item, dataset_version="v0-fixture", model_id="fake/c", model_version_string="fake",
            model_family="f", access_date="2026-10-01", provider_terms_version="n/a", output_release="withheld",
            track="A", prompt_id="reconstruction-A-v2", prompt_sha256="0" * 64, decoding_params={},
            image_encoding={}, attempts=[], harness_version="h", raw_output="r", raw_output_sha256="0" * 64,
            extraction_path="fenced", normalized_tex=str(tex.relative_to(lay.root)).replace("\\", "/"),
            outcome_code="C_ERROR", cpu_seconds=float(i))
        cdir = lay.compile_dir("fake/c", "A", item)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "submission_stage.json").write_text(rec.model_dump_json(), encoding="utf-8")


def _runner(slow_doc: bytes, factor: float):
    """CPU = stored index; in the concurrent pass (pool threads) `slow_doc` takes `factor` x and hits SIGXCPU."""
    def run(tex: bytes, th) -> RunArtifacts:
        i = int(tex.split()[1])
        pooled = threading.current_thread().name.startswith("ThreadPoolExecutor")
        cpu = i * (factor if pooled and tex == slow_doc else 1.0)
        timing = (f"Command terminated by signal 24\n{cpu} 0 5 1 0" if cpu >= th.value("cpu_seconds")
                  else f"Command exited with non-zero status 1\n{cpu} 0 5 1 1")
        return RunArtifacts(False, False, 0, 1.0, {"meta/time.txt": timing.encode(), "out/cand.log": b"! err"})
    return run


def test_cputime_check_samples_top_cpu_and_flags_outcome_change(th, tmp_path):
    lay = Layout(tmp_path.resolve())
    _store(lay, 12)
    rep = cputime_check_stored(lay, th, n=6, concurrency=3, reps=2, runner=_runner(b"doc 11", 2.0))
    keys = [r["submission"] for r in rep["rows"]]
    assert rep["n_stored"] == 12 and rep["n"] == 6 and rep["n_top_cpu"] == 3
    for i in (9, 10, 11):                                   # the highest stored CPU seconds are always sampled
        assert f"fake_c/A/hmb_{i:016x}" in keys
    slow = next(r for r in rep["rows"] if r["submission"].endswith(f"{11:016x}"))
    assert slow["serial_cpu"] == [11.0, 11.0] and slow["parallel_cpu"] == [22.0, 22.0] and slow["ratio"] == 2.0
    assert slow["outcomes"] == {"serial": ["C_ERROR"], "parallel": ["C_TIMEOUT"]} and slow["outcome_changed"]
    assert rep["outcome_changes"] == 1 and rep["ratio_max"] == 2.0 and rep["ratio_median"] == 1.0
    assert not (tmp_path / "reports").exists()               # writes nothing under the root
    # identical CPU in both passes: no change, ratio 1
    rep2 = cputime_check_stored(lay, th, n=6, concurrency=3, reps=1, runner=_runner(b"none", 1.0))
    assert rep2["outcome_changes"] == 0 and rep2["ratio_max"] == 1.0
    json.dumps(rep2)
