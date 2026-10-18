"""§9.5 outcome coding from sandbox run artifacts (pure; no Docker). The same `classify` is what
compile_in_sandbox applies to real container output (exercised in test_w3_sandbox_docker.py)."""
import re

import pytest

from hmb.submission import SandboxUnavailable
from hmb.submission import sandbox as SB
from hmb.submission.sandbox import (ENGINE_LINE, SANDBOX_DIR, RunArtifacts, classify, find_violations,
                                    parse_time)
from hmb.vocab import Outcome
from w3_helpers import fast_th


def art(time_txt="1.00 0.20 400000 1.5 0", log="", files=None, oom=False, wall=False, exit_=0):
    f = {"meta/time.txt": time_txt.encode(), "out/cand.log": log.encode(), "meta/exit": str(exit_).encode()}
    f.update(files or {})
    return RunArtifacts(wall_killed=wall, oom_killed=oom, container_exit=0, wall_seconds=1.0, files=f)


def test_engine_line_matches_runner_script():
    runner = (SANDBOX_DIR / "run-job.sh").read_text(encoding="utf-8")
    assert ENGINE_LINE in " ".join(runner.split())
    assert "--safer" not in ENGINE_LINE          # C16 (documented deviation)


def test_parse_time():
    assert parse_time("1.50 0.25 1234 2.00 0") == (0, None, 1.75, 1234)
    assert parse_time("Command terminated by signal 24\n19.9 0.1 5 20.0 0") == (0, 24, 20.0, 5)
    assert parse_time("Command exited with non-zero status 1\n0.5 0.1 5 1 1") == (1, None, 0.6, 5)


@pytest.mark.parametrize("log,expect", [
    ("runsystem(touch /x)...(blocked).", ["S1_shell_escape"]),
    ("HMB-SANDBOX-VIOLATION execute touch /x", ["S2_lua_io_process"]),
    ("! I can't write on file `/etc/pwn.txt'.", ["S3_openout"]),
    ("! I can't write on file `../up.txt'.", ["S3_openout"]),
    ("! I can't write on file `/job/out/ok.txt'.", []),
    ("(/etc/passwd", ["S4_read_outside:/etc/passwd"]),
    ("(/usr/local/texlive/2025/texmf-dist/tex/latex/base/article.cls", []),
    ("(/job/out/cand.aux)", []),
])
def test_violation_signatures(log, expect):
    assert find_violations(log) == expect


def test_wallclock_first(th, tmp_path):
    assert classify(art(wall=True), th, tmp_path).outcome == Outcome.C_WALLCLOCK


def test_violation_beats_error_and_timeout(th, tmp_path):
    r = classify(art("Command terminated by signal 24\n20 0 5 21 0", log="HMB-SANDBOX-VIOLATION popen id"), th, tmp_path)
    assert r.outcome == Outcome.C_SANDBOX_VIOLATION


def test_cpu_limit_is_timeout(th, tmp_path):
    assert classify(art("Command terminated by signal 24\n20 0 5 21 0"), th, tmp_path).outcome == Outcome.C_TIMEOUT
    assert classify(art("Command terminated by signal 9\n19.8 0.3 5 21 0"), th, tmp_path).outcome == Outcome.C_TIMEOUT


def test_resources(th, tmp_path):
    assert classify(art(oom=True), th, tmp_path).outcome == Outcome.C_RESOURCE
    assert classify(art("Command terminated by signal 25\n1 0 5 1 0"), th, tmp_path).outcome == Outcome.C_RESOURCE
    assert classify(art("Command exited with non-zero status 1\n1 0 5 1 1",
                        log="! TeX capacity exceeded, sorry [input stack size=10000]."), th,
                    tmp_path).outcome == Outcome.C_RESOURCE
    big = "x" * (th.value("log_bytes") + 1)
    r = classify(art(log=big), th, tmp_path)
    assert r.outcome == Outcome.C_RESOURCE and r.log_truncated
    assert (tmp_path / "cand.log").stat().st_size == th.value("log_bytes")


def test_missing_glyph_then_error(th, tmp_path):
    log = 'Missing character: There is no 😀 (U+1F600) in font "name:Noto Serif:mode=node;"!'
    r = classify(art("Command exited with non-zero status 1\n1 0 5 1 1", log=log), th, tmp_path)
    assert r.outcome == Outcome.C_MISSING_GLYPH and r.missing_glyphs[0].startswith("U+1F600")
    r = classify(art("Command exited with non-zero status 1\n1 0 5 1 1", log="! Undefined control sequence."), th, tmp_path)
    assert r.outcome == Outcome.C_ERROR


def test_no_pages_is_multipage_and_missing_pdf_is_error(th, tmp_path):
    assert classify(art(log="No pages of output."), th, tmp_path).outcome == Outcome.V_MULTIPAGE
    assert classify(art(log="Output written."), th, tmp_path).outcome == Outcome.C_ERROR


def _pdf_bytes() -> bytes:
    import fitz
    doc = fitz.open()
    doc.new_page(width=100, height=80)
    return doc.tobytes()


def _info(w, h, pages=1):
    return f"Pages:          {pages}\nCropBox:        0.00     0.00   {w:.2f}   {h:.2f}\n".encode()


def test_missing_glyph_in_terminal_only(th, tmp_path):
    """Red-team #8: detection searches the terminal output as well as cand.log."""
    a = art("Command exited with non-zero status 1\n1 0 5 1 1", log="! Emergency stop.",
            files={"meta/stdout.txt": b'Missing character: There is no \xe2\x98\x83 (U+2603) in font "x"!'})
    r = classify(a, th, tmp_path)
    assert r.outcome == Outcome.C_MISSING_GLYPH and r.missing_glyphs == ['U+2603 "x"']


def test_pdfinfo_failure_is_c_error(th, tmp_path):
    """Red-team #10: page_count None (pdfinfo failed) -> C_ERROR."""
    a = art(files={"out/cand.pdf": _pdf_bytes(), "meta/pdfinfo.txt": b"Syntax Error: Couldn't read xref table"})
    assert classify(a, th, tmp_path).outcome == Outcome.C_ERROR


def test_zero_size_cropbox_is_v_empty(th, tmp_path):
    """Red-team #6: an empty picture with a zero-size CropBox is V_EMPTY (ink 0), not C_RESOURCE."""
    for w, h in ((0, 0), (0, 12.5), (12.5, 0)):
        r = classify(art(files={"out/cand.pdf": _pdf_bytes(), "meta/pdfinfo.txt": _info(w, h)}), th, tmp_path)
        assert r.outcome == Outcome.V_EMPTY and r.ink_fraction == 0.0


def test_raster_cap_formula_is_shared_with_runner(th, tmp_path):
    """Red-team #12: ONE formula, evaluated on raw points in Python exactly as in run-job.sh."""
    runner = (SANDBOX_DIR / "run-job.sh").read_text(encoding="utf-8")
    assert SB.RASTER_CAP_AWK in runner
    lim = SB._limits(th)
    assert runner.count('cap="$DIMPT"') == 1 and lim["dim_pt"] == float(f"{200 * 72 / 2.54:.6f}")
    pix, dpi = lim["pix"], lim["dpi"]
    side = (pix ** 0.5) * 72 / dpi                   # square page exactly at the pixel cap
    assert SB.within_raster_caps(side, side, lim)
    assert not SB.within_raster_caps(side + 0.01, side, lim)
    # classification of a page just over the pixel cap but under the dimension cap -> V_OVERSIZE,
    # computed from the raw points (a cm-rounding path would disagree at this boundary)
    a = art(files={"out/cand.pdf": _pdf_bytes(), "meta/pdfinfo.txt": _info(side + 0.01, side)})
    assert classify(a, th, tmp_path).outcome == Outcome.V_OVERSIZE
    a = art(files={"out/cand.pdf": _pdf_bytes(), "meta/pdfinfo.txt": _info(lim["dim_pt"] + 0.01, 10)})
    assert classify(a, th, tmp_path).outcome == Outcome.V_OVERSIZE


def test_no_host_fallback(monkeypatch, th, tmp_path):
    monkeypatch.setattr(SB, "read_lock", lambda: {"image_id": "sha256:" + "0" * 64})
    with pytest.raises(SandboxUnavailable):
        SB.compile_in_sandbox(b"\\relax", th, tmp_path)
    monkeypatch.setattr(SB.shutil, "which", lambda name: None)
    with pytest.raises(SandboxUnavailable):
        SB.compile_in_sandbox(b"\\relax", th, tmp_path)


def test_container_has_no_mounts_and_confinement_flags():
    args = SB.container_args("img", SB._limits(fast_th()), "n", ["x"])
    joined = " ".join(args)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--pids-limit 64", "--user 10001:10001"):
        assert flag in joined
    assert "-v" not in args and "--volume" not in args and "--mount" not in args
    assert re.search(r"--tmpfs /job:rw,nosuid,nodev,noexec,size=\d+", joined)
    assert "seccomp" not in joined          # Docker's default seccomp profile stays active
