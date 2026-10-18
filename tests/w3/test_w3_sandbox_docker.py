"""§9.6 sandbox through REAL compiles in the pinned image (marked docker; skipped when unavailable).

Hostile inputs are wrapped in realistic response envelopes and go through the production harness
(extract -> normalize -> compile_in_sandbox). Every expected code is hand-written from §9.5 and the
documented signatures in hmb.submission.sandbox. Tests run serially (CPU seconds are load-dependent
inside the Docker Desktop VM; see the W3 report)."""
import json
import re
from pathlib import Path

import fitz
import pytest

from hmb.schemas import ModelSpec
from hmb.submission import (SandboxUnavailable, font_coverage_gate, harness_determinism_rerun,
                            process_submission)
from hmb.submission import sandbox as SB
from hmb.submission.sandbox import image_selftest
from hmb.vocab import Outcome
from w3_helpers import compile_response, fast_th, tikz

pytestmark = pytest.mark.docker


def D(doc: str) -> str:
    return "Full document:\n```latex\n" + doc + "\n```\n"


def P(extra: str) -> str:
    return tikz("\\draw (0,0) -- (2,1);\n" + extra)


HOSTILE = [
    # (id, response, thresholds overrides, expected outcome)
    ("write18", P(r"\immediate\write18{touch /job/pwned}"), {}, Outcome.C_SANDBOX_VIOLATION),
    ("lua_os_execute", P(r"\directlua{os.execute('touch /job/pwned')}"), {}, Outcome.C_SANDBOX_VIOLATION),
    ("lua_io_popen", P(r"\directlua{local p = io.popen('id')}"), {}, Outcome.C_SANDBOX_VIOLATION),
    ("lua_io_open_tmp", P(r"\directlua{local f = io.open('/tmp/x','w')}"), {}, Outcome.C_SANDBOX_VIOLATION),
    ("lua_io_open_etc", P(r"\directlua{local f = io.open('/etc/x','w')}"), {}, Outcome.C_SANDBOX_VIOLATION),
    ("lua_lfs_mkdir", P(r"\directlua{lfs.mkdir('/opt/hmb/d')}"), {}, Outcome.C_SANDBOX_VIOLATION),
    ("lua_os_remove", P(r"\directlua{os.remove('/opt/hmb/run-job.sh')}"), {}, Outcome.C_SANDBOX_VIOLATION),
    ("input_passwd", P(r"\node {\input{/etc/passwd}};"), {}, Outcome.C_SANDBOX_VIOLATION),
    ("openout_etc", P(r"\newwrite\hmbf\immediate\openout\hmbf=/etc/pwn.txt\immediate\write\hmbf{x}"), {},
     Outcome.C_SANDBOX_VIOLATION),
    ("infinite_loop", P(r"\def\x{\x}\x"), {"cpu_seconds": 6}, Outcome.C_TIMEOUT),
    ("tex_capacity", P(r"\def\a{\a\a}\a"), {}, Outcome.C_RESOURCE),
    ("huge_output", P(r"\directlua{for i=1,200000 do texio.write_nl('log', string.rep('x',100)) end}"),
     {"output_bytes": 2 * 1024 * 1024, "log_bytes": 1024 * 1024}, Outcome.C_RESOURCE),
    ("memory", P(r"\directlua{local t={} for i=1,400 do t[i]=string.rep('y',1e7)..i end}"), {},
     Outcome.C_RESOURCE),
    ("three_pages", D("\\documentclass{article}\n\\usepackage{tikz}\n\\begin{document}\n"
                      "\\begin{tikzpicture}\\draw (0,0)--(1,1);\\end{tikzpicture}\n\\newpage x\\newpage x\n"
                      "\\end{document}"), {}, Outcome.V_MULTIPAGE),
    ("six_pages", D("\\documentclass{article}\n\\usepackage{tikz}\n\\begin{document}\n"
                    "\\begin{tikzpicture}\\draw (0,0)--(1,1);\\end{tikzpicture}\n"
                    + "\\newpage x" * 5 + "\n\\end{document}"), {}, Outcome.C_RESOURCE),
    ("includegraphics", D("\\documentclass{standalone}\n\\usepackage{graphicx}\n\\begin{document}\n"
                          "\\begin{tikzpicture}\\node{\\includegraphics{example-image}};\\end{tikzpicture}\n"
                          "\\end{document}"), {}, Outcome.P_POLICY_REJECT),
    ("raster_saveimageresource",
     P(r"\node {\saveimageresource{example-image.png}\useimageresource\lastsavedimageresourceindex};"), {},
     Outcome.V_RASTER),
    ("raster_pdfximage", P(r"\node {\pdfximage{example-image.png}\pdfrefximage\pdflastximage};"), {},
     Outcome.V_RASTER),
    ("raster_inline_literal",
     P(r"\node {\pdfextension literal{q 10 0 0 10 0 0 cm BI /W 2 /H 2 /CS /G /BPC 8 ID abcd EI Q}};"), {},
     Outcome.V_RASTER),
    ("special_ps_ignored", P(r"\node {\special{ps: 0 0 moveto}};"), {}, Outcome.OK),
    ("missing_glyph", P("\\node at (1,1) {\U0001F600};"), {}, Outcome.C_MISSING_GLYPH),
    ("empty_picture", tikz(""), {}, Outcome.V_EMPTY),
    # measured in the pinned image: an empty picture in [tikz,border=0pt]standalone ships TWO pages
    # (612x792 + 343.7x0 pt), so §9.5 "pages != 1" applies; never C_RESOURCE (red-team #6)
    ("empty_picture_border0_classD", D("\\documentclass[tikz,border=0pt]{standalone}\n\\begin{document}\n"
                                       "\\begin{tikzpicture}\n\\end{tikzpicture}\n\\end{document}"), {},
     Outcome.V_MULTIPAGE),
    ("oversize", P(r"\draw (0,0) -- (250,0);"), {}, Outcome.V_OVERSIZE),
    ("syntax_error", P(r"\draw (0,0) -- ;"), {}, Outcome.C_ERROR),
    ("undefined_env_classD", D("\\documentclass{article}\n\\begin{document}\n"
                               "\\begin{tikzpicture}\\draw (0,0)--(1,1);\\end{tikzpicture}\n\\end{document}"), {},
     Outcome.C_ERROR),
    ("multi_candidate", "```latex\n\\begin{tikzpicture}\\end{tikzpicture}\n```\n```latex\n"
                        "\\begin{tikzpicture}\\draw(0,0)--(1,1);\\end{tikzpicture}\n\\begin{tikzpicture}"
                        "\\end{tikzpicture}\n```", {}, Outcome.E_MULTI_CANDIDATE),
    ("wallclock", P(""), {"wall_seconds": 0.3}, Outcome.C_WALLCLOCK),
]


@pytest.mark.parametrize("cid,raw,over,expected", HOSTILE, ids=[h[0] for h in HOSTILE])
def test_hostile_inputs_map_to_documented_codes(cid, raw, over, expected, tmp_path):
    th = fast_th(**over)
    ex, nz, cr = compile_response(raw, th, tmp_path / cid)
    got = ex.outcome if not ex.ok else (nz.outcome if nz.outcome else cr.outcome)
    assert got == expected, (cid, cr.to_json() if cr else None)
    if cid == "wallclock":
        assert cr.wallclock_reruns == 1
    if expected == Outcome.C_SANDBOX_VIOLATION:
        assert cr.violations, cr.to_json()
    if expected == Outcome.V_RASTER:
        assert cr.image_xobjects >= 1


def test_selftest_under_production_flags(th):
    r = image_selftest(th)
    assert r["pass"], r["output"]
    assert "safer_incompatible_with_luaotfload (C16)" in r["output"]


def test_ok_multiscript_and_process_submission_artifacts(th, layout):
    raw = tikz("\\draw (0,0) -- (4,0) -- (1,3) -- cycle;\n\\draw (2,1) circle (1.5);\n"
               "\\node[below left] at (0,0) {A};\n\\node[below right] at (4,0) {Β};\n"
               "\\node[above] at (1,3) {Ж 中國 か ệ};")
    model = ModelSpec(model_id="fake/c", provider="fake", family="f", role="contestant")
    meta = {"dataset_version": "v0-fixture", "model_version_string": "fake", "access_date": "2026-09-22",
            "provider_terms_version": "n/a", "output_release": "withheld",
            "image_encoding": {"format": "png", "width": 1024, "height": 1024, "crop_sha256": "0" * 64}}
    rec = process_submission(raw, "end_turn", item_id="hmb_0000000000000001", model=model, track="C",
                             layout=layout, th=th, meta=meta)
    assert rec.outcome_code == "OK" and rec.page_count == 1 and rec.ink_fraction >= th.value("eps_ink")
    cdir = layout.compile_dir("fake_c", "C", "hmb_0000000000000001")
    for name in ("cand.pdf", "cand.log", "render.png", "compile.json", "extraction.json"):
        assert (cdir / name).exists(), name
    from hmb.hashing import sha256_file
    assert sha256_file(cdir / "cand.pdf") == rec.pdf_sha256 and sha256_file(cdir / "render.png") == rec.render_sha256
    tex = (layout.root / rec.normalized_tex).read_bytes()
    assert sha256_file(layout.root / rec.normalized_tex) == rec.normalized_tex_sha256
    assert rec.sandbox_image_digest == SB.sandbox_image_digest() and rec.wrapper_class == "P"
    assert b"\\tracinglostchars=3" in tex
    meta_c = json.loads((cdir / "compile.json").read_text())
    assert meta_c["missing_glyphs"] == [] and meta_c["violations"] == []
    # red-team #1 lifecycle: the stage record lives in compile_dir; the §5.3 record is written once by
    # finalize_submission after CGX, verdicts and scores exist
    final_path = layout.submission_record("fake_c", "C", "hmb_0000000000000001")
    assert (cdir / "submission_stage.json").exists() and not final_path.exists()
    from hmb.cgx import extract_cgx, write_cgx
    from hmb.submission import finalize_submission
    cgx = extract_cgx(cdir / "cand.pdf", item_id=rec.item_id, model_id=rec.model_id, track="C", th=th)
    cgx_path, cgx_sha = write_cgx(cgx, layout, rec.model_id, "C", rec.item_id)
    assert cgx_path == layout.cgx("fake_c", "C", rec.item_id) and sha256_file(cgx_path) == cgx_sha
    assert write_cgx(cgx, layout, rec.model_id, "C", rec.item_id) == (cgx_path, cgx_sha)   # idempotent
    vpath, spath = layout.verdicts("fake_c", "C", rec.item_id), layout.scores("fake_c", "C", rec.item_id)
    for pth, body in ((vpath, b'{"v": 1}'), (spath, b'{"s": 1}')):
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_bytes(body)
    with pytest.raises(ValueError):
        finalize_submission(rec, layout, cgx_path=cgx_path, scores_path=spath)      # OK needs verdicts too
    final = finalize_submission(rec, layout, cgx_path=cgx_path, verdicts_path=vpath, scores_path=spath,
                                judge_config_id="judge-1")
    assert final.cgx_record == "cgx/fake_c/C/hmb_0000000000000001.json" and final.cgx_sha256 == cgx_sha
    assert final.verdicts_sha256 == sha256_file(vpath) and final.scores_sha256 == sha256_file(spath)
    assert json.loads(final_path.read_text("utf-8"))["cgx_sha256"] == cgx_sha


def test_cli_font_gate_and_determinism_rerun(th, tmp_path, capsys):
    """CLI: `font-coverage-gate --cards <dir>` and `determinism-rerun --sample` over stored submissions."""
    from hmb.cli import main
    from hmb.schemas import Card
    from hmb.store import Layout
    cards = tmp_path / "cards"
    cards.mkdir()
    card = Card(item_id="hmb_00000000000000a1", asc_version="asc-1.0.0", card_uid="card_0000000000000001",
                entities=[{"entity_id": "p1", "type": "point", "label": "圓"}],
                facts=[{"fact_id": "f1", "class": "label", "predicate": "label", "args": ["甲乙", "p1"],
                        "strictness": "strict", "decidability": "geo", "anchor": "A", "eligible": True}],
                builder_order=[], family_map={}, profile="fixture")
    (cards / "hmb_00000000000000a1.json").write_text(card.model_dump_json(by_alias=True), encoding="utf-8")
    root = tmp_path / "root"
    assert main(["--root", str(root), "font-coverage-gate", "--cards", str(cards)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["pass"] and out["n_labels"] == 2 and out["n_cards"] == 1
    # two stored OK submissions -> ceil(5% of 2) = 1 sampled, rerun twice, must match the stored render
    lay = Layout(root.resolve())
    model = ModelSpec(model_id="fake/c", provider="fake", family="f", role="contestant")
    meta = {"dataset_version": "v0-fixture", "model_version_string": "fake", "access_date": "2026-09-22",
            "provider_terms_version": "n/a", "output_release": "withheld",
            "image_encoding": {"format": "png", "width": 1024, "height": 1024, "crop_sha256": "0" * 64}}
    for i, body in enumerate(("\\draw (0,0) circle (1);", "\\draw (0,0) -- (2,1) node {中};")):
        rec = process_submission(tikz(body), "end_turn", item_id=f"hmb_000000000000000{i}", model=model,
                                 track="C", layout=lay, th=th, meta=meta)
        assert rec.outcome_code == "OK"
    assert main(["--root", str(root), "determinism-rerun", "--sample"]) == 0
    rep = json.loads((lay.reports / "harness_determinism.json").read_text())
    assert rep["n_ok_submissions"] == 2 and rep["n_rerun"] == 1 and rep["HarnessDeterminism@rerun"] == 1.0
    assert rep["rows"][0]["pass"] and rep["rows"][0]["render_sha256"][0] == rep["rows"][0]["stored_render_sha256"]


def test_determinism_rerun_identical_pdf_and_render(th, tmp_path):
    """§9.10: SOURCE_DATE_EPOCH honoured by LuaTeX -> byte-identical PDF and render on rerun."""
    tex_raw = tikz("\\draw[thick] (0,0) circle (1) node {中 Ж α};\n\\draw (0,0) -- (1,1);")
    from hmb.submission import extract_candidate, normalize
    nz = normalize(extract_candidate(tex_raw, "end_turn"), th)
    rep = harness_determinism_rerun(nz.tex_bytes, th, tmp_path)
    assert rep["outcomes"] == ["OK", "OK"]
    assert rep["identical_pdf"] and rep["identical_render"], rep
    pdf = (tmp_path / "rerun1" / "cand.pdf").read_bytes()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        assert doc.metadata.get("creationDate", "").startswith("D:19700101")


def test_font_coverage_gate_traditional_cjk_kana_and_scripts(th, tmp_path):
    """§9.10: CJK coverage for 1920s traditional forms (and kana, Vietnamese, Greek, Cyrillic)."""
    labels = ["圓", "綫", "圜", "於", "爲", "衆", "卽", "旣", "敎", "塲", "甲乙丙丁", "戊己庚辛", "壬癸",
              "第一圖", "幾何學", "勾股弦", "直角三角形", "イロハ", "ヰヱ", "ゐゑ", "いろは",
              "Hình", "đường tròn", "Ệ", "αβγδ", "ΑΒΓΔ", "ϑϕ", "АБВГ", "Черт.", "ѣіѳѵ", "A'", "B₁", "O"]
    rep = font_coverage_gate(labels, th, tmp_path)
    assert rep["pass"], rep["missing"]
    bad = font_coverage_gate(["A", "\U0001F600", "\U00017000"], th, tmp_path / "neg")
    assert not bad["pass"]
    assert {m["codepoint"] for m in bad["missing"]} == {"U+1F600", "U+17000"}
