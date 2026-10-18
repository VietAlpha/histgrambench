"""§9.1 prompts + contestant call assembly; process_submission artifact writing (pre-compile paths,
no Docker); outcome distribution. The OK path through a real compile is in test_w3_sandbox_docker.py."""
import json

import pytest

from hmb.hashing import sha256_bytes, sha256_file
from hmb.ontology import ALLOWED_VARIATION
from hmb.schemas import ModelSpec, SubmissionRecord
from hmb.store import ImmutableConflict
from hmb.submission import build_contestant_request, load_prompt, outcome_distribution, process_submission
from hmb.submission import finalize_submission
from hmb.submission.process import run_contestant
from hmb.submission.wrapper import LIB_ALLOW, PKG_ALLOW
from hmb.vocab import Outcome

MODEL = ModelSpec(model_id="fake/contestant:1", provider="fake", family="fakefam", role="contestant")
META = {"dataset_version": "v0-fixture", "model_version_string": "fake-1", "access_date": "2026-09-22",
        "provider_terms_version": "n/a", "output_release": "withheld",
        "image_encoding": {"format": "png", "width": 1024, "height": 1024, "crop_sha256": "0" * 64}}


def test_prompts_state_six_elements_verbatim():
    _pid, a, sha = load_prompt("A")
    assert sha == sha256_bytes(a.encode("utf-8"))
    must = [
        "Reconstruct the mathematical figure the original author intended; do not reproduce scan damage, paper texture, or print defects.",
        "Exactly one fenced block tagged ```latex, containing exactly one tikzpicture (optionally inside a standalone document). Any reasoning goes outside the block. Nothing after the block.",
        "Packages (PKG_ALLOW): " + ", ".join(PKG_ALLOW) + ".",
        "TikZ libraries (LIB_ALLOW): " + ", ".join(LIB_ALLOW) + ".",
        "Shell escape, \\write18, \\directlua, \\input, \\include, \\openin, \\openout, filecontents, \\includegraphics, \\pdfximage, \\special, font-selection macros, external files, network access.",
        "Absolute coordinates, scale, line width, font, colour, small label offsets.",
        "Labels may be written literally in their original script (Chinese, Japanese, Vietnamese, Cyrillic, Greek, Latin); the compiler supplies covering fonts.",
    ]
    for m in must:
        assert m in a, m
    assert len(ALLOWED_VARIATION) == 9   # the prompt's not-judged list enumerates the 9 §7.6 items
    pid_b, b, _ = load_prompt("B")
    assert pid_b == "reconstruction-B-v2" and b.startswith(a)
    assert "surrounding text from the same page and may contain the figure's caption" in b
    assert load_prompt("C")[0] == "reconstruction-A-v2"


def test_rendered_contestant_prompts_make_no_stale_scoring_claims(th, tmp_path):
    """2026-09-26: reflection is not scored (no chirality-bearing card fact since asc-2.0.0) and the extras channel
    was removed, so no prompt a contestant actually receives may say otherwise; v2 ids keep v1 caches apart."""
    import re
    from hmb.submission.prompts import build_t3_request
    crop = tmp_path / "c.png"
    crop.write_bytes(b"x")
    text_only = MODEL.model_copy(update={"input_mode": "text_only"})
    reqs = [build_contestant_request(t, m, crop, "0" * 64, "Fig. 1.", th, phase="FINAL", cache_scope="s")
            for t in ("A", "B", "C") for m in (MODEL, text_only)]
    reqs.append(build_t3_request(MODEL, "Fig. 1.", th, phase="FINAL", cache_scope="s"))
    for r in reqs:
        assert not re.search(r"reflect|mirror|extra|unsupported element|over-?draw", r.text, re.I), r.cache_scope
        assert r.cache_scope.endswith("-v2") and "-v1" not in r.cache_scope


def test_contestant_request(th, tmp_path):
    crop = tmp_path / "c.png"
    crop.write_bytes(b"\x89PNG fake")
    req = build_contestant_request("B", MODEL, crop, sha256_file(crop), "Fig. 3. Triangle ABC.", th,
                                   phase="FINAL", cache_scope="harness-x")
    assert req.max_tokens == th.value("contestant_max_tokens") and req.purpose == "contestant"
    assert req.text.endswith("Fig. 3. Triangle ABC.") and req.image_sha256 == [sha256_file(crop)]
    assert req.spec.decoding["temperature"] == 0.0


def test_text_only_contestant_request_has_no_image(th, tmp_path):
    """input_mode text_only (DeepSeek V3.1 floor): no image on any track, the frozen no-image prompt
    plus the context text; a vision contestant keeps its image, and Track A stays context-free for it."""
    from hmb.submission.prompts import contestant_prompt, decoding_params, image_encoding, load_t3_prompt
    crop = tmp_path / "c.png"
    crop.write_bytes(b"\x89PNG fake")
    sha = sha256_file(crop)
    floor = MODEL.model_copy(update={"model_id": "deepseek.v3.1", "input_mode": "text_only"})
    t3_pid, t3_text, _ = load_t3_prompt()
    for t in ("A", "B"):
        r = build_contestant_request(t, floor, crop, sha, "Fig. 3. Triangle ABC.", th, phase="FINAL", cache_scope="h")
        assert r.image_paths == [] and r.image_sha256 == []
        assert r.text == t3_text + "\n" + "Fig. 3. Triangle ABC." and r.cache_scope == f"h|{t3_pid}"
        assert contestant_prompt(t, floor)[0] == t3_pid
        v = build_contestant_request(t, MODEL, crop, sha, "Fig. 3. Triangle ABC.", th, phase="FINAL", cache_scope="h")
        assert v.image_paths == [str(crop)] and v.image_sha256 == [sha]
        assert ("Fig. 3. Triangle ABC." in v.text) == (t == "B") and contestant_prompt(t, MODEL)[0] == load_prompt(t)[0]
    assert image_encoding(floor, th, sha) == {"format": "none", "crop_sha256": None}
    assert image_encoding(MODEL, th, sha)["crop_sha256"] == sha
    effortful = MODEL.model_copy(update={"effort": "high"})
    assert decoding_params(effortful, th)["effort"] == "high" and decoding_params(MODEL, th)["effort"] is None


@pytest.mark.parametrize("raw,stop,code", [
    ("", "end_turn", Outcome.E_NO_OUTPUT),
    ("no code", "length", Outcome.E_TRUNCATED),
    ("```latex\n\\begin{tikzpicture}\\end{tikzpicture}\n\\begin{tikzpicture}\\end{tikzpicture}\n```",
     "end_turn", Outcome.E_MULTI_CANDIDATE),
    ("```latex\n\\usepackage{graphicx}\n\\begin{tikzpicture}\\end{tikzpicture}\n```", "end_turn",
     Outcome.P_POLICY_REJECT),
])
def test_precompile_outcomes_write_artifacts(raw, stop, code, th, layout):
    rec = process_submission(raw, stop, item_id="hmb_00000000000000aa", model=MODEL, track="A",
                             layout=layout, th=th, meta=META)
    assert isinstance(rec, SubmissionRecord) and rec.outcome_code == code.value
    rawp = layout.root / rec.raw_output
    assert rawp.read_bytes() == raw.encode() and rec.raw_output_sha256 == sha256_bytes(raw.encode())
    stage = layout.compile_dir("fake_contestant_1", "A", "hmb_00000000000000aa") / "submission_stage.json"
    stored = json.loads(stage.read_text("utf-8"))
    assert not layout.submission_record("fake_contestant_1", "A", "hmb_00000000000000aa").exists()
    assert stored["outcome_code"] == code.value and len(stored["harness_version"]) == 64
    assert rec.pdf_sha256 is None and rec.normalized_tex is None
    ext = json.loads((layout.compile_dir("fake_contestant_1", "A", "hmb_00000000000000aa") / "extraction.json").read_text())
    assert ext["outcome"] in (code.value, None)
    if code == Outcome.P_POLICY_REJECT:
        assert ext["reject_token"] == "\\usepackage{graphicx}" and rec.wrapper_class == "P"


@pytest.mark.parametrize("raw,stop,code", [
    ("I will now draw the figure as", "max_tokens", Outcome.E_TRUNCATED),     # Bedrock Converse spelling
    ("I will now draw the figure as", "length", Outcome.E_TRUNCATED),         # Bedrock Mantle spelling
    ("```latex\n\\begin{tikzpicture}\\draw (0,0) --", "max_tokens", Outcome.E_TRUNCATED),
    ("I will now draw the figure as", "end_turn", Outcome.E_NO_TIKZ),         # not truncated: no TikZ
    ("I will now draw the figure as", "stop", Outcome.E_NO_TIKZ),
    ("```latex\n\\begin{tikzpicture}\\draw (0,0) --", "end_turn", Outcome.E_NO_TIKZ),
])
def test_truncation_is_classified_for_both_bedrock_transports(raw, stop, code, th, layout):
    """A reply cut at the output cap is E_TRUNCATED whether Converse (`max_tokens`) or Mantle (`length`)
    reports it, before the no-TikZ rule; the raw stop reason is kept in the record."""
    iid = "hmb_" + sha256_bytes(f"{raw}|{stop}".encode())[:16]
    rec = process_submission(raw, stop, item_id=iid, model=MODEL, track="A", layout=layout, th=th, meta=META)
    assert rec.outcome_code == code.value and rec.stop_reason == stop


def test_resume_same_raw_and_one_output_per_cell(th, layout):
    kw = dict(item_id="hmb_00000000000000bb", model=MODEL, track="A", layout=layout, th=th, meta=META)
    r1 = process_submission("no code", "end_turn", **kw)
    r2 = process_submission("no code", "end_turn", **kw)
    assert r1 == r2
    with pytest.raises(ImmutableConflict):
        process_submission("different output", "end_turn", **kw)


def test_thinking_stored_separately(th, layout):
    raw = "<think>I will draw \\begin{tikzpicture}</think>no code after thinking"
    rec = process_submission(raw, "end_turn", item_id="hmb_00000000000000cc", model=MODEL, track="B",
                             layout=layout, th=th, meta=META)
    assert rec.outcome_code == Outcome.E_NO_TIKZ.value      # thinking excluded from extraction
    p = layout.raw_output("fake_contestant_1", "B", "hmb_00000000000000cc").with_suffix(".thinking.txt")
    assert p.read_text("utf-8").startswith("I will draw")


class FakeProvider:
    """Explicitly marked fake contestant (tests only)."""
    def __init__(self, text, stop="end_turn"):
        self.text, self.stop, self.calls = text, stop, 0

    def send(self, req, images):
        self.calls += 1
        return {"text": self.text, "stop_reason": self.stop, "model_version_string": "fake-1",
                "input_tokens": 10, "output_tokens": 5}


class FakeLedger:
    """Duck-typed ledger for a fake provider call (no prices, no spend)."""
    def estimate_usd(self, req):
        return 0.0

    def reserve(self, *a, **k):
        return "r1"

    def settle(self, *a, **k):
        return 0.0

    def release(self, *a, **k):
        return None


def test_run_contestant_with_fake_provider(th, layout, tmp_path):
    crop = tmp_path / "c.png"
    crop.write_bytes(b"\x89PNG fake")
    fake = FakeProvider("I can only describe it in words.")
    rec = run_contestant(item_id="hmb_00000000000000dd", track="A", model=MODEL, crop_path=crop,
                         crop_sha256=sha256_file(crop), context_text=None, layout=layout, th=th,
                         ledger=FakeLedger(), allow_live=True, phase="FINAL", cache_scope="t",
                         meta={k: v for k, v in META.items() if k != "model_version_string"},
                         provider_override=fake)
    assert rec.outcome_code == Outcome.E_NO_TIKZ.value and fake.calls == 1
    assert rec.decoding_params["max_tokens"] == th.value("contestant_max_tokens")
    assert rec.attempts and rec.attempts[0].index == 1


def test_outcome_distribution():
    codes = ["OK", "OK", "V_EMPTY", "C_TIMEOUT", "E_TRUNCATED", "P_POLICY_REJECT", "V_MULTIPAGE", "E_MULTI_CANDIDATE"]
    d = outcome_distribution(codes)
    assert d["n"] == 8 and d["distribution"]["OK"] == 2 and sum(d["distribution"].values()) == 8
    assert d["rates"]["Valid@1"] == 2 / 8 and d["rates"]["Compile@1"] == 4 / 8
    assert d["rates"]["Protocol-violation@1"] == 2 / 8 and d["rates"]["Timeout@1"] == 1 / 8
    assert outcome_distribution([])["rates"]["Valid@1"] is None


def test_finalize_non_ok_with_scores_only(th, layout):
    stage = process_submission("no code", "end_turn", item_id="hmb_00000000000000ee", model=MODEL, track="A",
                               layout=layout, th=th, meta=META)
    scores = layout.scores("fake_contestant_1", "A", "hmb_00000000000000ee")
    scores.parent.mkdir(parents=True, exist_ok=True)
    scores.write_bytes(b'{"S": 0.0}')
    cgx = layout.cgx("fake_contestant_1", "A", "hmb_00000000000000ee")
    cgx.parent.mkdir(parents=True, exist_ok=True)
    cgx.write_bytes(b"{}")
    with pytest.raises(ValueError):                  # a non-OK cell never reaches adjudication
        finalize_submission(stage, layout, cgx_path=cgx, scores_path=scores)
    with pytest.raises(ValueError):
        finalize_submission(stage, layout)
    final = finalize_submission(stage, layout, scores_path=scores, judge_config_id="judge-1")
    assert final.scores == "scores/fake_contestant_1/A/hmb_00000000000000ee.json"
    assert final.scores_sha256 == sha256_bytes(b'{"S": 0.0}') and final.cgx_record is None
    on_disk = json.loads(layout.submission_record("fake_contestant_1", "A", "hmb_00000000000000ee").read_text("utf-8"))
    assert on_disk["scores_sha256"] == final.scores_sha256 and on_disk["judge_config_id"] == "judge-1"
    assert finalize_submission(stage, layout, scores_path=scores, judge_config_id="judge-1") == final   # idempotent
    scores2 = scores.with_name("other.json")
    scores2.write_bytes(b'{"S": 1.0}')
    with pytest.raises(ImmutableConflict):          # written once
        finalize_submission(stage, layout, scores_path=scores2, judge_config_id="judge-1")


def test_t3_text_only_probe(th, layout):
    """§13.6 T3: context text, NO image, same harness, derived item id `__t3`, its own prompt_id."""
    from hmb.submission import build_t3_request, is_t3_item, run_t3_probe, t3_item_id
    from hmb.submission.prompts import load_t3_prompt
    pid, text, sha = load_t3_prompt()
    assert pid == "reconstruction-T3-v2" and "You are given no image." in text
    _a_pid, a_text, _ = load_prompt("A")
    for element in a_text.split("\n\n")[1:]:           # the six §9.1 elements, verbatim
        assert element.strip() in text
    req = build_t3_request(MODEL, "Fig. 7. Two tangent circles.", th, phase="FINAL", cache_scope="t")
    assert req.image_paths == [] and req.image_sha256 == [] and req.text.endswith("Fig. 7. Two tangent circles.")
    fake = FakeProvider("```latex\n\\usepackage{graphicx}\n\\begin{tikzpicture}\\end{tikzpicture}\n```")
    rec = run_t3_probe(item_id="hmb_00000000000000ff", model=MODEL, context_text="Fig. 7.", layout=layout,
                       th=th, ledger=FakeLedger(), allow_live=True, phase="FINAL", cache_scope="t",
                       meta={k: v for k, v in META.items() if k not in ("model_version_string", "image_encoding")},
                       provider_override=fake)
    assert rec.item_id == t3_item_id("hmb_00000000000000ff") == "hmb_00000000000000ff__t3" and is_t3_item(rec.item_id)
    assert rec.prompt_id == pid and rec.prompt_sha256 == sha and rec.image_encoding["format"] == "none"
    assert rec.outcome_code == Outcome.P_POLICY_REJECT.value         # same harness rules apply
    assert (layout.compile_dir("fake_contestant_1", "B", rec.item_id) / "submission_stage.json").exists()


def test_code_hash_covers_cgx_dependencies(tmp_path, monkeypatch):
    """harness_version's code hash covers every module that decides CGX output, including
    synthetic/geo_predicates.py, cards/closure.py and ontology.py; unrelated modules stay out."""
    import shutil
    from hmb.submission import process as PR
    src = PR.PKG_DIR.parent
    root = tmp_path / "hmb"
    shutil.copytree(src / "submission", root / "submission", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(src / "cgx", root / "cgx", ignore=shutil.ignore_patterns("__pycache__"))
    for rel in PR.CGX_DEPENDENCIES + ("grading/cli.py",):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, root / rel)
    monkeypatch.setattr(PR, "PKG_DIR", root / "submission")
    monkeypatch.setattr(PR, "CGX_DIR", root / "cgx")
    base = PR.code_hash()
    assert {"synthetic/geo_predicates.py", "cards/closure.py", "ontology.py"} <= set(PR.CGX_DEPENDENCIES)
    for rel in PR.CGX_DEPENDENCIES + ("cgx/extract.py", "submission/wrapper.py"):
        f = root / rel
        orig = f.read_bytes()
        f.write_bytes(orig + b"\n# changed\n")
        assert PR.code_hash() != base, rel
        f.write_bytes(orig)
    assert PR.code_hash() == base
    (root / "grading" / "cli.py").write_bytes(b"# unrelated change\n")          # not a CGX dependency
    assert PR.code_hash() == base


def test_single_harness_version_formula(th, monkeypatch):
    """ONE §15.1 formula: submission records and Opus 1's release build compute the same value."""
    from hmb.manifests import HARNESS_KEYS, _harness_object, harness_version_of
    from hmb.submission import process as PR
    ho = PR.harness_object(th)
    assert set(ho) == set(HARNESS_KEYS)
    assert PR.harness_version(th) == harness_version_of(ho)
    rel = _harness_object(th)
    assert rel["submission_harness_version_equal"] is True
    assert harness_version_of({k: rel[k] for k in HARNESS_KEYS}) == PR.harness_version(th)
    # the engine line (C16) is folded into wrapper_version
    before = PR.wrapper_version_id()
    monkeypatch.setattr(PR._sandbox, "ENGINE_LINE", PR._sandbox.ENGINE_LINE + " --safer")
    assert PR.wrapper_version_id() != before
