"""Stage-aware level access (2026-09-25; milestones 2026-09-26): MS1 -> pre-freeze PILOT/DEV -> level freeze ->
production, through the CLI.

The production profile and the real pipeline run throughout; the only substitute is the model transport,
replaced at `hmb.providers.adapter_for` by an explicit stub (no network), so responses are treated as real
calls under the production profile. Nothing here is benchmark evidence.
"""
import json
import re

import cv2
import numpy as np
import pytest
import yaml

from hmb import providers
from hmb.cli import main
from hmb.config import (CalibrationMissing, ThresholdsError, fixture_thresholds, load_thresholds,
                        resolve_thresholds_arg)
from hmb.hashing import sha256_file
from hmb.store import Layout, iter_jsonl, read_json, write_mutable

from w2_scenario import triangle_json

DEV, TEST = "bk_dev000000001", "bk_tst000000001"
PRICING = ("version: pricing-1\nentries:\n  - unit_price_id: p0\n    model_id: stub\n    currency: USD\n"
           "    usd_per_mtok_input: 0\n    usd_per_mtok_output: 0\n    usd_per_image: 0\n"
           "    pricing_url: https://invalid.example/zero\n    retrieved_utc: '2026-09-25'\n")


class StubTransport:
    """Stands in for the Bedrock transport: generation answers a triangle, verification votes true."""
    calls = 0

    def send(self, req, images):
        StubTransport.calls += 1
        if req.purpose == "contestant":                   # a correct triangle ABC (MS3 probe, C42 fix)
            text = ("```latex\n\\begin{tikzpicture}\n\\draw (0,0) -- (4,0) -- (2,3) -- cycle;\n"
                    "\\fill (0,0) circle (1.5pt) node[below left] {A};\n\\fill (4,0) circle (1.5pt) node[below right] {B};\n"
                    "\\fill (2,3) circle (1.5pt) node[above] {C};\n\\end{tikzpicture}\n```")
            return {"text": text, "stop_reason": "end_turn", "input_tokens": 10, "output_tokens": 10,
                    "model_version_string": req.spec.model_id}
        if req.purpose == "card_generation":
            ocr = json.loads(req.text.split("OCR_WORDS: ", 1)[1].splitlines()[0])
            text = triangle_json(req.spec.model_id, labels=tuple(w["text"] for w in ocr) or ("A", "B", "C"))
        else:
            text = json.dumps({"votes": [{"id": s, "vote": "true"} for s in re.findall(r"^(s\d+):", req.text, re.M)]})
        return {"text": text, "stop_reason": "end_turn", "input_tokens": 10, "output_tokens": 10,
                "model_version_string": req.spec.model_id}


def cli(lay, *args):
    return main(["--root", str(lay.root), *args])


def roster(tmp_path) -> str:
    def m(mid, fam, **kw):
        return {"model_id": mid, "provider": "bedrock", "family": fam, "snapshot": mid, "unit_price_id": "p0",
                "effort": "medium", "decoding": {"additional_model_request_fields": {"effort": "medium"}}, **kw}
    d = {"version": "roster-1", "builder": [m("w1", "fam_a"), m("w2", "fam_b")],
         "judge": [m("g1", "fam_q"), m("g2", "fam_r"), m("g3", "fam_s")]}
    p = tmp_path / "roster.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return str(p)


def candidate(lay, iid, book, k):
    img = np.full((300, 300), 255, np.uint8)
    for a, b in [((40, 260), (260, 260)), ((260, 260), (150, 40)), ((150, 40), (40, 260))]:
        cv2.line(img, a, b, 0, 2 + k)
    crop = lay.crop_canonical(iid)
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    rec = {"id": iid, "dataset_version": "v1.0", "source": {"book_id": book, "page_index": k},
           "extraction": {"crop_bbox": [0, 0, 300, 300]},
           "assets": {"crop_canonical_image": f"crops_canonical/{iid}.png", "crop_canonical_sha256": sha256_file(crop)},
           "ocr_words": [{"text": t, "box": [0, 0, .05, .05]} for t in ("ABC", "DEF", "GHI")[k]],
           "automatic_metadata": {"image_stats": {"stats_version": "imgstat-1", "blur_norm": 0.2 + 0.1 * k,
                                                  "ink_frac": 0.05 + 0.01 * k}}}
    write_mutable(lay.p("data", "candidates", f"{iid}.json"), json.dumps(rec).encode())


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A production root after MS1: template, form freeze and the owner's approval; transport stubbed."""
    monkeypatch.setattr(providers, "adapter_for", lambda provider: StubTransport())
    lay = Layout(tmp_path / "root")
    assert cli(lay, "thresholds-init", "--profile", "production") == 0
    with pytest.raises(ThresholdsError, match="MS1"):
        resolve_thresholds_arg("prefreeze:DEV", lay)                  # pre-freeze work follows MS1
    assert cli(lay, "form-freeze") == 0
    assert cli(lay, "form-freeze", "--record-owner-approval", "--actor", "owner:test") == 0
    write_mutable(lay.pricing, PRICING.encode())
    return lay


def provisional(lay, tmp_path, values) -> int:
    p = tmp_path / "prov_values.json"
    p.write_text(json.dumps(values), encoding="utf-8")
    return cli(lay, "thresholds-provisional", "--values", str(p))


def test_prefreeze_context_reads_only_its_levels(root, tmp_path):
    th = resolve_thresholds_arg("prefreeze:DEV", root)
    assert th.is_production and th.context == "DEV"
    assert th.value("g0b_min_xheight_px") == 12 and th.reads == {"g0b_min_xheight_px": 12}   # plan value, recorded
    with pytest.raises(CalibrationMissing, match="not readable"):
        th.value("F_bar_min")                                         # a test-set level: never before the freeze
    with pytest.raises(CalibrationMissing, match="pre-freeze value"):
        th.value("contestant_max_tokens")                             # readable by a development run, once set
    with pytest.raises(CalibrationMissing, match="pre-freeze value"):
        th.value("g8_max_tokens")                                     # calibrated: no value yet
    assert provisional(root, tmp_path, {"g0b_min_xheight_px": {"value": 5, "source": "x", "rule": "x"}}) == 2
    assert provisional(root, tmp_path, {"g8_max_tokens": {"value": 512, "source": "pilot metering", "rule": "dev"},
                                        "delta_star": {"value": 0.05, "source": "owner", "rule": "dev"}}) == 0
    th = resolve_thresholds_arg("prefreeze:DEV", root)
    assert th.value("g8_max_tokens") == 512
    assert th.value("delta_star") == 0.05                            # the owner's pre-freeze δ* (DEV sizing)
    pilot = resolve_thresholds_arg("prefreeze:PILOT", root)
    assert pilot.value("g8_max_tokens") == 512
    with pytest.raises(CalibrationMissing, match="not readable"):
        pilot.value("delta_star")                                     # sizing is MS3, not the pilot
    prod = load_thresholds(root.thresholds)                           # no context: the freeze still binds
    with pytest.raises(CalibrationMissing, match="level freeze"):
        prod.value("g0b_min_xheight_px")
    for gone in ("CAL", "CP6", "TEST"):                               # the CAL control context was deleted
        with pytest.raises(ThresholdsError, match="unknown pre-freeze context"):
            resolve_thresholds_arg(f"prefreeze:{gone}", root)
    fx = resolve_thresholds_arg("fixture", root)                      # fixture behaviour unchanged
    assert not fx.is_prefreeze and fx.value("g8_max_tokens") == fixture_thresholds().value("g8_max_tokens")


def test_development_set_then_freeze_then_production(root, tmp_path, capsys):
    r = roster(tmp_path)
    from hmb.acquisition.drivers import declare_development_book
    declare_development_book(root, DEV, "Dev book.pdf", "DEV")        # what acq-run does in prefreeze:DEV
    for k in range(3):
        candidate(root, f"hmb_00000000000000d{k}", DEV, k)
    candidate(root, "hmb_00000000000000e0", TEST, 0)                  # not a development book
    assert provisional(root, tmp_path, {
        "card_builder_max_tokens": {"value": 4096, "source": "pilot metering", "rule": "pre-freeze development value"}}) == 0
    ctx = ["--thresholds", "prefreeze:DEV"]
    assert cli(root, "cards-build", "--roster", r, *ctx, "--phase", "TEST", "--allow-live") == 2  # test cards: MS4
    assert cli(root, "cards-build", "--roster", r, *ctx, "--phase", "DEV", "--allow-live") == 0
    assert StubTransport.calls > 0
    gates = read_json(root.p("data", "card_gates", "hmb_00000000000000d0.json"))
    assert gates["levels"]["levels"] == "provisional" and gates["levels"]["context"] == "DEV"
    assert gates["profile"] == "production"
    assert not root.card("hmb_00000000000000e0").exists()             # the non-development book is not built
    capsys.readouterr()
    assert cli(root, "dev-measure", *ctx) == 0
    art = _last_json(capsys, root)
    doc = read_json(root.p(*art.split("/")))
    ec = doc["execution_context"]
    assert ec["levels"] == "provisional" and ec["partition"] == "development" and ec["stage"].startswith("MS3")
    assert ec["form_freeze_sha256"] and ec["registry_hash"] and ec["thresholds_hash"]
    assert doc["measurements"]["n_carded"] == 3 and "w_p" not in doc["measurements"]
    assert set(doc["inputs"]["cards"]) == {f"hmb_00000000000000d{k}" for k in range(3)}
    runs = list(iter_jsonl(root.p("manifests", "prefreeze_runs.jsonl")))
    build = next(x for x in runs if x["command"].startswith("cards-build"))
    assert build["params_read"]["card_builder_max_tokens"] == 4096
    assert build["outputs"] and all(len(v) == 64 for v in build["outputs"].values())
    # MS3 probe (C42 fix): development items run before the freeze, the test split never does
    from hmb.pipeline import RunRefused, development_probe_items, require_official_run
    from hmb.roster import LOCKED_CONTESTANTS
    from hmb.schemas import ModelSpec
    route = next(x for x in LOCKED_CONTESTANTS if x.image_input)
    probe = ModelSpec(model_id=route.model_id, provider=route.provider, family=route.family, role="contestant",
                      endpoint=route.endpoint, input_mode="vision")
    dth = resolve_thresholds_arg("prefreeze:DEV", root)
    require_official_run(root, dth, probe, [], fake=False, split="development")          # no freeze, release, window
    with pytest.raises(RunRefused, match="development books"):
        require_official_run(root, dth, probe, [], fake=False, split="test")
    assert {r["id"] for r in development_probe_items(root)} == {f"hmb_00000000000000d{k}" for k in range(3)}
    capsys.readouterr()
    assert cli(root, "model-run", "--roster", r, "--model-id", "not-a-contestant", "--track", "A", "--split",
               "development", "--access-date", "d", "--provider-terms-version", "t", "--thresholds",
               "prefreeze:DEV") == 2
    assert "not a roster contestant" in capsys.readouterr().out      # the pre-freeze context passed resolution
    # the frozen drift-probe triple set from a judged development cell (verdict file written by hand here)
    from hmb.schemas import Card
    card = Card.model_validate(read_json(root.card("hmb_00000000000000d0")))
    f0, f1 = card.facts[0].fact_id, card.facts[1].fact_id
    write_mutable(root.verdicts("m1", "A", "hmb_00000000000000d0"), json.dumps({
        "item_id": "hmb_00000000000000d0", "model_id": "m1", "track": "A", "judge_config_hash": "h" * 64,
        "profile": "production", "facts": [{"fact_id": f0, "verdict": "PRESENT", "source": "judges"},
                                           {"fact_id": f1, "verdict": "ABSENT", "source": "extractor"}]}).encode())
    write_mutable(root.compile_dir("m1", "A", "hmb_00000000000000d0") / "render.png", b"png")
    capsys.readouterr()
    assert cli(root, "drift-triples", "--seed", "1") == 0
    trip = read_json(root.p("manifests", "drift_probe", "triples.json"))
    assert [(t["reference"], t["render_png"]) for t in trip] == [(True, "compiled/m1/A/hmb_00000000000000d0/render.png")]
    assert trip[0]["fact_text"] and trip[0]["triple_id"].endswith(f0)     # only judge-decided facts are triples
    # pre-freeze production/test paths still refuse
    assert cli(root, "items-finalize", "--release-version", "v1", "--seed", "1", "--thresholds", "prefreeze:DEV") == 2
    assert cli(root, "score-report", "--models", "m", "--track", "A", "--thresholds", "prefreeze:DEV") == 2
    assert cli(root, "release-build", "--version", "1.0.0") != 0
    assert cli(root, "build", "--roster", r, "--seed", "1") == 2
    assert "needs the level freeze" in capsys.readouterr().out
    # the level freeze can take a value from a pre-freeze development artifact (plumbing only: the field
    # is arbitrary here), and every calibrated parameter needs a provenance record
    dev_rel = art
    prov = {"card_builder_max_tokens": {"artifact": dev_rel, "field": "measurements.n_carded"}}
    for name in ("g8_max_tokens", "contestant_max_tokens", "judge_max_tokens", "delta_star", "N_star", "G_star"):
        prov[name] = {"value": 1, "calibration_artifact_id": "owner", "sha256": "0" * 64, "date": "2026-09-25",
                      "rule": "test"}
    bad = dict(prov, card_builder_max_tokens={"artifact": "manifests/form_freeze.json", "field": "version"})
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    assert cli(root, "thresholds-freeze", "--provenance", str(tmp_path / "bad.json")) == 2   # not a pre-freeze artifact
    (tmp_path / "prov.json").write_text(json.dumps(prov), encoding="utf-8")
    assert cli(root, "thresholds-freeze", "--provenance", str(tmp_path / "prov.json")) == 0
    th = load_thresholds(root.thresholds)
    w = th.params["card_builder_max_tokens"]["provenance"]
    assert w["calibration_artifact_id"] == dev_rel and w["sha256"] == sha256_file(root.p(*dev_rel.split("/")))
    assert th.value("card_builder_max_tokens") == 3
    assert th.params["g0b_min_xheight_px"]["provenance"]["rule"].startswith("plan starting value confirmed")
    # post-freeze: production reads frozen levels; the pre-freeze contexts are closed
    assert th.value("g0b_min_xheight_px") == 12 and th.value("contestant_max_tokens") == 1
    with pytest.raises(ThresholdsError, match="frozen"):
        resolve_thresholds_arg("prefreeze:DEV", root)
    assert cli(root, "develop", "--roster", r, "--source", "x.pdf") == 2
    capsys.readouterr()
    assert cli(root, "build", "--roster", r, "--seed", "1") == 2
    assert "no approved book" in capsys.readouterr().out            # past the freeze check


def _last_json(capsys, root) -> str:
    out = capsys.readouterr().out
    doc = json.loads(out[out.index("{"):])
    return doc["artifact"].replace("\\", "/").split(str(root.root).replace("\\", "/") + "/")[-1]


@pytest.mark.docker
def test_prefreeze_development_probe_runs_end_to_end(root, tmp_path):
    """C42 fix: `model-run --split development --thresholds prefreeze:DEV` scores development items through the real
    sandbox, CGX, adjudication and scoring before any level freeze, release or scoring window; no test artifact."""
    from hmb.submission.sandbox import sandbox_available
    if not sandbox_available():
        pytest.skip("pinned sandbox image unavailable (Docker)")
    from hmb.acquisition.drivers import declare_development_book
    from hmb.roster import LOCKED_CONTESTANTS
    route = next(x for x in LOCKED_CONTESTANTS if x.image_input)
    d = yaml.safe_load(open(roster(tmp_path), encoding="utf-8"))
    d["contestant"] = [{"model_id": route.model_id, "provider": route.provider, "family": route.family,
                        "snapshot": route.model_id, "unit_price_id": "p0", "input_mode": "vision",
                        **({"endpoint": route.endpoint} if route.endpoint else {})}]
    r = tmp_path / "roster_probe.yaml"
    r.write_text(yaml.safe_dump(d), encoding="utf-8")
    declare_development_book(root, DEV, "Dev book.pdf", "DEV")
    for k in range(3):
        candidate(root, f"hmb_00000000000000d{k}", DEV, k)
    assert provisional(root, tmp_path, {n: {"value": v, "source": "pilot metering", "rule": "test"} for n, v in (
        ("card_builder_max_tokens", 4096), ("contestant_max_tokens", 2048), ("judge_max_tokens", 1024))}) == 0
    ctx = ["--thresholds", "prefreeze:DEV"]
    assert cli(root, "cards-build", "--roster", str(r), *ctx, "--phase", "DEV", "--allow-live") == 0
    assert cli(root, "model-run", "--roster", str(r), "--model-id", route.model_id, "--track", "A", "--split",
               "development", "--access-date", "2026-09-26", "--provider-terms-version", "t", *ctx, "--allow-live") == 0
    scores = sorted(root.p("scores").rglob("hmb_*.json"))
    assert len(scores) == 3
    recs = [read_json(p) for p in scores]
    assert all(x["profile"] == "production" and x["outcome_code"] == "OK" and x["S"] is not None for x in recs), recs
    assert not root.test_artifacts_exist() and not root.release_manifest.exists()
    runs = [x for x in iter_jsonl(root.p("manifests", "prefreeze_runs.jsonl")) if x["command"].startswith("model-run")]
    assert runs and runs[-1]["context"] == "DEV" and "contestant_max_tokens" in runs[-1]["params_read"]
