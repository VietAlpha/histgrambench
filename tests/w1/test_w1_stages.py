"""The milestone stage commands (hmb.stages) and the MS1 owner-approval check they and release-build rely on."""
from __future__ import annotations

import json

import yaml

from hmb.cli import main
from hmb.manifests import cp1_approval_problems
from hmb.roster import LOCKED_CONTESTANTS, request_shape
from hmb.store import Layout, safe_model_dir, write_json_immutable

MANTLE = "https://bedrock-mantle.us-east-1.api.aws"


# three sample contestants of the locked roster (owner decision 2026-09-26; names kept from the 4-model era)
CLAUDE, LLAMA, QWEN = ("us.meta.llama4-maverick-17b-instruct-v1:0", "moonshotai.kimi-k2.5", "google.gemma-4-26b-a4b")


def roster(tmp_path):
    """The 2026-09-23 roster shape with placeholder ids for the other roles (never the owner's file) and the
    contestants locked on 2026-09-25."""
    def m(mid, prov, fam, **kw):
        return {"model_id": mid, "provider": prov, "family": fam, "snapshot": mid, "unit_price_id": "p-" + mid, **kw}
    conv = lambda e: {"additional_model_request_fields": {"effort": e}}  # noqa: E731
    d = {"version": "roster-1", "inclusion_rule_date": "2026-09-23",
         "verifier": [m("gk", "bedrock", "openai", effort="high", decoding=conv("high"))],
         "builder": [m(f"w{k}", "bedrock", f, effort="medium", decoding=conv("medium"))
                     for k, f in enumerate(("moonshot", "openai"), 1)],
         "judge": [m("g1", "bedrock", "qwen", effort="high", decoding=conv("high")),
                   m("google.gemma-4-31b", "bedrock_mantle", "google", effort="high", endpoint=MANTLE + "/openai/v1"),
                   m("g3", "bedrock", "mistral", effort="high", decoding=conv("high"))],
         "contestant": [m(r.model_id, r.provider, r.family, input_mode="vision" if r.image_input else "text_only",
                          **({"endpoint": r.endpoint} if r.endpoint else {}),
                          **({"decoding": {"temperature": 0.0}} if request_shape(r.model_id).get("sampling_exclusive")
                             else {})) for r in LOCKED_CONTESTANTS]}
    p = tmp_path / "roster.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return str(p)


def cli(root, *args):
    return main(["--root", str(root), *args])


def dry(root, capsys, *args) -> list[str]:
    capsys.readouterr()
    assert cli(root, *args, "--dry-run") == 0
    return [s["command"] for s in json.loads(capsys.readouterr().out)]


def test_setup_then_owner_approval(tmp_path, capsys):
    root = tmp_path / "r"
    r = roster(tmp_path)
    assert cli(root, "setup", "--roster", r) == 0
    lay = Layout(root)
    assert lay.thresholds.exists() and lay.p("manifests", "form_freeze.json").exists()
    assert cp1_approval_problems(lay)                                 # not approved yet
    assert cli(root, "setup", "--roster", r, "--approve", "--actor", "owner:test") == 0
    assert cp1_approval_problems(lay) == []
    # a second setup does not re-create the thresholds template
    assert "thresholds-init" not in " ".join(dry(root, capsys, "setup", "--roster", r))


def test_setup_stops_at_the_failing_step(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: roster-1\n", encoding="utf-8")          # every role empty
    assert cli(tmp_path / "r", "setup", "--roster", str(bad)) != 0
    out = capsys.readouterr().out
    assert "STOPPED at step 2/" in out and "roster check" in out and "rerun `hmb setup" in out


def test_stage_plans(tmp_path, capsys):
    root = tmp_path / "r"
    r = roster(tmp_path)
    for gone in ("controls", "calibrate"):                           # CP3 and its calibration stage were deleted
        try:
            cli(root, gone, "--dry-run")
            raise AssertionError(gone)
        except SystemExit:
            pass
    assert dry(root, capsys, "freeze", "--provenance", "p.json")[0] == "hmb thresholds-freeze --provenance p.json"
    run = dry(root, capsys, "run", "--roster", r, "--access-date", "2026-10-01", "--provider-terms-version", "t1")
    # the text-only DeepSeek V3.1 never runs the direct-figure tracks: the 14 image contestants are scheduled
    img = [x.model_id for x in LOCKED_CONTESTANTS if x.image_input]
    n = len(img)
    assert run[0].startswith("hmb drift-probe --window open") and run[n + 1].startswith("hmb drift-probe --window close")
    assert [c.split()[1] for c in run[1:n + 1]] == ["model-run"] * n
    assert [c.split()[1] for c in run[n + 2:]] == ["model-close"] * n
    assert not any("deepseek" in c for c in run)
    rep = dry(root, capsys, "report", "--roster", r)
    dirs = " ".join(safe_model_dir(m) for m in img)
    assert rep[0].startswith(f"hmb score-report --models {dirs} --track")
    assert rep[1].split()[1] == "gates-evaluate" and "--validation" not in rep[1]
    # build refuses before the level freeze: production acquisition and cards read frozen levels
    assert cli(root, "build", "--roster", r, "--seed", "1") == 2
    assert "needs the level freeze" in capsys.readouterr().out


def test_prefreeze_stage_plans(tmp_path, capsys):
    """pilot (MS2) and develop (MS3) run their single commands in the named pre-freeze context."""
    root = tmp_path / "r"
    r = roster(tmp_path)
    dev = dry(root, capsys, "develop", "--roster", r, "--source", "Book.pdf", "--provisional", "v.json")
    assert [c.split()[1] for c in dev] == ["thresholds-provisional", "acq-run", "acq-g8", "cards-build", "dev-measure"]
    assert all("prefreeze:DEV" in c for c in dev[1:]) and "--phase DEV" in dev[3]
    pilot = dry(root, capsys, "pilot", "--roster", r, "--source", "Book.pdf")
    assert [c.split()[1] for c in pilot] == ["acq-run", "acq-g8", "cards-build"] and "--phase PILOT" in pilot[2]
    assert all("prefreeze:PILOT" in c for c in pilot)


def test_run_resumes_after_the_window_closed(tmp_path, capsys):
    root = tmp_path / "r"
    r = roster(tmp_path)
    lay = Layout(root)
    for w in ("open", "close"):
        write_json_immutable(lay.p("manifests", "drift_probe", f"{w}.json"), {"window": w})
    write_json_immutable(lay.run_manifest(CLAUDE), {"x": 1})
    run = dry(root, capsys, "run", "--roster", r, "--access-date", "d", "--provider-terms-version", "t")
    assert run == [f"hmb model-close --roster {r} --model-id {m} --track A --track B --thresholds {lay.thresholds}"
                   for m in [x.model_id for x in LOCKED_CONTESTANTS if x.image_input] if m != CLAUDE]


def test_approved_sources_from_rights_evidence(tmp_path):
    from test_w1_scaffold4 import manifest, write_manifests

    from hmb.stages import approved_sources
    lay = Layout(tmp_path)
    ok, pending = manifest(0), manifest(1, approver="fixture:smoke")    # a fixture approver fails production R3
    write_manifests(lay, [ok, pending, manifest(2)])
    write_json_immutable(lay.evidence_dir(ok["book_id"]) / "master.json", {"file_name": "Book zero.pdf"})
    srcs, problems = approved_sources(lay)
    assert srcs == ["Book zero.pdf"]                                    # book 1 fails production R3
    assert len(problems) == 1 and manifest(2)["book_id"] in problems[0]
