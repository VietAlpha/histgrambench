"""§15 hash chain, release/run manifests, tombstones, version triggers, plotting rule."""
import hashlib
import json

import pytest

from hmb import manifests as M
from hmb.store import Layout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_chain_hashes_match_independent_encoding():
    page = b"\x89PNG fake page bytes"
    assert M.page_hash(page) == sha(page)
    ph = M.page_hash(page)
    crop = b"crop-bytes"
    # H(crop || version || page_hash || bbox) with 0x1F separators; bbox as canonical JSON
    expect = sha(crop + b"\x1f" + b"extract-1.0.0" + b"\x1f" + ph.encode() + b"\x1f" + b"[1,2,30,40]")
    assert M.item_hash(crop, "extract-1.0.0", ph, [1, 2, 30, 40]) == expect
    pairs = [("b" * 64, "c" * 64), ("a" * 64, "d" * 64)]
    expect_cs = sha(json.dumps(sorted([list(p) for p in pairs]), separators=(",", ":")).encode())
    assert M.cardset_hash(pairs) == expect_cs
    assert M.cardset_hash(list(reversed(pairs))) == expect_cs


def _hashes():
    return {k: sha(k.encode()) for k in M.RELEASE_KEYS}


HARNESS = {"eval_commit": "abc123", "container_digest": "sha256:" + "d" * 64, "tex_distribution": "texlive-2025",
           "wrapper_version": "wrapper-1", "extraction_rules_version": "extract-1", "rasterizer": "pdftoppm 24.02",
           "dpi": 300, "canvas": "1024x1024"}


def test_release_hash_and_verify():
    h = _hashes()
    rel = M.build_release(version="1.0.0", created="2026-09-22", hashes=h, thresholds_frozen_at="2026-09-01",
                          harness=HARNESS, counts={"books": 1, "split_groups": 1, "items_eligible": 2,
                                                   "facts_eligible": 10}, funnel_path="manifests/funnel_1.0.0.json")
    hv = sha(b"\x1f".join([b"abc123", HARNESS["container_digest"].encode(), b"wrapper-1", b"extract-1",
                           b'{"canvas":"1024x1024","dpi":300,"rasterizer":"pdftoppm 24.02"}']))
    parts = [h["cardset_hash"], h["split_hash"], h["thresholds_hash"], h["judge_config_hash"], hv,
             h["metric_hash"]]
    assert rel["release_hash"] == "sha256:" + sha(b"\x1f".join(p.encode() for p in parts))
    assert "harness_version" not in rel and rel["harness"]["tex_distribution"] == "texlive-2025"
    assert rel["data_licence"]["L4_cards_and_metadata"] == "Apache-2.0"            # D9, 2026-09-23
    assert "rights_manifest" in rel["data_licence"]["scans_and_crops"]             # not a project licence
    tampered = dict(rel, harness=dict(HARNESS, dpi=600))
    assert not M.verify_release(tampered)
    with pytest.raises(M.ManifestError):
        M.build_release(version="1.0.0", created="c", hashes=h, thresholds_frozen_at="t", harness={"dpi": 300},
                        counts={"books": 1, "split_groups": 1, "items_eligible": 0, "facts_eligible": 0},
                        funnel_path="f")
    assert rel["benchmark_version"] == "histmathbench-1.0.0"
    assert M.verify_release(rel)
    rel["metric_hash"] = sha(b"other")
    assert not M.verify_release(rel)


@pytest.mark.parametrize("kw", [dict(version="1.0"), dict(thresholds_frozen_at=None),
                                dict(counts={"books": -1, "split_groups": 1, "items_eligible": 0, "facts_eligible": 0})])
def test_release_rejects_bad_inputs(kw):
    base = dict(version="1.0.0", created="c", hashes=_hashes(), thresholds_frozen_at="t", harness=HARNESS,
                counts={"books": 1, "split_groups": 1, "items_eligible": 0, "facts_eligible": 0}, funnel_path="f")
    base.update(kw)
    with pytest.raises(M.ManifestError):
        M.build_release(**base)
    bad = _hashes()
    bad["split_hash"] = "not-a-hash"
    with pytest.raises(M.ManifestError):
        M.build_release(**{**base, "version": "1.0.0", "thresholds_frozen_at": "t", "hashes": bad,
                           "counts": {"books": 1, "split_groups": 1, "items_eligible": 0, "facts_eligible": 0}})


def test_write_release_immutable(tmp_path):
    lay = Layout(tmp_path)
    rel = M.build_release(version="1.0.0", created="c", hashes=_hashes(), thresholds_frozen_at="t", harness=HARNESS,
                          counts={"books": 1, "split_groups": 1, "items_eligible": 0, "facts_eligible": 0},
                          funnel_path="f")
    p = M.write_release(lay, rel)
    assert p.exists() and lay.release_manifest.exists()
    rel2 = dict(rel, created="other")
    with pytest.raises(Exception):
        M.write_release(lay, rel2)          # same version, different bytes


def test_run_manifest_requires_fields():
    run = M.build_run_manifest(release_hash_value="sha256:" + "a" * 64,
                               model_record={"model_id": "m", "provider_version": "v", "family": "f",
                                             "access_date": "d", "decoding": {}, "attempts": 1},
                               prompt_sha256="b" * 64, scoring_window={"open": "o", "close": "c"},
                               drift_probe={"open_agreement": 0.9, "close_agreement": 0.9, "kappa_min": 0.8,
                                            "pass": True}, verdicts_hash="c" * 64, funnel_eligibility_rate=0.4,
                               output_release="public")
    assert run["release_hash"] == "sha256:" + "a" * 64
    with pytest.raises(M.ManifestError):
        M.build_run_manifest(release_hash_value="a" * 64, model_record={"model_id": "m"}, prompt_sha256="b" * 64,
                             scoring_window={"open": "o", "close": "c"}, drift_probe={}, verdicts_hash="c" * 64,
                             funnel_eligibility_rate=None, output_release="public")


def test_tombstones_never_reuse(tmp_path):
    lay = Layout(tmp_path)
    M.add_tombstone(lay, "hmb_" + "0" * 16, "1.1.0", "rights", "evidence/x")
    M.add_tombstone(lay, "hmb_" + "1" * 16, "1.2.0", "rights", "evidence/y")
    with pytest.raises(M.ManifestError):
        M.add_tombstone(lay, "hmb_" + "0" * 16, "1.3.0", "again", "e")
    with pytest.raises(M.ManifestError):
        M.assert_id_usable(lay, "hmb_" + "1" * 16)
    M.assert_id_usable(lay, "hmb_" + "2" * 16)
    assert [s["retired_in_version"] for s in M.load_tombstones(lay)] == ["1.1.0", "1.2.0"]


@pytest.mark.parametrize("change,bump", [("card_content", "MAJOR"), ("judge_snapshot", "MAJOR"), ("dpi", "MAJOR"),
                                         ("split_assignment", "MAJOR"), ("add_books_or_items", "MINOR"),
                                         ("rights_retirement", "MINOR"), ("documentation", "PATCH")])
def test_version_triggers(change, bump):
    assert M.version_bump(change) == bump


def test_next_version_and_unknown_change():
    assert M.next_version("1.4.2", ["documentation", "add_analysis_slice"]) == "1.5.0"
    assert M.next_version("1.4.2", ["paths"]) == "1.4.3"
    assert M.next_version("1.4.2", ["paths", "metric_caps"]) == "2.0.0"
    with pytest.raises(M.ManifestError):
        M.version_bump("tweak_something")
    with pytest.raises(M.ManifestError):
        M.next_version("1.4.2", [])


def test_plot_axis_rule():
    assert M.check_plot_axis([{"release_hash": "sha256:x"}, {"release_hash": "sha256:x"}]) == "sha256:x"
    with pytest.raises(M.ManifestError):
        M.check_plot_axis([{"release_hash": "sha256:x"}, {"release_hash": "sha256:y"}])
    with pytest.raises(M.ManifestError):
        M.check_plot_axis([{"model": "m"}])
