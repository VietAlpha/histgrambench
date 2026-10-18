"""§4 rights: rule us-pd-pre1931-v1, R1 draft, R3 value-checked gate, evidence, provenance view."""
import copy
import csv
import hashlib
import io
import json

import cv2
import fitz
import numpy as np
import pytest

from hmb import rights
from hmb.hashing import canonical_json
from hmb.schemas import RightsManifest
from hmb.store import Layout


def _png(mode_gray=True):
    img = np.full((40, 30), 200, np.uint8) if mode_gray else np.full((40, 30, 3), 200, np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def valid_record(approver="owner:alice", year=1905, published="yes"):
    sha = hashlib.sha256(b"master bytes").hexdigest()
    rec = {
        "book_id": "bk_" + sha[:12], "work_id": "wk_0031", "display_title": "T", "author": "A", "publisher": "P",
        "publisher_id": "pub_1", "edition_statement": "1st", "is_reprint_of_earlier_work": "no",
        "publication_year": year, "published": published, "publication_place": "Boston",
        "first_publication_country": "US", "language": "en",
        "acquisition": {"source_repository": "IA", "source_url": "u", "download_utc": "2026-09-21T00:00:00Z",
                        "source_file_sha256": sha, "source_file_bytes": 12, "page_count": 3,
                        "source_codec": "pdf_mixed", "source_bitdepth": 8, "source_colorspace": "gray"},
        "publication_year_evidence": {"catalog_record": "evidence/x/catalog.json", "catalog_record_sha256": "a" * 64,
                                      "titlepage_asset": "evidence/x/titlepage.png", "titlepage_asset_sha256": "b" * 64,
                                      "titlepage_page_index": 0, "agree": True},
        "terms": {"terms_url": "u", "terms_snapshot": "evidence/x/terms.html", "terms_snapshot_sha256": "c" * 64,
                  "terms_snapshot_utc": "t", "repository_rights_statement_verbatim": "v",
                  "redistribution_permitted": "yes", "commercial_reuse_permitted": "unknown",
                  "ai_use_clause": "unknown", "required_attribution_string": None},
        "determination": {"us_status": "not_determined", "us_rule_id": "us-pd-pre1931-v1", "us_determined_by": "rule",
                          "source_country": "US", "source_country_status": "not_determined",
                          "release_country_status": "not_determined", "determined_by_model": False},
        "prominence": {"hosted_by_major_aggregators": 2, "public_fulltext_layer": True},
        "release_mode": "full_public", "approved_by": approver, "approved_utc": "2026-09-22",
        "manifest_version": "rights-1"}
    rec["manifest_sha256"] = rights.manifest_sha256(rec)
    return rights.apply_us_rule(rec)


@pytest.mark.parametrize("year,published,expected", [
    (1930, "yes", "public_domain"), (1850, "yes", "public_domain"), (1931, "yes", "not_determined"),
    (1938, "yes", "not_determined"), (1905, "no", "not_determined"), (1905, "unknown", "not_determined"),
    (None, "yes", "not_determined"), ("1905", "yes", "not_determined"), (True, "yes", "not_determined")])
def test_rule_table(year, published, expected):
    assert rights.us_status_by_rule(year, published) == expected


def test_apply_rule_requires_agreeing_evidence():
    rec = valid_record()
    assert rec["determination"]["us_status"] == "public_domain"
    rec2 = copy.deepcopy(rec)
    rec2["publication_year_evidence"]["agree"] = False
    assert rights.apply_us_rule(rec2)["determination"]["us_status"] == "not_determined"


def test_manifest_sha256_excludes_own_field_and_tracks_content():
    rec = valid_record()
    body = {k: v for k, v in rec.items() if k != "manifest_sha256"}
    assert rec["manifest_sha256"] == hashlib.sha256(canonical_json(body)).hexdigest()
    other = copy.deepcopy(rec)
    other["author"] = "B"
    assert rights.manifest_sha256(other) != rec["manifest_sha256"]
    RightsManifest.model_validate(rec)


def test_r3_passes_valid_production_record():
    assert rights.r3_gate(valid_record(), "production") == {"passed": True, "failures": [], "profile": "production"}


def _rehash(rec):
    rec["manifest_sha256"] = rights.manifest_sha256(rec)
    return rec


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r["determination"].update(us_status="not_determined"), "us_status"),
    (lambda r: r.update(release_mode="excluded"), "release_mode"),
    (lambda r: r["publication_year_evidence"].update(agree=False), "year_evidence_agree"),
    (lambda r: r["terms"].update(redistribution_permitted="no", ) or r.update(release_mode="server_only"),
     "redistribution_permitted"),
    (lambda r: r.update(approved_by=None), "approved_by_unset"),
    (lambda r: r.update(approved_by="   "), "approved_by_unset"),
])
def test_r3_each_condition_is_value_checked(mutate, needle):
    rec = valid_record()
    mutate(rec)
    res = rights.r3_gate(_rehash(rec), "production")
    assert not res["passed"]
    assert any(needle in f for f in res["failures"])


def test_r3_presence_is_not_a_gate():
    # fields present but with non-passing or garbage values
    rec = valid_record()
    rec["terms"]["redistribution_permitted"] = "No"   # not the closed value "yes"/"unknown"
    rec["publication_year_evidence"]["agree"] = "true"  # a string is not True
    res = rights.r3_gate(_rehash(rec), "production")
    assert not res["passed"]
    assert any("redistribution" in f for f in res["failures"])
    assert any("agree" in f for f in res["failures"])


def test_r3_fixture_approver_rules():
    fx = valid_record(approver="fixture:smoke")
    assert rights.r3_gate(fx, "fixture")["passed"]
    assert "fixture_approver_in_production" in rights.r3_gate(fx, "production")["failures"]
    real = valid_record()
    assert "fixture_profile_requires_fixture_approver" in rights.r3_gate(real, "fixture")["failures"]


def test_r3_tamper_and_hostile_input():
    rec = valid_record()
    rec["release_mode"] = "server_only"          # changed without re-hashing
    assert "manifest_sha256_mismatch" in rights.r3_gate(rec, "production")["failures"]
    assert not rights.r3_gate("not a dict", "production")["passed"]
    assert not rights.r3_gate({}, "production")["passed"]
    bad = valid_record()
    bad["determination"]["determined_by_model"] = True
    assert any(f.startswith("schema_invalid") for f in rights.r3_gate(_rehash(bad), "production")["failures"])


def test_rule_cannot_make_post1930_public_domain_even_if_hand_set():
    rec = valid_record(year=1938)
    assert rec["determination"]["us_status"] == "not_determined"
    rec["determination"]["us_status"] = "public_domain"
    assert not rights.r3_gate(_rehash(rec), "production")["passed"]


def test_draft_is_mechanical_only(tmp_path):
    doc = fitz.open()
    pg = doc.new_page(width=200, height=300)
    pg.insert_image(fitz.Rect(10, 10, 110, 110), stream=_png())
    doc.new_page(width=200, height=300)
    p = tmp_path / "book.pdf"
    doc.save(p)
    d = rights.draft_manifest(p)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert d["book_id"] == "bk_" + sha[:12]
    assert d["acquisition"]["source_file_sha256"] == sha
    assert d["acquisition"]["page_count"] == 2
    assert (d["acquisition"]["source_codec"], d["acquisition"]["source_bitdepth"],
            d["acquisition"]["source_colorspace"]) == ("pdf_mixed", 8, "gray")
    for human in ("release_mode", "approved_by", "publication_year", "published"):
        assert d[human] is None
    assert d["terms"]["redistribution_permitted"] is None
    assert d["determination"]["us_status"] == "not_determined"
    assert d["determination"]["source_country_status"] == "not_determined"
    assert not rights.r3_gate(d, "production")["passed"]


def test_probe_image_modes(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png(mode_gray=False))
    pr = rights.probe_master(p)
    assert (pr["source_codec"], pr["source_bitdepth"], pr["source_colorspace"]) == ("png", 24, "rgb")
    q = tmp_path / "b.txt"
    q.write_bytes(b"hello")
    with pytest.raises(rights.RightsError):
        rights.probe_master(q)


def test_evidence_bundle_and_tamper(tmp_path):
    lay = Layout(tmp_path)
    rec = valid_record()
    bid = rec["book_id"]
    cat = rights.write_evidence(lay, bid, "catalog.json", b'{"year": 1905}')
    tp = rights.write_evidence(lay, bid, "titlepage.png", _png())
    te = rights.write_evidence(lay, bid, "terms.html", b"<html>terms</html>")
    assert cat["sha256"] == hashlib.sha256(b'{"year": 1905}').hexdigest()
    rec["publication_year_evidence"].update(catalog_record=cat["path"], catalog_record_sha256=cat["sha256"],
                                            titlepage_asset=tp["path"], titlepage_asset_sha256=tp["sha256"])
    rec["terms"].update(terms_snapshot=te["path"], terms_snapshot_sha256=te["sha256"])
    assert rights.verify_evidence(lay, rec) == []
    from hmb.cli import main
    lay.rights_manifest.parent.mkdir(parents=True, exist_ok=True)
    lay.rights_manifest.write_text(rights.dump_yaml([rights.apply_us_rule(dict(
        rec, manifest_sha256=rights.manifest_sha256(rec)))]).decode(), encoding="utf-8")
    assert main(["--root", str(lay.root), "rights-check"]) == 0          # evidence re-hash clean
    (lay.evidence_dir(bid) / "terms.html").write_bytes(b"changed")
    assert any("terms.html" in x for x in rights.verify_evidence(lay, rec))
    assert main(["--root", str(lay.root), "rights-check"]) == 2          # approved book, changed evidence
    with pytest.raises(rights.RightsError):
        rights.write_evidence(lay, bid, "../escape.txt", b"x")
    idx = json.loads((lay.evidence_dir(bid) / "index.json").read_text())
    assert set(idx) == {"catalog.json", "titlepage.png", "terms.html"}


def test_manifest_file_and_provenance_view(tmp_path):
    lay = Layout(tmp_path)
    a, b = valid_record(), valid_record(approver="owner:bob")
    b["book_id"] = "bk_" + "f" * 12
    lay.rights_manifest.parent.mkdir(parents=True)
    lay.rights_manifest.write_text(rights.dump_yaml([a, b]).decode(), encoding="utf-8")
    mans = rights.load_manifests(lay.rights_manifest)
    assert set(mans) == {a["book_id"], b["book_id"]}
    rights.write_provenance_csv(lay)
    rows = list(csv.DictReader(io.StringIO(lay.provenance_csv.read_text(encoding="utf-8"))))
    assert [r["book_id"] for r in rows] == sorted([a["book_id"], b["book_id"]])
    assert rows[0]["us_status"] in ("public_domain",)
    lay.rights_manifest.write_text(rights.dump_yaml([a, a]).decode(), encoding="utf-8")
    with pytest.raises(rights.RightsError):
        rights.load_manifests(lay.rights_manifest)


def test_benchmark_card_sentence_is_verbatim():
    s = rights.BENCHMARK_CARD_SENTENCE
    assert s.startswith("Rights status was determined once per source book.")
    assert "Circulars 15A and 38B" in s and s.endswith("independently of their public-domain status.")
