"""Scaffold 0 contract tests: valid records pass, materially invalid boundary inputs fail."""
import copy
import json

import pytest
from pydantic import ValidationError

from hmb import hashing, ontology
from hmb.config import (CalibrationMissing, SPECS, ThresholdsError, fixture_thresholds,
                        level_freeze, load_thresholds, production_template)
from hmb.schemas import BuilderOutput, FactRecord, RightsManifest, SubmissionRecord
from hmb.store import FreezeViolation, ImmutableConflict, Layout, append_jsonl, iter_jsonl, write_immutable
from hmb.vocab import Exclusion, Outcome

SHA = "a" * 64


# ---------------------------------------------------------------- hashing (§5.1, §15.1)
def test_canonical_json_rfc8785_examples():
    # Key ordering by UTF-16 code units, no whitespace, ES6 numbers.
    assert hashing.canonical_json({"b": 1, "a": [1.0, 0.5, 1e21, 1e-7]}) == b'{"a":[1,0.5,1e+21,1e-7],"b":1}'
    assert hashing.canonical_json({"\u20ac": 1, "\r": 2}) == '{"\\r":2,"\u20ac":1}'.encode()
    assert hashing.canonical_json(1e20) == b"100000000000000000000"
    assert hashing.canonical_json(-0.000001) == b"-0.000001"
    assert hashing.canonical_json(123.456) == b"123.456"
    with pytest.raises(hashing.CanonicalJSONError):
        hashing.canonical_json(float("nan"))


def test_nfc_normalisation_changes_nothing_semantically():
    decomposed = "Hi\u0301nh"   # i + combining acute
    composed = "H\u00ednh"
    assert hashing.canonical_json(decomposed) == hashing.canonical_json(composed)


def test_content_ids():
    bk = hashing.book_id("9f2c1a77b0de" + "0" * 52)
    assert bk == "bk_9f2c1a77b0de"
    iid = hashing.item_id(bk, 147, [10, 20, 300, 400])
    assert iid.startswith("hmb_") and len(iid) == 20
    assert iid == hashing.item_id(bk, 147, (10, 20, 300, 400))
    assert iid != hashing.item_id(bk, 147, [10, 20, 300, 401])


# ---------------------------------------------------------------- ontology (§7.1, §8.2)
def test_ontology_counts_and_classes():
    assert len(ontology.PREDICATES) == 46   # plan says 47; contradiction C2 recorded
    assert ontology.PREDICATES["on"].decidability.value == "geo"
    assert ontology.PREDICATES["region"].decidability.value == "model"
    assert ontology.PREDICATES["concyclic"].anchor.value == "B"
    assert ontology.PREDICATES["left_of"].strictness.value == "coarse"


def test_canonical_form_symmetry_and_sets():
    assert ontology.fact_key("parallel", ["l2", "l1"]) == ontology.fact_key("parallel", ["l1", "l2"])
    assert ontology.fact_key("collinear", [["c", "a", "b"]]) == ("collinear", (("a", "b", "c"),))
    assert ontology.fact_key("between", ["r", "q", "p"]) == ("between", ("p", "q", "r"))
    assert ontology.fact_key("intersect_at", ["y", "x", "p"]) == ("intersect_at", ("x", "y", "p"))
    assert ontology.fact_key("order_along", ["s", ["c", "b", "a"]]) == ("order_along", ("s", ("a", "b", "c")))
    # antisymmetric layout keeps direction
    assert ontology.fact_key("left_of", ["b", "a"]) != ontology.fact_key("left_of", ["a", "b"])


@pytest.mark.parametrize("pred,args", [
    ("secant_to", ["l", "c"]),                 # derived-only, not in vocabulary
    ("on", ["p"]),                             # arity
    ("collinear", [["a", "b"]]),               # set too small
    ("concyclic", [["a", "b", "c"]]),          # set too small (>= 4)
    ("axis", ["x", "diagonal"]),               # enum
    ("polygon", ["g", 2]),                     # n >= 3
    ("label", ["  ", "p"]),                    # empty text
    ("on", ["unknown", "c"]),                  # unknown only allowed for tangent point
])
def test_invalid_facts_rejected(pred, args):
    with pytest.raises(ontology.OntologyError):
        ontology.validate_fact(pred, args)


def test_tangent_unknown_point_allowed():
    ontology.validate_fact("tangent_to", ["l", "c", "unknown"])
    assert ontology.entity_args("tangent_to", ["l", "c", "unknown"]) == ["l", "c"]


# ---------------------------------------------------------------- schemas
def _rights(**over):
    rec = {
        "book_id": "bk_" + SHA[:12], "work_id": "wk_0031", "display_title": "T", "author": "A",
        "publisher": "P", "publisher_id": "pub_1", "edition_statement": "1st",
        "is_reprint_of_earlier_work": "no", "publication_year": 1905, "published": "yes", "publication_place": "X",
        "first_publication_country": "US", "language": "en",
        "acquisition": {"source_repository": "IA", "source_url": "u", "download_utc": "t",
                        "source_file_sha256": SHA, "source_file_bytes": 10, "page_count": 3,
                        "source_codec": "pdf_mixed", "source_bitdepth": 8, "source_colorspace": "gray"},
        "publication_year_evidence": {"catalog_record": "c", "catalog_record_sha256": SHA,
                                      "titlepage_asset": "t", "titlepage_asset_sha256": SHA,
                                      "titlepage_page_index": 0, "agree": True},
        "terms": {"terms_url": "u", "terms_snapshot": "s", "terms_snapshot_sha256": SHA,
                  "terms_snapshot_utc": "t", "repository_rights_statement_verbatim": "v",
                  "redistribution_permitted": "yes", "commercial_reuse_permitted": "unknown",
                  "ai_use_clause": "none"},
        "determination": {"us_status": "public_domain", "us_rule_id": "us-pd-pre1931-v1",
                          "us_determined_by": "rule", "source_country": "US",
                          "source_country_status": "not_determined",
                          "release_country_status": "not_determined", "determined_by_model": False},
        "prominence": {"hosted_by_major_aggregators": 2, "public_fulltext_layer": True},
        "release_mode": "full_public", "approved_by": "owner", "approved_utc": "t",
        "manifest_version": "rights-1", "manifest_sha256": SHA,
    }
    for k, v in over.items():
        cur = rec
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = v
    return rec


def test_rights_manifest_valid():
    RightsManifest.model_validate(_rights())


@pytest.mark.parametrize("override", [
    {"determination.determined_by_model": True},             # R1
    {"determination.source_country_status": "public_domain"},  # R2
    {"publication_year": 1938},                               # rule cannot give PD after 1930
    {"published": "unknown"},                                 # rule requires a published work
    {"terms.redistribution_permitted": "no"},                 # cannot be full_public
    {"book_id": "bk_000000000000"},                           # id not content-derived
])
def test_rights_manifest_invalid(override):
    with pytest.raises(ValidationError):
        RightsManifest.model_validate(_rights(**override))


def test_builder_output_rejects_oov():
    good = {"asc_version": "asc-1.0.0", "builder_id": "b1", "builder_family": "fam_x",
            "entities": [{"local_id": "e1", "type": "point", "label": "A",
                          "anchor": {"x": .3, "y": .6, "w": .02, "h": .02}}],
            "facts": [{"predicate": "point", "args": ["e1"]}]}
    BuilderOutput.model_validate(good)
    bad = copy.deepcopy(good)
    bad["facts"].append({"predicate": "radius_of", "args": ["s", "c"]})
    with pytest.raises(ValidationError):
        BuilderOutput.model_validate(bad)
    bad2 = copy.deepcopy(good)
    bad2["entities"][0]["type"] = "blob"
    with pytest.raises(ValidationError):
        BuilderOutput.model_validate(bad2)


def test_fact_record_alias_and_validation():
    f = FactRecord.model_validate({"fact_id": "r_004", "class": "relation", "predicate": "on",
                                   "args": ["p_A", "c_1"], "strictness": "strict",
                                   "decidability": "geo", "anchor": "A", "eligible": True})
    assert f.fact_class == "relation"
    with pytest.raises(ValidationError):
        FactRecord.model_validate({"fact_id": "x", "class": "relation", "predicate": "on",
                                   "args": ["p"], "strictness": "strict", "decidability": "geo",
                                   "anchor": "A", "eligible": True})


def test_submission_ok_requires_artifacts():
    base = {"item_id": "hmb_" + "0" * 16, "dataset_version": "v1.0", "model_id": "m",
            "model_version_string": "m", "model_family": "f", "access_date": "d",
            "provider_terms_version": "p", "output_release": "public", "track": "A",
            "prompt_id": "reconstruction-A-v1", "prompt_sha256": SHA, "decoding_params": {},
            "image_encoding": {}, "attempts": [{"index": 1, "utc": "t"}],
            "harness_version": "h", "raw_output": "r", "raw_output_sha256": SHA,
            "extraction_path": "fenced", "outcome_code": "OK"}
    with pytest.raises(ValidationError):
        SubmissionRecord.model_validate(base)
    base["outcome_code"] = "C_ERROR"
    SubmissionRecord.model_validate(base)
    base["outcome_code"] = "COMPILED"      # not in the closed vocabulary
    with pytest.raises(ValidationError):
        SubmissionRecord.model_validate(base)


def test_vocab_closed_sets():
    assert len(list(Outcome)) == 16
    assert len(list(Exclusion)) == 31   # 28 plan codes + physical_size_unknown (C11) + book_share_cap (C14)
    #                                     + precard_duplicate, selection_budget (card-select, owner 2026-09-28)
    #                                     - degradation_above_ceiling (s* deleted with CP3, 2026-09-26)


# ---------------------------------------------------------------- store
def test_immutable_write(tmp_path):
    p = tmp_path / "a" / "x.bin"
    d1 = write_immutable(p, b"abc")
    assert write_immutable(p, b"abc") == d1
    with pytest.raises(ImmutableConflict):
        write_immutable(p, b"abd")
    assert p.read_bytes() == b"abc"


def test_append_only_jsonl(tmp_path):
    p = tmp_path / "l.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"a": 2})
    assert [r["a"] for r in iter_jsonl(p)] == [1, 2]


# ---------------------------------------------------------------- thresholds / freeze (§17.3)
def test_fixture_profile_reads_everything():
    th = fixture_thresholds()
    assert th.value("g3_ink_min") == 0.005
    assert th.value("align_theta_match") == 0.9          # form constant since 2026-09-25
    assert th.value("delta_star") == 0.05                # fixture value of a calibrated parameter


def test_production_refuses_uncalibrated():
    th = production_template()
    assert th.value("align_theta_match") == 0.9                      # form constant OK
    with pytest.raises(CalibrationMissing):
        th.value("N_star")                                 # calibrated, no value
    with pytest.raises(CalibrationMissing):
        th.value("g3_ink_min")                             # starting value, not level-frozen
    assert "N_star" in th.missing_for_production()


def test_level_freeze_requires_provenance_and_blocks_change_after_test(tmp_path):
    lay = Layout(tmp_path)
    tmpl = production_template()
    with pytest.raises(ThresholdsError):
        level_freeze(tmpl, {}, lay)
    prov = {name: {"value": (s.plan_value if s.plan_value is not None else 1.0),
                   "calibration_artifact_id": "cal_1", "sha256": SHA, "date": "2026-09-22",
                   "rule": "test"} for name, s in SPECS.items() if s.kind != "form_constant"}
    th = level_freeze(tmpl, prov, lay)
    assert th.value("N_star") == 1.0
    assert load_thresholds(lay.thresholds).value("g3_ink_min") == 0.005
    lay.mark_first_test_artifact("test crop", "2026-09-22T00:00:00Z")
    prov2 = copy.deepcopy(prov)
    prov2["g3_ink_min"]["value"] = 0.004
    with pytest.raises(FreezeViolation):
        level_freeze(tmpl, prov2, lay)


def test_level_freeze_confirms_provisional_values_by_default(tmp_path):
    """Only calibrated parameters need a provenance record; a provisional one without a record is
    confirmed at its plan value, and a record overrides it (simplification 2026-09-25)."""
    from hmb.config import PLAN_VALUE_CONFIRMED, registry_hash
    lay = Layout(tmp_path)
    prov = {name: {"value": 1.0, "calibration_artifact_id": "cal_1", "sha256": SHA, "date": "2026-09-22",
                   "rule": "test"} for name, s in SPECS.items() if s.kind == "calibrated"}
    prov["g3_ink_min"] = {"value": 0.004, "calibration_artifact_id": "eng_set", "sha256": SHA,
                          "date": "2026-09-22", "rule": "engineering set"}
    th = level_freeze(production_template(), prov, lay)
    assert th.value("g3_ink_min") == 0.004                           # override kept
    assert th.value("g3_ink_max") == 0.45                            # plan value confirmed
    pv = th.params["g3_ink_max"]["provenance"]
    assert pv["rule"] == PLAN_VALUE_CONFIRMED and pv["sha256"] == registry_hash()
    assert th.missing_for_production() == []
    with pytest.raises(ThresholdsError):                              # a calibrated value is still required
        level_freeze(production_template(), {k: v for k, v in prov.items() if k != "N_star"}, Layout(tmp_path / "b"))


def test_production_file_rejects_fixture_status(tmp_path):
    th = fixture_thresholds()
    d = th.to_json()
    d["profile"] = "production"
    p = tmp_path / "t.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ThresholdsError):
        load_thresholds(p)


# ---------------------------------------------------------------- B0 Gemini round-1 regressions
def _prov():
    return {name: {"value": (s.plan_value if s.plan_value is not None else 1.0),
                   "calibration_artifact_id": "c", "sha256": SHA, "date": "d", "rule": "r"}
            for name, s in SPECS.items() if s.kind != "form_constant"}


def test_freeze_refused_when_test_artifacts_predate_it(tmp_path):
    lay = Layout(tmp_path)
    lay.mark_first_test_artifact("crop", "2026-09-22T00:00:00Z")
    with pytest.raises(FreezeViolation):
        level_freeze(production_template(), _prov(), lay)


def test_immutable_concurrent_writers(tmp_path):
    import threading
    p = tmp_path / "x.bin"
    errs, oks = [], []

    def w(b):
        try:
            write_immutable(p, b)
            oks.append(b)
        except ImmutableConflict:
            errs.append(b)
    ts = [threading.Thread(target=w, args=(bytes([i]) * 1000,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(oks) == 1 and len(errs) == 7 and p.read_bytes() == oks[0]


def test_concurrent_appends_are_whole_rows(tmp_path):
    import threading
    p = tmp_path / "l.jsonl"

    def w(i):
        for _ in range(20):
            append_jsonl(p, {"i": i, "pad": "x" * 5000})
    ts = [threading.Thread(target=w, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(list(iter_jsonl(p))) == 120


def test_builder_fact_must_reference_declared_entity():
    bad = {"asc_version": "asc-1.0.0", "builder_id": "b1", "builder_family": "f",
           "entities": [{"local_id": "e1", "type": "point", "anchor": {"x": 0, "y": 0, "w": 0, "h": 0}}],
           "facts": [{"predicate": "on", "args": ["e1", "e9"]}]}
    with pytest.raises(ValidationError):
        BuilderOutput.model_validate(bad)


def test_fact_record_attributes_must_match_ontology():
    with pytest.raises(ValidationError):
        FactRecord.model_validate({"fact_id": "r", "class": "layout", "predicate": "on",
                                   "args": ["p", "c"], "strictness": "coarse", "decidability": "geo",
                                   "anchor": "A", "eligible": True})


def test_chain_hash_accepts_structures():
    assert hashing.H([("a", "b")], "x") == hashing.H(hashing.canonical_json([["a", "b"]]), "x")
