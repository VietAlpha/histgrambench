"""Scaffold 4: form-freeze, release-build (incl. the MS1 approval check), funnel-report CLIs, acq-g8, items-finalize
on tiny fixture roots (never the real corpus; providers are explicit FAKES)."""
import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from hmb import manifests as M
from hmb import rights
from hmb.cli import main
from hmb.config import REGISTRY, fixture_thresholds, level_freeze, production_template
from hmb.schemas import Card, ItemRecord, ModelSpec
from hmb.store import Layout, append_jsonl, iter_jsonl, write_json_immutable
from w1_helpers import blank, draw_diagram, source_crop


# --------------------------------------------------------------------------- helpers
def cli(root, *args):
    return main(["--root", str(root), *args])


def manifest(i: int, approver="owner:alice"):
    from test_w1_rights import valid_record
    rec = valid_record(approver=approver)
    sha = hashlib.sha256(f"book-{i}".encode()).hexdigest()
    rec["book_id"] = "bk_" + sha[:12]
    rec["acquisition"]["source_file_sha256"] = sha
    rec["work_id"] = f"wk_{i}"
    rec["display_title"] = f"Title {i}"
    rec["author"] = f"Author {i}"
    rec["manifest_sha256"] = rights.manifest_sha256(rec)
    return rights.apply_us_rule(rec)


def write_manifests(lay, recs):
    lay.rights_manifest.parent.mkdir(parents=True, exist_ok=True)
    lay.rights_manifest.write_text(rights.dump_yaml(recs).decode(), encoding="utf-8")


def frozen_thresholds(lay, now="2026-01-01T00:00:00+00:00"):
    fx = fixture_thresholds()
    prov = {}
    for s in REGISTRY:
        if s.kind == "form_constant":
            continue
        v = fx.params[s.name]["value"]
        prov[s.name] = {"value": 1 if v is None else (v if v != {} else {"x": 1}),
                        "calibration_artifact_id": "test", "sha256": "0" * 64, "date": "2026-01-01", "rule": "test"}
    return level_freeze(production_template(), prov, lay, now=now)


def fact(fid, pred, args, cls="entity", eligible=True):
    return {"fact_id": fid, "class": cls, "predicate": pred, "args": args, "strictness": "strict",
            "decidability": "geo", "anchor": "A", "eligible": eligible}


def make_card(iid, n_eligible, profile="fixture"):
    facts = [fact(f"f{k}", "point", [f"p{k}"]) for k in range(n_eligible)] + [fact("fz", "point", ["pz"], eligible=False)]
    return Card.model_validate({"item_id": iid, "asc_version": "asc-1.0.0+test", "card_uid": "card_" + "0" * 15 + str(n_eligible % 10),
                                "entities": [{"entity_id": f"p{k}", "type": "point"} for k in range(n_eligible)]
                                + [{"entity_id": "pz", "type": "point"}],
                                "facts": facts, "builder_order": [],
                                "family_map": {}, "profile": profile})


def add_candidate(lay, th, book, page, box, *, cluster="singleton", frag=1.0, n_facts=3, gate_reasons=(),
                  sc=0.8, profile="fixture"):
    from hmb.acquisition.funnel import append_row, assemble_item, proposal_id
    pid = proposal_id(book, page, box, "paddle_layout@1.6", None)
    img = draw_diagram(blank(400, 400), 200, 200, 120)
    pp = lay.page_image(book, page)
    pp.parent.mkdir(parents=True, exist_ok=True)
    data = cv2.imencode(".png", img)[1].tobytes()
    pp.write_bytes(data)
    row = {"proposal_id": pid, "book_id": book, "page_index": page, "box_xyxy_page_px": box,
           "box_source": "paddle_layout@1.6", "box_frame": "paddle_preprocessed", "source_crop": source_crop(box),
           "gates": [], "exclusion_reasons": [],
           "img_stats": {"frag_index": frag}, "phash_cluster_id": cluster, "pipeline_version": "extract-1.0.0",
           "config_hash": "c"}
    crop = cv2.imencode(".png", img[box[1]:box[3], box[0]:box[2]])[1].tobytes()
    rec = assemble_item(lay, row, page_png_path=pp, page_png_sha256=hashlib.sha256(data).hexdigest(), crop_png=crop,
                        context={"context_text": "", "context_bbox": box, "context_char_count": 0,
                                 "context_block_count": 0},
                        page_meta={"render_dpi": 400, "renderer_version": "t"}, ocr_engine_version="none",
                        ocr_status="unavailable", work_id="wk_1", th=th, dataset_version="v1.0",
                        rights_manifest_sha256=None)
    iid = rec["id"]
    write_json_immutable(lay.p("data", "candidates", f"{iid}.json"), rec)
    append_row(lay, dict(row, item_id=iid))
    card = make_card(iid, n_facts, profile)
    write_json_immutable(lay.card(iid), card.model_dump(mode="json", by_alias=True))
    write_json_immutable(lay.card_manifest(card.card_uid, iid), {"item_id": iid, "card_uid": card.card_uid})
    write_json_immutable(lay.p("data", "card_gates", f"{iid}.json"),
                         {"item_id": iid, "card_uid": card.card_uid, "exclusion_reasons": list(gate_reasons),
                          "structure_coverage_proxy": sc, "profile": profile})
    return iid


# --------------------------------------------------------------------------- (1) form-freeze
def test_form_freeze_never_claims_frozen_and_records_owner_approval(tmp_path, capsys):
    lay = Layout(tmp_path)
    assert cli(tmp_path, "form-freeze") == 0
    doc = json.loads(lay.p("manifests", "form_freeze.json").read_text(encoding="utf-8"))
    assert doc["status"] == "awaiting_owner_hash_and_commit" and "frozen" not in doc["status"].replace("freeze", "")
    for k in ("parameter_registry", "ontology_table", "rules_R_E", "normalization_and_confusables", "builder_prompts",
              "contestant_prompts_and_wrapper", "judge_prompts_and_verdict_schema", "outcome_vocabulary",
              "metric_code", "estimator_code", "gate_forms"):
        assert doc["components"][k]["sha256"], k
    from hmb.config import registry_hash
    assert doc["components"]["parameter_registry"]["detail"]["registry_hash"] == registry_hash()
    names = {a["name"] for a in doc["awaiting_level_freeze"]}
    assert {"N_star", "render_dpi", "g8_max_tokens"} <= names and "s_star" not in names and "alpha" not in names    # FC excluded
    assert cli(tmp_path, "form-freeze", "--record-owner-approval") == 2                   # actor required
    assert cli(tmp_path, "form-freeze", "--record-owner-approval", "--actor", "owner:harry") == 0
    rows = list(iter_jsonl(lay.human_audit_log))
    assert rows[-1]["scope"] == "protocol" and rows[-1]["actor"] == "owner:harry"
    assert doc["form_freeze_sha256"] in rows[-1]["description"]
    assert json.loads(lay.p("manifests", "form_freeze.json").read_text(encoding="utf-8"))["status"] == doc["status"]


def test_form_freeze_refuses_changed_forms_after_approval(tmp_path):
    lay = Layout(tmp_path)
    append_jsonl(lay.human_audit_log, {"utc": "t", "actor": "owner:x", "scope": "protocol",
                                       "action": "cp1_form_freeze_approval",
                                       "description": "form_freeze_sha256=" + "f" * 64})
    with pytest.raises(M.ManifestError):
        M.write_form_freeze(lay, "t")


# --------------------------------------------------------------------------- (2) release-build
def test_release_build_refusals(tmp_path, capsys):
    lay = Layout(tmp_path)
    assert cli(tmp_path, "release-build", "--version", "1.0.0") == 2                  # no thresholds at all
    assert cli(tmp_path, "thresholds-init", "--profile", "production") == 0
    assert cli(tmp_path, "release-build", "--version", "1.0.0") == 2
    assert "not level-frozen" in capsys.readouterr().out
    lay2 = Layout(tmp_path / "late")
    frozen_thresholds(lay2, now="2026-01-02T00:00:00+00:00")
    write_json_immutable(lay2.first_test_artifact_marker, {"description": "x", "utc": "2026-01-01T00:00:00+00:00"})
    with pytest.raises(M.ManifestError, match="predate"):
        M.build_release_from_root(lay2, "1.0.0", created="c")


def _split_groups(lay, th, books):
    from hmb import sampling as SM
    g = SM.apply_collapse_guard(books, [], {b["book_id"]: 1 for b in books}, th)
    SM.write_split_groups(lay, g, SM.assign_splits(g, 1, th))


def test_release_build_fixture_is_labelled_non_releasable(tmp_path):
    lay = Layout(tmp_path)
    th = fixture_thresholds()
    assert cli(tmp_path, "thresholds-init", "--profile", "fixture") == 0
    with pytest.raises(M.ManifestError, match="split_groups"):
        M.build_release_from_root(lay, "0.0.1", fixture=True, created="c")
    mans = [manifest(i) for i in range(5)]
    _split_groups(lay, th, mans)
    iid = add_candidate(lay, th, mans[0]["book_id"], 0, [100, 80, 330, 330], n_facts=2)
    rec = json.loads(lay.p("data", "candidates", f"{iid}.json").read_text(encoding="utf-8"))
    rec["quality"]["eligible"] = True
    write_json_immutable(lay.item_record("development", iid), rec)
    assert cli(tmp_path, "release-build", "--version", "0.0.1", "--fixture") == 0
    rel = json.loads(lay.release_manifest.read_text(encoding="utf-8"))
    assert rel["releasable"] is False and rel["profile"] == "fixture" and rel["placeholders"]
    assert rel["counts"] == {"books": 1, "split_groups": 5, "items_eligible": 1, "facts_eligible": 2}
    assert M.verify_release(rel)
    assert rel["split_hash"] == M.file_hash(lay.split_groups)
    # a different card changes the cardset (and so the release) hash
    lay.card(iid).unlink()
    write_json_immutable(lay.card(iid), make_card(iid, 3).model_dump(mode="json", by_alias=True))
    rel2 = M.build_release_from_root(lay, "0.0.2", fixture=True, created="c")
    assert rel2["cardset_hash"] != rel["cardset_hash"] and rel2["counts"]["facts_eligible"] == 3


# --------------------------------------------------------------------------- (3) funnel-report / split-assign
def test_funnel_report_cli(tmp_path, capsys):
    lay = Layout(tmp_path)
    mans = [manifest(i) for i in range(5)]
    write_manifests(lay, mans)
    for i, m in enumerate(mans):
        for k in range(2):
            append_jsonl(lay.funnel, {"proposal_id": f"p{i}{k}", "book_id": m["book_id"], "page_index": k,
                                      "gates": [{"gate_id": "G2", "version": "g2-1", "passed": k == 0,
                                                 "measured_value": None, "threshold": None}],
                                      "exclusion_reasons": [] if k == 0 else ["box_too_small"],
                                      "img_stats": {"ink_frac": 0.1 * (k + 1)}, "phash_cluster_id": "singleton",
                                      "outcome": None if k == 0 else "excluded", "pipeline_version": "p",
                                      "config_hash": "c"})
    assert cli(tmp_path, "funnel-report") == 0
    rep = json.loads((lay.reports / "funnel_report.json").read_text(encoding="utf-8"))
    assert rep["overall"]["S2"]["n"] == 10 and rep["overall"]["S3"]["n"] == 10    # S3 == S2 (single source)
    assert rep["per_gate"]["G2"]["conditional_pass_rate"] == 0.5
    assert rep["exclusion_reasons"] == {"box_too_small": 5}
    assert rep["by_stratum"]["language"]["en"]["S2"]["n"] == 10


# --------------------------------------------------------------------------- (4) acq-g8
class FakeG8:
    """Explicit fake gatekeeper: the only G8 call is G8b (role audit 2026-09-23)."""

    def __init__(self):
        self.calls = 0
        self.purposes = []

    def send(self, req, images):
        self.calls += 1
        self.purposes.append(req.purpose)
        text = {"G8b": '{"target_type": true, "readable": true, "tikz_representable": true}'}[req.purpose]
        return {"text": text, "input_tokens": 10, "output_tokens": 5, "model_version_string": "fake"}


def _roster():
    mk = lambda mid, fam, role: ModelSpec(model_id=mid, provider="fake", family=fam, role=role, unit_price_id="p",  # noqa
                                          effort="high")
    return {"verifier": [mk("v1", "B", "verifier")]}


def _ladder_root(tmp_path, th):
    from test_w1_collection_ladder import _manifest_for, make_corpus
    from hmb.acquisition import collection as col
    from hmb.acquisition.ladder import process_source
    make_corpus(tmp_path / "corpus", name="E.pdf")
    snap = col.snapshot_source(tmp_path / "corpus", "E.pdf")
    lay = Layout(tmp_path / "root")
    col.store_snapshot(lay, snap)
    man = _manifest_for(snap)
    write_manifests(lay, [man])
    process_source(lay, snap, man, th, profile="fixture")
    return lay


def test_acq_g8_candidates_resumable_and_cached(tmp_path, th, ledger):
    from hmb.acquisition.drivers import run_g8
    lay = _ladder_root(tmp_path, th)
    stop = run_g8(lay, th, _roster(), ledger=None, allow_live=False, provider_override=FakeG8())
    assert stop["stopped"] and stop["pending"] == 1 and stop["processed"] == 0          # no live -> no guess
    fp = FakeG8()
    out = run_g8(lay, th, _roster(), ledger=ledger, allow_live=True, provider_override=fp)
    assert out["survivors"] == 1 and fp.calls == 1 and fp.purposes == ["G8b"] and "g8c" not in out
    iid = out["candidates"][0]
    rec = ItemRecord.model_validate(json.loads(lay.p("data", "candidates", f"{iid}.json").read_text(encoding="utf-8")))
    am = rec.automatic_metadata
    assert "diagram_category" not in am and "g8d_metadata" not in am and am["ocr_words_level"] == "block"
    assert isinstance(am["ocr_words"], list)
    man = rights.load_manifests(lay.rights_manifest)[rec.source["book_id"]]
    assert rec.source["rights_manifest_sha256"] == man["manifest_sha256"]
    latest = {r["proposal_id"]: r for r in iter_jsonl(lay.funnel)}
    g8row = next(r for r in latest.values() if r.get("item_id") == iid)
    assert [g["gate_id"] for g in g8row["gates"]][-1] == "G8b" and g8row["verifier_pred"] == "accept"
    assert not any(g["gate_id"] in ("G8a", "G8c", "G8d") for g in g8row["gates"])
    assert "category_pred" not in g8row and g8row["model_calls"][-1]["effort"] == "high"
    fp2 = FakeG8()
    again = run_g8(lay, th, _roster(), ledger=ledger, allow_live=True, provider_override=fp2)
    assert again["skipped_done"] >= 1 and again["processed"] == 0 and fp2.calls == 0
    with pytest.raises(Exception):
        run_g8(lay, production_template(), _roster(), ledger=ledger, allow_live=True, provider_override=FakeG8())


def test_acq_g8_meters_under_the_prefreeze_context(tmp_path, th, ledger, monkeypatch):
    """A pre-freeze G8b call is metered under its context (PILOT), not the test phase (2026-09-27 pilot finding)."""
    import dataclasses

    from hmb.acquisition import gates as G
    from hmb.acquisition.drivers import G8_PHASE, declare_development_book, run_g8
    lay = _ladder_root(tmp_path, th)
    for r in {r["proposal_id"]: r for r in iter_jsonl(lay.funnel)}.values():
        declare_development_book(lay, r["book_id"], "E.pdf", "PILOT")
    seen, real = [], G.g8b
    monkeypatch.setattr(G, "g8b", lambda *a, **k: (seen.append(k["phase"]), real(*a, **k))[1])
    run_g8(lay, dataclasses.replace(th, context="PILOT"), _roster(), ledger=ledger, allow_live=True,
           provider_override=FakeG8())
    assert seen == ["PILOT"] and G8_PHASE == "TEST"


def test_acq_g8_never_issues_a_tombstoned_id(tmp_path, th, ledger):
    """§15.4 tombstone check at item issue: a retired item id is never issued again (no call is made)."""
    from hmb.acquisition.drivers import run_g8
    from hmb.hashing import item_id
    lay = _ladder_root(tmp_path, th)
    row = next(iter({r["proposal_id"]: r for r in iter_jsonl(lay.funnel)}.values()))
    iid = item_id(row["book_id"], row["page_index"], [int(v) for v in row["box_xyxy_page_px"]])
    M.add_tombstone(lay, iid, "0.9.0", "rights_retracted", "evidence/x")
    fp = FakeG8()
    with pytest.raises(Exception, match="tombstoned"):
        run_g8(lay, th, _roster(), ledger=ledger, allow_live=True, provider_override=fp)
    assert fp.calls == 0 and not lay.p("data", "candidates", f"{iid}.json").exists()


def test_acq_g8_needs_exactly_one_gatekeeper(tmp_path, th, ledger):
    from hmb.acquisition.drivers import run_g8
    from hmb.acquisition.gates import GateBlocked
    lay = _ladder_root(tmp_path, th)
    for bad in ({"verifier": []}, {"verifier": _roster()["verifier"] * 2}):
        with pytest.raises(GateBlocked, match="exactly one gatekeeper"):
            run_g8(lay, th, bad, ledger=ledger, allow_live=True, provider_override=FakeG8())


def test_ocr_words_in_crop_block_level():
    from hmb.acquisition.drivers import ocr_words_in_crop
    blocks = [{"bbox": [10, 10, 50, 30], "text": "A"}, {"bbox": [0, 90, 400, 200], "text": "paragraph"}]
    assert ocr_words_in_crop(blocks, [0, 0, 100, 100]) == [{"text": "A", "bbox": [10, 10, 50, 30], "conf": None,
                                                           "level": "block"}]
    assert ocr_words_in_crop([], [0, 0, 100, 100]) == []


# --------------------------------------------------------------------------- (5) items-finalize
def _finalize_root(tmp_path, th, profile="fixture", approver="fixture:t"):
    lay = Layout(tmp_path)
    mans = [manifest(i, approver=approver) for i in range(5)]
    write_manifests(lay, mans)
    b = [m["book_id"] for m in mans]
    ids = {
        "i1": add_candidate(lay, th, b[0], 0, [100, 80, 330, 330], cluster="dup_x", frag=1.0, n_facts=4, profile=profile),
        "i5": add_candidate(lay, th, b[0], 1, [100, 80, 330, 330], cluster="dup_x", frag=2.0, n_facts=4, profile=profile),
        "i2": add_candidate(lay, th, b[1], 0, [100, 80, 330, 330], gate_reasons=["card_low_support"], profile=profile),
        "i3": add_candidate(lay, th, b[2], 0, [100, 80, 330, 330], n_facts=5, profile=profile),
        "i7": add_candidate(lay, th, b[2], 1, [100, 80, 330, 330], n_facts=2, profile=profile),
        "i4": add_candidate(lay, th, b[3], 0, [100, 80, 330, 330], profile=profile),
        "i6": add_candidate(lay, th, b[4], 0, [100, 80, 330, 330], profile=profile),
    }
    return lay, ids


def test_items_finalize_fixture_refuses_test_records(tmp_path, th):
    from hmb.acquisition.drivers import finalize_items
    lay, ids = _finalize_root(tmp_path, th)
    rep = finalize_items(lay, th, release_version="0.1.0", seed=1)
    assert rep["refused"] and "test-split" in rep["refused"] and not lay.p("data", "development").exists()
    rep = finalize_items(lay, th, release_version="0.1.0", seed=1, development_only=True)
    assert rep["refused"] is None and set(rep["records"].values()) == {"development"}
    recs = {i: ItemRecord.model_validate(json.loads(lay.item_record("development", ids[i]).read_text(encoding="utf-8")))
            for i in ids if lay.item_record("development", ids[i]).exists()}
    for name, r in recs.items():
        if name == "i5":
            assert r.quality.exclusion_reasons == ["duplicate_representative_lost"] and not r.quality.eligible
            assert r.quality.funnel_exit_stage == "S5" and not r.quality.is_cluster_representative
        if name == "i1":
            assert r.quality.eligible and r.quality.is_cluster_representative and r.semantic_card
        if name == "i2":
            assert r.quality.exclusion_reasons == ["card_low_support"] and not r.quality.eligible
        if name == "i7":
            assert r.quality.exclusion_reasons == ["book_share_cap"]      # 2 of 5 eligible from one book
    assert rep["book_cap"]["moved"] == [ids["i7"]]
    outcomes = {r["item_id"]: r.get("outcome") for r in
                {x["proposal_id"]: x for x in iter_jsonl(lay.funnel)}.values()}
    expect = {"i5": "excluded", "i2": "unscored_pool", "i7": "unscored_pool", "i1": "eligible",
              "i3": "eligible", "i4": "eligible", "i6": "eligible"}
    for name, iid in ids.items():                      # final outcomes appended for every written record
        if iid in rep["records"]:
            assert outcomes[iid] == expect[name], name
    assert not lay.test_artifacts_exist()


def test_items_finalize_records_the_approved_manifest_sha256(tmp_path, th):
    """Item records carry the manifest_sha256 of the rights record R3 approved, not a schema tag; a
    changed approved record changes the recorded hash."""
    from hmb.acquisition.drivers import finalize_items
    got = []
    for k, attribution in enumerate(("Scan: Library A", "Scan: Library A (revised)")):
        lay = Layout(tmp_path / f"r{k}")
        mans = []
        for i in range(5):
            m = manifest(i, approver="fixture:t")
            m["required_attribution_string"] = attribution
            m["manifest_sha256"] = rights.manifest_sha256(m)
            mans.append(m)
        write_manifests(lay, mans)
        iids = [add_candidate(lay, th, m["book_id"], 0, [100, 80, 330, 330]) for m in mans]
        rep = finalize_items(lay, th, release_version="0.1.0", seed=1, development_only=True)
        written = [i for i in iids if lay.item_record("development", i).exists()]
        assert written and rep["refused"] is None
        for iid in written:
            rec = json.loads(lay.item_record("development", iid).read_text(encoding="utf-8"))
            book = rec["source"]["book_id"]
            want = next(m["manifest_sha256"] for m in mans if m["book_id"] == book)
            assert rec["source"]["rights_manifest_sha256"] == want and rec["source"]["rights_manifest_version"] == "rights-1"
            got.append((k, book, rec["source"]["rights_manifest_sha256"]))
    common = {b for k, b, _ in got if k == 0} & {b for k, b, _ in got if k == 1}
    assert common
    for b in common:
        assert len({h for _, bb, h in got if bb == b}) == 2      # changed record -> changed hash


def test_items_finalize_production_writes_test_and_marks_first_artifact(tmp_path):
    from hmb.acquisition.drivers import finalize_items
    lay = Layout(tmp_path)
    th = frozen_thresholds(lay)
    lay2, ids = _finalize_root(tmp_path, th, profile="production", approver="owner:alice")
    rep = finalize_items(lay, th, release_version="1.0.0", seed=1)
    assert rep["refused"] is None and "test" in set(rep["records"].values())
    assert lay.test_artifacts_exist()
    # re-run is idempotent (immutable records, no duplicate outcome rows)
    n_rows = len(list(iter_jsonl(lay.funnel)))
    finalize_items(lay, th, release_version="1.0.0", seed=1)
    assert len(list(iter_jsonl(lay.funnel))) == n_rows


def test_items_finalize_keeps_prefreeze_development_books_out_of_test(tmp_path):
    """Partition separation: a book processed in a pre-freeze PILOT/DEV context is a development book, so its
    split group is always development (never drawn into test); with every group forced to development no
    test split exists and items-finalize refuses (nothing written)."""
    from hmb.acquisition.drivers import declare_development_book, finalize_items
    lay = Layout(tmp_path)
    th = frozen_thresholds(lay)
    _finalize_root(tmp_path, th, profile="production", approver="owner:alice")
    cands = {json.loads(p.read_text(encoding="utf-8"))["source"]["book_id"]
             for p in lay.p("data", "candidates").glob("*.json")}
    for b in sorted(cands):
        declare_development_book(lay, b, f"{b}.pdf", "DEV")
    rep = finalize_items(lay, th, release_version="1.0.0", seed=1)
    assert rep["refused"] and "cannot give" in rep["refused"] and "test split" in rep["refused"]
    assert not lay.test_artifacts_exist() and not lay.p("data", "test").exists()


def test_assign_splits_forces_development_book_groups(th):
    """Groups holding pre-freeze development books are development for every seed; the seeded draw fills the
    development split up to dev_groups (3) from the rest."""
    from hmb.sampling import assign_splits
    graph = {"groups": {f"g{k}": [f"bk{k}"] for k in range(8)}, "collapse_guard": {"held_out_claim": True}}
    for seed in range(20):
        s = assign_splits(graph, seed, th, {"bk5"})
        dev = {g for g, sp in s["assignment"].items() if sp == "development"}
        assert "g5" in dev and len(dev) == 3 and s["forced_development_groups"] == ["g5"] and s["n_test_groups"] == 5
    many = assign_splits(graph, 0, th, {f"bk{k}" for k in range(5)})     # more development groups than dev_groups
    assert sum(sp == "development" for sp in many["assignment"].values()) == 5 and many["n_test_groups"] == 3


def test_items_finalize_applies_the_standing_collapse_rule(tmp_path, th):
    from hmb.acquisition.drivers import finalize_items
    lay = Layout(tmp_path)
    mans = [manifest(i, approver="fixture:t") for i in range(5)]
    write_manifests(lay, mans)
    for k in range(3):            # books 0-2 share a pHash cluster -> one split group holding 3/5 = 60%
        add_candidate(lay, th, mans[k]["book_id"], 0, [100, 80, 330, 330], cluster="dup_c", n_facts=2 + k)
    for k in (3, 4):
        add_candidate(lay, th, mans[k]["book_id"], 0, [100, 80, 330, 330])
    rep = finalize_items(lay, th, release_version="0.1.0", seed=1, development_only=True)
    # no single drop of books 0-2 splits their shared cluster into two groups -> single_group, no manual stop
    assert rep["refused"] is None and rep["split"]["collapse_guard_action"] == "single_group"
    assert rep["split"]["held_out_claim"] is False


def _approve_forms(lay, utc):
    M.write_form_freeze(lay, utc)
    M.record_form_freeze_approval(lay, "owner:test", utc)


def test_production_release_requires_the_cp1_approval_of_the_current_forms(tmp_path):
    lay = Layout(tmp_path)
    frozen_thresholds(lay, now="2026-01-01T00:00:00+00:00")
    write_json_immutable(lay.first_test_artifact_marker, {"description": "x", "utc": "2026-02-01T00:00:00+00:00"})
    _split_groups(lay, fixture_thresholds(), [manifest(i) for i in range(5)])
    with pytest.raises(M.ManifestError, match="no MS1 owner approval"):
        M.build_release_from_root(lay, "1.0.0", created="c")
    _approve_forms(lay, "2026-01-05T00:00:00+00:00")                # approved AFTER the level freeze
    with pytest.raises(M.ManifestError, match="later than the level freeze"):
        M.build_release_from_root(lay, "1.0.0", created="c")
    lay2 = Layout(tmp_path / "b")
    frozen_thresholds(lay2, now="2026-01-01T00:00:00+00:00")
    _approve_forms(lay2, "2025-12-31T00:00:00+00:00")
    assert M.cp1_approval_problems(lay2, "2026-01-01T00:00:00+00:00") == []
    # an approval of different forms does not count (a changed prompt or registry is a new version)
    rows = lay2.human_audit_log.read_text(encoding="utf-8").replace("form_freeze_sha256=", "form_freeze_sha256=0")
    lay2.human_audit_log.write_text(rows, encoding="utf-8")
    assert M.cp1_approval_problems(lay2)[0].startswith("no MS1 owner approval")


def test_release_judge_config_check_reads_test_items_only(tmp_path):
    """C42 fix: verdicts of development items judged before the freeze never block the release's judge-config check."""
    lay = Layout(tmp_path)
    write_json_immutable(lay.item_record("test", "hmb_t"), {"id": "hmb_t"})
    write_json_immutable(lay.verdicts("m", "A", "hmb_t"), {"item_id": "hmb_t", "judge_config_hash": "final"})
    write_json_immutable(lay.verdicts("m", "A", "hmb_d"), {"item_id": "hmb_d", "judge_config_hash": "pre-freeze"})
    assert M.test_verdict_judge_hashes(lay) == {"final"}


def test_production_release_refuses_fixture_cards_verdicts_scores_and_names_them(tmp_path):
    lay = Layout(tmp_path)
    th = frozen_thresholds(lay, now="2026-01-01T00:00:00+00:00")
    _approve_forms(lay, "2025-12-31T00:00:00+00:00")
    write_json_immutable(lay.first_test_artifact_marker, {"description": "x", "utc": "2026-02-01T00:00:00+00:00"})
    _split_groups(lay, th, [manifest(i) for i in range(5)])
    iid = add_candidate(lay, th, manifest(0)["book_id"], 0, [100, 80, 330, 330], profile="fixture")
    rec = json.loads(lay.p("data", "candidates", f"{iid}.json").read_text(encoding="utf-8"))
    rec["quality"]["eligible"] = True
    write_json_immutable(lay.item_record("test", iid), rec)
    write_json_immutable(lay.verdicts("m", "A", iid), {"profile": "fixture"})
    write_json_immutable(lay.scores("m", "A", iid), {"profile": "fixture"})
    with pytest.raises(M.ManifestError) as e:
        M.build_release_from_root(lay, "1.0.0", created="c")
    msg = str(e.value)
    assert f"card:{iid}" in msg and f"verdicts:m/A/{iid}.json" in msg and f"scores:m/A/{iid}.json" in msg
    rel = M.build_release_from_root(lay, "0.0.9", fixture=True, created="c")     # fixture: allowed, labelled
    assert rel["profile"] == "fixture" and rel["releasable"] is False
    on_disk = json.loads(lay.release_manifest.read_text(encoding="utf-8"))
    assert on_disk["profile"] == "fixture" and on_disk["releasable"] is False
