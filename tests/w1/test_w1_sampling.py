"""§13.1-13.6 split groups, collapse guard, split assignment, dedup, funnel statistics, IRLS."""
import math

import numpy as np
import pytest
import yaml

from hmb import sampling as SM
from hmb.store import Layout


def book(bid, **kw):
    return {"book_id": bid, **kw}


def test_fold_scripts_and_diacritics():
    assert SM.fold("Géométrie  Élémentaire!") == "geometrie elementaire"
    assert SM.fold("Геометрия") == "geometriia"
    assert SM.fold("Γεωμετρία") == "geometria"
    assert SM.work_key("Геометрия", "Киселёв") == SM.work_key("Geometriia", "Kiselev")


def test_split_graph_hard_and_soft_edges_on_manifest_fields(th):
    # RightsManifest field names only (display_title, author, work_id, publisher_id, publication_year)
    books = [book("bk_a", display_title="Elements", author="Euclid", work_id="wk_1", publisher_id="pub_p",
                  publication_year=1900),
             book("bk_b", display_title="ELEMENTS.", author="Euclid", work_id="wk_2", publisher_id="pub_q"),
             book("bk_c", display_title="Conics", author="Euclid", work_id="wk_3", publisher_id="pub_p",
                  publication_year=1905),
             book("bk_d", display_title="Trig", author="X", work_id="wk_9"),          # translation: same work_id
             book("bk_e", display_title="Trigonometrie", author="Y", work_id="wk_9"),
             book("bk_f", display_title="Solid", author="Z", work_id="wk_5"),
             book("bk_g", display_title="Other", author="W", work_id="wk_6", publisher_id="pub_p",
                  publication_year=1925)]
    items = [{"item_id": "i1", "book_id": "bk_f", "phash_cluster_id": "dup_abc"},
             {"item_id": "i2", "book_id": "bk_g", "phash_cluster_id": "dup_abc"},
             {"item_id": "i3", "book_id": "bk_a", "phash_cluster_id": "singleton"}]
    g = SM.split_graph(books, items, th)
    groups = sorted(sorted(v) for v in g["groups"].values())
    assert groups == [["bk_a", "bk_b"], ["bk_c"], ["bk_d", "bk_e"], ["bk_f", "bk_g"]]
    types = {(e["a"], e["b"], e["type"], e["evidence"]) for e in g["hard_edges"]}
    assert ("bk_a", "bk_b", "work_identity", "title+author") in types
    assert ("bk_d", "bk_e", "work_identity", "work_id") in types
    assert ("bk_f", "bk_g", "duplicate", "dup_abc") in types
    soft = {(e["a"], e["b"], e["type"]) for e in g["soft_edges"]}
    assert ("bk_a", "bk_c", "same_author") in soft and ("bk_a", "bk_c", "same_publisher_within_years") in soft
    assert ("bk_a", "bk_g", "same_publisher_within_years") not in soft      # 25 years apart > 10
    assert ("bk_a", "bk_b", "same_author") not in soft


def test_split_graph_uses_stored_cluster_ids_not_phash(th):
    books = [book("bk_a", display_title="A", author="a"), book("bk_b", display_title="B", author="b")]
    # identical pHashes but stored clusters are singletons -> no duplicate edge (no re-clustering here)
    items = [{"item_id": "i1", "book_id": "bk_a", "phash64": "0" * 16, "phash_cluster_id": "singleton"},
             {"item_id": "i2", "book_id": "bk_b", "phash64": "0" * 16, "phash_cluster_id": "singleton"}]
    assert SM.split_graph(books, items, th)["hard_edges"] == []
    items[0]["phash_cluster_id"] = items[1]["phash_cluster_id"] = "dup_g8c_merged"
    assert SM.split_graph(books, items, th)["hard_edges"][0]["type"] == "duplicate"


def test_collapse_guard_standing_rule(th):
    books = [book(f"bk_{c}", display_title=c, author=c) for c in "abcdef"]
    items = [{"item_id": "x1", "book_id": "bk_a", "phash_cluster_id": "dup_1"},
             {"item_id": "x2", "book_id": "bk_b", "phash_cluster_id": "dup_1"},
             {"item_id": "x3", "book_id": "bk_b", "phash_cluster_id": "dup_2"},
             {"item_id": "x4", "book_id": "bk_c", "phash_cluster_id": "dup_2"}]
    elig = {"bk_a": 30, "bk_b": 25, "bk_c": 30, "bk_d": 10, "bk_e": 10, "bk_f": 10}   # a-b-c = 74%
    # b (25, the smallest count) bridges a-b-c; dropping it leaves a and c apart, none above 50%
    g = SM.apply_collapse_guard(books, items, elig, th)
    cg = g["collapse_guard"]
    assert cg["action"] == "drop_book" and cg["dropped_books"] == ["bk_b"] and cg["held_out_claim"]
    assert "bk_b" not in g["book_group"] and not cg["after_drop"]["collapsed"]
    # the smallest-count book is tried first, but skipped when its drop does not break the component
    g2 = SM.apply_collapse_guard(books, items, dict(elig, bk_a=5, bk_b=25), th)
    assert g2["collapse_guard"]["dropped_books"] == ["bk_b"]
    # D12 floor: if only the dominant book's drop would work, it is never dropped -> single_group
    star = [{"item_id": "s1", "book_id": "bk_a", "phash_cluster_id": "dup_1"},      # a is the hub of a-b, a-c
            {"item_id": "s2", "book_id": "bk_b", "phash_cluster_id": "dup_1"},
            {"item_id": "s3", "book_id": "bk_a", "phash_cluster_id": "dup_4"},
            {"item_id": "s4", "book_id": "bk_c", "phash_cluster_id": "dup_4"}]
    big = SM.apply_collapse_guard(books, star, {"bk_a": 90, "bk_b": 1, "bk_c": 1, "bk_d": 1}, th)
    assert big["collapse_guard"]["action"] == "single_group" and "dropped_books" not in big["collapse_guard"]
    assert not big["collapse_guard"]["held_out_claim"]
    # a triangle no single drop can split -> single_group, held-out claim abandoned
    tri = items + [{"item_id": "x5", "book_id": "bk_a", "phash_cluster_id": "dup_3"},
                   {"item_id": "x6", "book_id": "bk_c", "phash_cluster_id": "dup_3"}]
    one = SM.apply_collapse_guard(books, tri, elig, th)
    assert one["collapse_guard"]["action"] == "single_group" and not one["collapse_guard"]["held_out_claim"]
    split = SM.assign_splits(one, 7, th)
    assert split["held_out_claim"] is False and set(split["assignment"].values()) == {"test"}
    ok = SM.apply_collapse_guard(books, [], elig, th)                  # no hard edges: nothing collapses
    assert ok["collapse_guard"]["action"] is None and not ok["collapse_guard"]["collapsed"]


def test_assign_splits_deterministic_with_three_dev_groups(th, tmp_path):
    books = [book(f"bk_{i:02d}", display_title=str(i), author=str(i)) for i in range(12)]
    g = SM.apply_collapse_guard(books, [], {b["book_id"]: 10 for b in books}, th)
    s1, s2 = SM.assign_splits(g, 42, th), SM.assign_splits(g, 42, th)
    assert s1 == s2
    assert list(s1["assignment"].values()).count("development") == 3 and s1["n_test_groups"] == 9
    assert s1 != SM.assign_splits(g, 43, th)
    SM.write_split_groups(Layout(tmp_path), g, s1)
    doc = yaml.safe_load(Layout(tmp_path).split_groups.read_text(encoding="utf-8"))
    assert doc["seed"] == 42 and len(doc["groups"]) == 12


def test_soft_crossings_and_residual_duplicates(th):
    graph = {"book_group": {"bk_a": "g1", "bk_b": "g2"}, "soft_edges": [{"a": "bk_a", "b": "bk_b", "type": "same_author"}]}
    split = {"assignment": {"g1": "development", "g2": "test"}}
    assert SM.soft_edge_crossings(graph, split)[0]["type"] == "same_author"
    items = [{"item_id": "d1", "book_id": "bk_a", "phash64": "0" * 16},
             {"item_id": "t1", "book_id": "bk_b", "phash64": "000000000000000f"}]
    assert SM.residual_cross_split_duplicates(items, graph, split, 6) == ["d1"]
    assert SM.residual_cross_split_duplicates(items, graph, split, 3) == []


def test_dedup_representative_rule(th):
    base = [("point", ["e1"]), ("label", ["A", "e1"]), ("circle", ["e2"]), ("on", ["e1", "e2"]),
            ("center_of", ["e3", "e2"]), ("point", ["e3"]), ("label", ["O", "e3"]), ("segment", ["e4"]),
            ("endpoint_of", ["e1", "e4"]), ("endpoint_of", ["e3", "e4"])]
    # item y: same figure, different entity ids, label with diacritic/case variation -> identical after canonicalization
    ren = {"e1": "q7", "e2": "q8", "e3": "q9", "e4": "q10"}
    y = [(p, [ren.get(a, a) if not (p == "label" and i == 0) else a for i, a in enumerate(args)]) for p, args in base]
    y = [(p, ["Ó" if a == "O" else a for a in args]) for p, args in y]
    z = base[:3]
    facts = {"x": base, "y": y, "z": z}
    assert SM.jaccard(SM.canonical_fact_set(base), SM.canonical_fact_set(y)) < 1.0   # unlabeled e2/e4 ids differ
    names = {"x": {"e2": "C", "e4": "S"}, "y": {"q8": "C", "q10": "S"}}
    for k, m in (("x", {"e1": "L:a", "e3": "L:o"}), ("y", {"q7": "L:a", "q9": "L:o"})):
        names[k].update(m)
    assert SM.canonical_fact_set(base, names["x"]) == SM.canonical_fact_set(y, names["y"])
    out = SM.dedup_representatives({"dup_1": ["x", "y", "z"]}, facts, {"x": 8, "y": 8, "z": 3},
                                   {"x": 2.0, "y": 1.0, "z": 0.1}, th, entity_names=names)
    assert out["y"]["is_cluster_representative"] and not out["x"]["is_cluster_representative"]   # tie -> lower frag
    assert out["x"]["exclusion"] == "duplicate_representative_lost"
    assert out["z"]["is_cluster_representative"]                    # not redundant with anyone
    out2 = SM.dedup_representatives({"c": ["b_item", "a_item"]}, {"a_item": base, "b_item": base},
                                    {"a_item": 5, "b_item": 5}, {"a_item": None, "b_item": None}, th)
    assert out2["a_item"]["is_cluster_representative"]              # full tie -> smallest item_id
    with pytest.raises(SM.PipelineBug):
        SM.dedup_representatives({"c": ["q"]}, {}, {}, {}, th)


def test_funnel_table_retention_and_survival():
    units = [{"stages": ["S2"], "book_id": "a"}] * 4 + [{"stages": ["S2", "S3"], "book_id": "a"}] * 4 + \
            [{"stages": ["S2", "S3", "S4"], "book_id": "b"}] * 2
    t = SM.funnel_table(units)["all"]
    assert t["S2"]["n"] == 10 and t["S3"]["n"] == 6 and t["S4"]["n"] == 2
    assert t["S3"]["retention"] == pytest.approx(0.6) and t["S4"]["retention"] == pytest.approx(2 / 6)
    assert t["S4"]["survival"] == pytest.approx(0.2) and t["S5"]["retention"] == 0.0 and t["S6"]["retention"] is None
    by = SM.funnel_table(units, "book_id")
    assert by["b"]["S4"]["survival"] == 1.0


def test_stages_of_row_are_nested():
    row = {"gates": [{"gate_id": "G1", "passed": True}, {"gate_id": "G8b", "passed": True}],
           "is_cluster_representative": True, "outcome": "eligible"}
    assert SM.stages_of_row(row, {"built": True, "n_eligible_facts": 3}) == list(SM.PROPOSAL_STAGES)
    row["gates"][1]["passed"] = False
    assert SM.stages_of_row(row, {"built": True, "n_eligible_facts": 3}) == ["S2", "S3"]


def test_per_gate_rates():
    rows = [{"gates": [{"gate_id": "G1", "passed": True}, {"gate_id": "G2", "passed": True}]},
            {"gates": [{"gate_id": "G1", "passed": True}, {"gate_id": "G2", "passed": False}]},
            {"gates": [{"gate_id": "G1", "passed": False}]}]
    r = SM.per_gate_rates(rows, ["G1", "G2"])
    assert r["G1"]["conditional_pass_rate"] == pytest.approx(2 / 3)
    assert r["G2"]["reached"] == 2 and r["G2"]["conditional_pass_rate"] == 0.5
    assert r["G2"]["cumulative_survival"] == pytest.approx(1 / 3)


def test_s_adj_envelope_smd_by_hand():
    scores = {"i1": 0.8, "i2": 0.6, "i3": 0.2}
    strata = {"i1": "A", "i2": "A", "i3": "B"}
    r = SM.s_adj(scores, strata, {"A": 30, "B": 70})
    assert r["S_adj"] == pytest.approx(0.3 * 0.7 + 0.7 * 0.2)
    assert SM.s_adj(scores, strata, {"A": 30, "B": 60, "C": 10})["S_adj"] is None
    assert SM.envelope(0.6, 0.5) == pytest.approx([0.3, 0.8])
    with pytest.raises(ValueError):
        SM.envelope(1.2, 0.5)
    a, b = [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]
    assert SM.smd(a, b) == pytest.approx(-1.0)          # mean diff -1, both variances 1
    assert SM.smd([1.0], b) is None


def test_decile_and_survival_by_book():
    rows = [{"img_stats": {"x": float(i)}, "outcome": "excluded" if i < 50 else None} for i in range(100)]
    d = SM.decile_table(rows, "x")
    assert len(d) == 10 and d[0]["exclusion_rate"] == 1.0 and d[-1]["exclusion_rate"] == 0.0
    units = [{"book_id": "a", "stages": ["S2", "S8"]}, {"book_id": "a", "stages": ["S2"]},
             {"book_id": "b", "stages": ["S2", "S8"]}, {"book_id": "b", "stages": ["S2"]},
             {"book_id": "c", "stages": ["S2"]}, {"book_id": "c", "stages": ["S2"]}]
    s = SM.survival_by_book(units)
    assert s["survival"] == {"a": 0.5, "b": 0.5, "c": 0.0} and s["flag_2x"]["c"] and not s["flag_2x"]["a"]


def test_irls_recovers_known_coefficients_and_matches_mle():
    rng = np.random.default_rng(3)
    n = 4000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([-0.5, 1.2, -0.7])
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(float)
    fit = SM.logistic_irls(X, y)
    assert fit["converged"] and np.allclose(fit["coef"], beta, atol=0.15)
    from scipy.optimize import minimize

    def nll(b):
        eta = X @ b
        return np.sum(np.logaddexp(0, eta) - y * eta)
    ref = minimize(nll, np.zeros(3), method="BFGS").x
    assert np.allclose(fit["coef"], ref, atol=1e-4)
    se = SM.cluster_robust_se(X, y, fit["mu"], rng.integers(0, 20, n))
    assert np.all(se > 0)


def test_exclusion_logit_runs_with_book_effects():
    rng = np.random.default_rng(5)
    rows = []
    for b in range(6):
        for _ in range(60):
            x = rng.normal()
            p = 1 / (1 + np.exp(-(0.3 * b - 1 + 1.5 * x)))
            rows.append({"book_id": f"bk_{b}", "img_stats": {"ink_frac": x, "const": 1.0},
                         "outcome": "excluded" if rng.random() < p else None})
    r = SM.exclusion_logit(rows, ["ink_frac", "const"])
    assert r["fitted"] and r["dropped_constant_fields"] == ["const"]
    assert r["coefficients"]["ink_frac"]["coef_per_sd"] > 0.8 and r["n_books"] == 6


def test_family_overlap_matrix():
    m = SM.family_overlap_matrix({"c1": "fa", "c2": "fz"}, {"b1": "fa", "b2": "fb"}, {"j1": "fc", "j2": "fz"},
                                 {"cfg1": ["j1"], "cfg2": ["j2"]})
    assert m["matrix"]["c1"]["builder"]["fa"] is True and m["matrix"]["c2"]["judge"]["fz"] is True
    assert m["clean_configurations"]["c2"] == ["cfg1"]
    assert m["clean_configurations"]["c1"].startswith("none")


def test_prominence_terciles():
    mans = {f"bk_{i}": {"prominence": {"hosted_by_major_aggregators": i, "public_fulltext_layer": i > 3}}
            for i in range(6)}
    mans["bk_bad"] = {"prominence": {"hosted_by_major_aggregators": None, "public_fulltext_layer": True}}
    t = SM.prominence_terciles(mans)
    assert "bk_bad" not in t
    assert [t[f"bk_{i}"] for i in range(6)] == [1, 1, 2, 2, 3, 3]


def test_select_for_cards_one_scan_one_per_cluster_and_budget():
    """card-select (owner 2026-09-28): primary scan only, one per pHash cluster, seeded quota, dev untouched."""
    cands = ([{"item_id": f"a{i}", "book_id": "bk_a1", "group": "g1", "phash_cluster_id": "singleton"}
              for i in range(10)]
             + [{"item_id": f"b{i}", "book_id": "bk_a2", "group": "g1", "phash_cluster_id": "singleton"}
                for i in range(3)]                                  # second scan of g1's work
             + [{"item_id": f"c{i}", "book_id": "bk_c", "group": "g2", "phash_cluster_id": "pc_x" if i < 3 else "singleton"}
                for i in range(5)]                                  # 3 near-duplicates -> 1 kept
             + [{"item_id": f"d{i}", "book_id": "bk_d", "group": "gdev", "phash_cluster_id": "singleton"}
                for i in range(4)])
    sel = SM.select_for_cards(cands, {"gdev"}, budget=6, seed=7)
    ex = sel["excluded"]
    assert all(ex[f"b{i}"] == "precard_duplicate" for i in range(3))
    assert sum(1 for i in range(3) if f"c{i}" in ex) == 2 and all(ex[f"c{i}"] == "precard_duplicate"
                                                                     for i in range(3) if f"c{i}" in ex)
    assert not any(k.startswith("d") for k in ex)                   # development group never touched
    test_sel = [i for i in sel["selected"] if not i.startswith("d")]
    assert len(test_sel) <= 6 and sel["quota"] == 3                 # g2 has 3 after dedup; min(10,3)+min(3,3) = 6
    assert sum(1 for v in ex.values() if v == "selection_budget") == 10 - 3
    assert SM.select_for_cards(cands, {"gdev"}, budget=6, seed=7) == sel      # deterministic
    assert SM.select_for_cards(cands, {"gdev"}, budget=6, seed=8)["selected"] != sel["selected"]


def test_card_selection_lock_pins_the_selection(tmp_path):
    """card-lock (owner 2026-09-30): the lock returns the locked test ids and refuses a changed selection."""
    import json

    from hmb.cards.batch import card_selection_lock_path, card_selection_locked, card_selection_path
    lay = Layout(tmp_path)
    assert card_selection_locked(lay) is None                      # no lock: cards-build unchanged
    card_selection_path(lay).parent.mkdir(parents=True, exist_ok=True)
    card_selection_path(lay).write_text(json.dumps({"sha256": "a" * 64, "excluded": {}}), encoding="utf-8")
    card_selection_lock_path(lay).write_text(json.dumps({"selection_sha256": "a" * 64, "test_item_ids": ["x", "y"]}),
                                             encoding="utf-8")
    assert card_selection_locked(lay) == {"x", "y"}
    card_selection_path(lay).write_text(json.dumps({"sha256": "b" * 64, "excluded": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not the locked selection"):
        card_selection_locked(lay)
