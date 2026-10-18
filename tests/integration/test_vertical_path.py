"""Scaffold 2 acceptance: the integrated vertical path on a generated control (real sandbox).

Independent expectations only: outcome codes from §9.5 for known inputs; the flip set Φ from the
generator's derived diff; S from §11.2/§11.3 arithmetic recomputed here from the verdict file
(caps = inf in the fixture profile, so S is plain micro-recall over non-EXEMPT facts).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hmb.store import ImmutableConflict  # noqa: E402

pytestmark = pytest.mark.docker


def _need_sandbox():
    from hmb.submission.sandbox import sandbox_available
    if not sandbox_available():
        pytest.skip("sandbox image unavailable (Docker)")


def _strict_recall(verdicts: dict, card) -> float:
    """§11.3: PRESENT / scorable over the strict classes (layout is diagnostic only)."""
    strict = {f.fact_id for f in card.facts if f.eligible and f.fact_class != "layout"}
    fs = [f for f in verdicts["facts"] if f["fact_id"] in strict and f["verdict"] != "EXEMPT"]
    return sum(f["verdict"] == "PRESENT" for f in fs) / len(fs)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    _need_sandbox()
    import vertical
    return vertical.run(tmp_path_factory.mktemp("vertical"))


def test_outcomes_match_known_inputs(run):
    for card in ("oracle_card",):
        for key, exp in (("valid", "OK"), ("mutation", "OK"), ("compile_failure", "C_ERROR")):
            res, _ = run["results"][(card, key)]
            assert res.outcome == exp, (key, res.outcome)


def test_compile_failure_scores_zero_without_adjudication(run):
    res, sends = run["results"][("oracle_card", "compile_failure")]
    assert res.score["S"] == 0.0 and res.verdicts is None and sends == 0
    assert res.submission["outcome_code"] == "C_ERROR" and res.submission["cgx_record"] is None


def test_valid_candidate_scores_per_formula(run):
    for card, c in (("oracle_card", run["card_a"]), ("built_card", run["card_b"])):
        res, _ = run["results"][(card, "valid")]
        assert res.score["evaluator_failure"] is None
        assert res.score["S"] == pytest.approx(_strict_recall(res.verdicts, c))
        # §12.7 V1 target for an unmutated clean candidate: recall 1
        assert res.score["S"] == pytest.approx(1.0), res.summary()


def test_semantic_mutation_is_detected(run):
    phi = run["cands"]["mutation"]["phi_removed"]
    assert phi, "generator produced an empty derived diff"
    from hmb.cards.closure import canon
    res, _ = run["results"][("oracle_card", "mutation")]
    card = run["card_a"]
    by_key = {canon((f.predicate, f.args)): f.fact_id for f in card.facts}
    verdict = {f["fact_id"]: f["verdict"] for f in res.verdicts["facts"]}
    hit = [by_key[k] for k in phi if k in by_key]
    assert hit, "no Φ fact is on the card"
    assert all(verdict[fid] != "PRESENT" for fid in hit), {fid: verdict[fid] for fid in hit}
    assert res.score["S"] == pytest.approx(_strict_recall(res.verdicts, card))
    valid, _ = run["results"][("oracle_card", "valid")]
    assert res.score["S"] < valid.score["S"]


def test_final_submission_records_link_all_artifacts(run):
    import hashlib
    lay = run["layout"]
    for key in ("valid", "mutation"):
        res, _ = run["results"][("oracle_card", key)]
        sub = res.submission
        for f, h in (("cgx_record", "cgx_sha256"), ("verdicts", "verdicts_sha256"), ("scores", "scores_sha256")):
            p = lay.root / sub[f]                       # §5.3 field names
            assert hashlib.sha256(p.read_bytes()).hexdigest() == sub[h]


def test_rerun_resumes_without_new_calls_or_overwrites(run):
    """Repeated run: every stage returns its stored artifact; the cache serves every model call."""
    import vertical
    from hmb.pipeline import run_cell
    spec, lay, th = run["spec"], run["layout"], run["th"]
    for key in ("valid", "mutation"):
        cid = f"{spec.spec_id}__{key}"
        judge = vertical.OracleJudge(lambda *a: True)          # would answer differently if called
        again = run_cell(run["card_a"].model_copy(update={"item_id": cid}), run["cands"][key]["raw"],
                         "end_turn", item_id=cid, contestant=vertical.CONTESTANT, track="C", layout=lay,
                         th=th, judges=vertical.JUDGES, ledger=run["ledger"], allow_live=True,
                         crop_png=None, meta=vertical.META, provider_override=judge)
        first, _ = run["results"][("oracle_card", key)]
        assert judge.sends == 0
        assert again.score == first.score


def test_changed_evaluator_input_is_refused_not_overwritten(run):
    """A different judge configuration must not reuse or overwrite stored verdicts."""
    import vertical
    from hmb.adjudication import AdjudicationError
    from hmb.pipeline import run_cell
    spec, lay, th = run["spec"], run["layout"], run["th"]
    cid = f"{spec.spec_id}__valid"
    judges2 = [j.model_copy(update={"snapshot": "2099-01-01"}) for j in vertical.JUDGES]
    with pytest.raises((AdjudicationError, ImmutableConflict)):
        run_cell(run["card_a"].model_copy(update={"item_id": cid}), run["cands"]["valid"]["raw"],
                 "end_turn", item_id=cid, contestant=vertical.CONTESTANT, track="C", layout=lay, th=th,
                 judges=judges2, ledger=run["ledger"], allow_live=True, crop_png=None, meta=vertical.META,
                 provider_override=vertical.OracleJudge(lambda *a: True))


def test_ledger_and_report(run):
    import json
    rep = json.loads(run["report"].read_text(encoding="utf-8"))
    assert rep["profile"] == "fixture" and "not a benchmark score" in rep["claim_status"]
    assert rep["harness_version"] and rep["metric_hash"] and rep["thresholds_hash"]
    assert run["ledger"].spent() == 0.0                         # zero-priced fixture entries
    rows = (run["layout"].ledger).read_text(encoding="utf-8").splitlines()
    assert rows and all('"reserve"' in r or '"settle"' in r or '"release"' in r for r in rows)
