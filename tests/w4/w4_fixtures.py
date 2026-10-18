"""Opus 4 test fixtures: cards, CGX records, thresholds and an explicitly marked FAKE judge provider.

The fake provider replaces `hmb.adjudication.judges._call_model` (the indirection to the Opus 1
`call_model` contract); every response it returns has `fake=True`.
"""
from __future__ import annotations

import json
import math
import re

import pytest
from PIL import Image

from hmb.config import fixture_thresholds
from hmb.hashing import sha256_json
from hmb.ontology import PREDICATES
from hmb.schemas import (Card, CanonEntity, CgxFact, CgxPrimitive, CgxRecord, FactRecord, ModelResponse,
                         ModelSpec)
from hmb.store import Layout, write_json_immutable


def fact(fid: str, pred: str, args: list, eligible: bool = True) -> FactRecord:
    p = PREDICATES[pred]
    return FactRecord.model_validate({
        "fact_id": fid, "class": p.fclass.value, "predicate": pred, "args": args,
        "strictness": p.strictness.value, "decidability": p.decidability.value, "anchor": p.anchor.value,
        "eligible": eligible})


def card(item_id: str, entities: list[tuple], facts: list[FactRecord], abstained=None) -> Card:
    return Card(item_id=item_id, asc_version="asc-1.0.0", card_uid="card_" + "0" * 16,
                entities=[CanonEntity(entity_id=e[0], type=e[1], label=(e[2] if len(e) > 2 else None))
                          for e in entities],
                facts=facts, abstained=abstained or [],
                builder_order=["b1", "b2", "b3", "b4"],
                family_map={"b1": "fam_b1", "b2": "fam_b2", "b3": "fam_b3", "b4": "fam_b1"},
                profile="fixture")


def thresholds(**over):
    """Fixture thresholds; judging needs no fitted reliabilities (majority rule, 2026-09-25)."""
    th = fixture_thresholds()
    for k, v in over.items():
        th.params[k] = {"value": v, "status": th.params[k]["status"], "provenance": None}
    return th


def judges() -> list[ModelSpec]:
    return [ModelSpec(model_id=f"J{i}", provider="fake", family=f"fam_j{i}", role="judge") for i in (1, 2, 3, 4)]


# ------------------------------------------------------------------ the triangle scenario
def triangle_card(extra_model_facts: list[FactRecord] | None = None) -> Card:
    ents = [("p_A", "point", "A"), ("p_B", "point", "B"), ("p_C", "point", "C"), ("p_D", "point", "D"),
            ("s_1", "segment"), ("s_2", "segment"), ("s_3", "segment"), ("s_4", "segment")]
    fs = [fact("e1", "point", ["p_A"]), fact("e2", "point", ["p_B"]), fact("e3", "point", ["p_C"]),
          fact("e4", "point", ["p_D"]), fact("e5", "segment", ["s_1"]), fact("e6", "segment", ["s_2"]),
          fact("e7", "segment", ["s_3"]), fact("e8", "segment", ["s_4"]),
          fact("r1", "endpoint_of", ["p_A", "s_1"]), fact("r2", "endpoint_of", ["p_B", "s_1"]),
          fact("r3", "endpoint_of", ["p_B", "s_2"]), fact("r4", "endpoint_of", ["p_C", "s_2"]),
          fact("r5", "endpoint_of", ["p_C", "s_3"]), fact("r6", "endpoint_of", ["p_A", "s_3"]),
          fact("r7", "endpoint_of", ["p_D", "s_4"]), fact("r8", "equal_length", ["s_1", "s_3"]),
          fact("l1", "label", ["A", "p_A"]), fact("l2", "label", ["B", "p_B"]), fact("l3", "label", ["C", "p_C"]),
          fact("l4", "label", ["D", "p_D"]),
          fact("z1", "perpendicular", ["s_1", "s_2"], eligible=False)]
    return card("hmb_0000000000000001", ents, fs + (extra_model_facts or []))


def triangle_cgx(extra_prims=(), extra_facts=(), labels=None, glyph_h=8.0) -> CgxRecord:
    P = [CgxPrimitive(prim_id="P1", type="point", geom={"x": 0.0, "y": 0.0}),
         CgxPrimitive(prim_id="P2", type="point", geom={"x": 80.0, "y": 0.0}),
         CgxPrimitive(prim_id="P3", type="point", geom={"x": 0.0, "y": 60.0}),
         CgxPrimitive(prim_id="S1", type="segment", geom={"p": [0.0, 0.0], "q": [80.0, 0.0]}),
         CgxPrimitive(prim_id="S2", type="segment", geom={"p": [80.0, 0.0], "q": [0.0, 60.0]}),
         CgxPrimitive(prim_id="S3", type="segment", geom={"p": [0.0, 60.0], "q": [0.0, 0.0]}),
         CgxPrimitive(prim_id="K1", type="circle", geom={"c": [200.0, 200.0], "r": 20.0})]
    P += list(extra_prims)
    F = [("c01", "point", ["P1"]), ("c02", "point", ["P2"]), ("c03", "point", ["P3"]),
         ("c04", "segment", ["S1"]), ("c05", "segment", ["S2"]), ("c06", "segment", ["S3"]),
         ("c07", "circle", ["K1"]),
         ("c08", "endpoint_of", ["P1", "S1"]), ("c09", "endpoint_of", ["P2", "S1"]),
         ("c10", "endpoint_of", ["P2", "S2"]), ("c11", "endpoint_of", ["P3", "S2"]),
         ("c12", "endpoint_of", ["P3", "S3"]), ("c13", "endpoint_of", ["P1", "S3"]),
         ("c14", "left_of", ["P1", "P2"])]
    facts = [CgxFact(cfact_id=c, predicate=p, args=a, margin=5.0) for c, p, a in F] + list(extra_facts)
    h = glyph_h
    labs = labels if labels is not None else [
        {"text": "A", "bbox": [-9.0, -9.0, -9.0 + h * 0.6, -9.0 + h], "kind": "name", "bound_prim": "P1"},
        {"text": "B", "bbox": [84.0, -9.0, 84.0 + h * 0.6, -9.0 + h], "kind": "name", "bound_prim": "P2"},
        {"text": "C", "bbox": [-9.0, 62.0, -9.0 + h * 0.6, 62.0 + h], "kind": "name", "bound_prim": "P3"}]
    return CgxRecord(extractor_version="cgx-test-1", item_id="hmb_0000000000000001", model_id="contestant_x",
                     track="A", pdf_sha256="a" * 64, diagonal_D=math.hypot(220, 220), primitives=P,
                     glyphs=[{"text": l["text"], "bbox": l["bbox"]} for l in labs], labels=labs, facts=facts,
                     path_op_count=12)


# ------------------------------------------------------------------ fake judge provider
STMT = re.compile(r"^(s\d\d|q\d\d): (.*)$", re.M)


class FakeJudges:
    """Explicitly FAKE judge provider. `policy(model_id, question, stmt_text) -> bool | str | None`
    (None = omit the answer). `malformed[model_id] = n` makes the first n calls return garbage;
    `refuse[model_id]` makes every call a refusal."""

    def __init__(self, policy=None):
        self.policy = policy or (lambda j, q, t: default_truth(t))
        self.calls: list = []
        self.malformed: dict[str, int] = {}
        self.malformed_if = None                 # callable(req) -> bool
        self.refuse: set[str] = set()

    def __call__(self, req, layout, ledger, *, allow_live, provider_override=None) -> ModelResponse:
        self.calls.append(req)
        j = req.spec.model_id
        key = sha256_json(req.model_dump(mode="json"))
        if j in self.refuse:
            text = json.dumps({"schema": "hmb-judge-verdict-2", "refusal": True})
        elif self.malformed.get(j, 0) > 0 or (self.malformed_if is not None and self.malformed_if(req)):
            if self.malformed.get(j, 0) > 0:
                self.malformed[j] -= 1
            text = "I think the answer is yes."
        else:
            answers = []
            body = req.text.split("STATEMENTS:")[-1].split("PROPOSITIONS")[-1].split("QUESTIONS:")[-1] \
                .split("PAIRS:")[-1]
            for sid, t in STMT.findall(body):
                v = self.policy(j, req.purpose, t)
                if v is None:
                    continue
                answers.append({"id": sid, ("object" if req.purpose == "Q3" else "answer"): v})
            text = json.dumps({"schema": "hmb-judge-verdict-2", "answers": answers})
        return ModelResponse(text=text, model_version_string=f"{j}-fake", cache_key=key, fake=True)

    def count(self, question=None, judge=None) -> int:
        return sum(1 for r in self.calls if (question is None or r.purpose == question)
                   and (judge is None or r.spec.model_id == judge))


def default_truth(t: str) -> bool:
    if "circle" in t:                                    # the extra circle is not in the source
        return False
    return True


@pytest.fixture
def fake(monkeypatch):
    f = FakeJudges()
    monkeypatch.setattr("hmb.adjudication.judges._call_model", f)
    return f


@pytest.fixture
def lay(tmp_path):
    return Layout(tmp_path / "root")


@pytest.fixture
def images(tmp_path):
    r = tmp_path / "render.png"
    c = tmp_path / "crop.png"
    Image.new("RGB", (1024, 1024), "white").save(r)
    Image.new("RGB", (1024, 1024), "white").save(c)
    return r, c


def write_submission_stub(lay: Layout, model="contestant_x", track="A", item="hmb_0000000000000001",
                          family="fam_c") -> None:
    """The compile-stage record (adjudicate never reads the final submission record)."""
    write_json_immutable(lay.compile_dir(model, track, item) / "submission_stage.json", {"model_family": family})


def write_drift_dev(lay: Layout, c: Card, render, judge_ids, fact_ids, original=None, model="m", track="A") -> list[dict]:
    """Development record the replay drift probe reads (owner decision 2026-09-30): the card, one cached original
    Q1 call per judge (chunked STATEMENTS block, the render) and a verdict file whose facts carry each judge's Q1
    vote with that call's key. `original(judge_id, fact_id) -> bool` is the recorded answer (default True).
    Returns the frozen triples (drift-triples form)."""
    from hmb.adjudication.serialize import fact_statement, names_for_card
    from hmb.schemas import FactVerdict, JudgeVote, VerdictFile
    original = original or (lambda j, f: True)
    write_json_immutable(lay.card(c.item_id), c.model_dump(mode="json", by_alias=True))
    names, facts = names_for_card(c), {f.fact_id: f for f in c.facts}
    text = "header\n\nSTATEMENTS:\n" + "".join(f"s{n + 1:02d}: {fact_statement(facts[f], names)}\n"
                                              for n, f in enumerate(fact_ids))
    votes = {f: [] for f in fact_ids}
    for k, j in enumerate(judge_ids):
        key = f"{k + 1:02x}" * 32
        write_json_immutable(lay.model_call_cache / key[:2] / f"{key}.json",
                             {"cache_key": key, "request": {"text": text, "image_paths": [str(render)]}, "response": {}})
        for f in fact_ids:
            votes[f].append(JudgeVote(judge_id=j, judge_family=f"fam_{j.lower()}", question="Q1",
                                      positive=original(j, f), raw_call_sha256=key))
    vf = VerdictFile(item_id=c.item_id, model_id=model, track=track, judge_config_hash="cfg-dev", profile="fixture",
                     facts=[FactVerdict(fact_id=f, verdict="PRESENT", source="judges", votes=votes[f]) for f in fact_ids])
    write_json_immutable(lay.verdicts(model, track, c.item_id), vf.model_dump(mode="json"))
    return [{"triple_id": f"{model}|{track}|{c.item_id}|{f}", "fact_text": fact_statement(facts[f], names),
             "render_png": str(render), "reference": True} for f in fact_ids]
