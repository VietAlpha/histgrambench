"""Oracle graph (known-geometry fixtures, docs/CONTRACTS.md) -> Card with every fact eligible.

Used for direct oracle-card input in the offline tests (isolating downstream correctness).
The oracle facts are re-canonicalized through the same closure/MGS as real cards; entity-existence
facts of unary types and `label` facts from the entities' labels are added (set semantics, so an
oracle that already lists them is unchanged). Only card-class facts (E, R, L; asc-2.0.0) are kept: layout
facts of the graph are extractor-internal and never on a card. No card-class fact is entailed only by a
layout fact (the rule set derives layout from relations, never the reverse), so filtering after the MGS
equals the MGS of the filtered set. Entity anchors are the oracle point coordinates
mapped to crop-relative form (x / w, 1 - y / h: TikZ y points up, crop y points down).
"""
from __future__ import annotations

from hmb.hashing import card_uid, sha256_json
from hmb.ontology import ASC_MAJOR_MINOR_PATCH, CARD_PREDICATES, PREDICATES
from hmb.schemas import BuilderAnchor, Card, CanonEntity, FactRecord

from .closure import canon, closure_spec, minimal_generating_set
from .consensus import assign_fact_ids
from .normalize import normalization_spec
from .parse import UNARY_ENTITY_TYPES

ORACLE_BUILDER = "oracle"


def oracle_asc_version() -> str:
    h = sha256_json({"closure": closure_spec(), "normalization": normalization_spec()})[:16]
    return f"{ASC_MAJOR_MINOR_PATCH}+oracle-{h}"


def _anchor(geom: dict | None, canvas: dict | None) -> BuilderAnchor | None:
    if not geom or "x" not in geom or "y" not in geom or not canvas:
        return None
    w, h = float(canvas["w"]), float(canvas["h"])
    return BuilderAnchor(x=float(geom["x"]) / w, y=1.0 - float(geom["y"]) / h, w=0.0, h=0.0)


def oracle_facts(oracle_graph: dict) -> tuple[list, dict[str, str]]:
    """Γ as the MGS of the graph's facts plus entity-existence and label facts from its entities
    (the single definition used by `oracle_card` and `coverage.card_vs_oracle`)."""
    ents = oracle_graph["entities"]
    types = {e["entity_id"]: e["type"] for e in ents}
    facts = {canon((f["predicate"], f["args"])) for f in oracle_graph["facts"]}
    for e in ents:
        if e["type"] in UNARY_ENTITY_TYPES:
            facts.add(canon((e["type"], [e["entity_id"]])))
        if e.get("label"):
            facts.add(canon(("label", [e["label"], e["entity_id"]])))
    return [f for f in minimal_generating_set(facts, types) if f[0] in CARD_PREDICATES], types


def oracle_card(oracle_graph: dict, *, profile: str = "production") -> Card:
    ents = oracle_graph["entities"]
    M, _ = oracle_facts(oracle_graph)
    ids = assign_fact_ids(M)
    recs = []
    for f in sorted(M, key=lambda k: ids[k]):
        p = PREDICATES[f[0]]
        recs.append(FactRecord.model_validate({
            "fact_id": ids[f], "class": p.fclass.value, "predicate": f[0],
            "args": [list(a) if isinstance(a, tuple) else a for a in f[1]],
            "strictness": p.strictness.value, "decidability": p.decidability.value, "anchor": p.anchor.value,
            "eligible": True}))
    asc = oracle_asc_version()
    canvas = oracle_graph.get("canvas")
    entities = [CanonEntity(entity_id=e["entity_id"], type=e["type"], label=e.get("label"),
                            anchor=_anchor(e.get("geom"), canvas), members={ORACLE_BUILDER: e["entity_id"]})
                for e in ents]
    return Card(item_id=str(oracle_graph.get("spec_id", "")), asc_version=asc,
                card_uid=card_uid([{"predicate": r.predicate, "args": r.args} for r in recs], asc),
                entities=entities, facts=recs,
                builder_order=[ORACLE_BUILDER], family_map={ORACLE_BUILDER: ORACLE_BUILDER}, profile=profile)
