"""Oracle-graph JSON for a known-geometry specification (docs/CONTRACTS.md, "Oracle graph").

The oracle graph is the fixture answer key of the offline software tests (vertical path, model-run
integration, CGX round trip, card normalization). It is not a benchmark artifact.
"""
from __future__ import annotations

from .evaluator import OracleFacts, fact_dicts
from .spec import Spec, resolve_entities
from .tikz import emit_tikz


def oracle_graph(spec: Spec, facts: OracleFacts, partition: str = "fixture") -> dict:
    """docs/CONTRACTS.md oracle-graph JSON. An entity's `label` is the text BOUND to it in Γ (the
    §10.3 rule), which equals the drawn label for admitted base specifications."""
    bound = {}
    for p, a in facts.mgs:
        if p == "label":
            bound.setdefault(a[1], a[0])
    ents = [{"entity_id": e["entity_id"], "type": e["type"], "label": bound.get(e["entity_id"]),
             "geom": e["geom"]} for e in (facts.entities or resolve_entities(spec))]
    return {"spec_id": spec.spec_id, "family": spec.family, "seed": spec.seed, "partition": partition,
            "canvas": dict(spec.canvas), "entities": ents, "facts": fact_dicts(facts.mgs),
            "tikz": emit_tikz(spec)}


__all__ = ["oracle_graph"]
