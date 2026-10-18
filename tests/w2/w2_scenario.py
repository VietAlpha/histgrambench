"""Shared engineering fixtures for tests/w2 (fixture profile only; never evidence)."""
from __future__ import annotations

import copy
import json

from hmb.cards.align import AlignParams
from hmb.config import Thresholds, fixture_thresholds
from hmb.vocab import Profile

FAM = {"b1": "fam_x", "b3": "fam_y"}          # the locked roster shape: 2 writers in 2 families
ORDER = ["b1", "b3"]


def th_with(**vals) -> Thresholds:
    th = fixture_thresholds()
    for k, v in vals.items():
        th.params[k] = {"value": v, "status": "fixture", "provenance": None}
    return th


def card_th(**extra) -> Thresholds:
    """Fixture thresholds for card building: the §8.4 rule is fixed, so nothing is calibrated."""
    return th_with(**extra)


def as_production(th: Thresholds) -> Thresholds:
    params = copy.deepcopy(th.params)
    for v in params.values():
        if v["status"] == "fixture":
            v["status"] = "frozen"
    return Thresholds(Profile.production, params, "2026-09-22T00:00:00+00:00", "test")


PARAMS = AlignParams(w_L=1.0, w_G=1.0, w_R=0.5, theta=0.9, delta=0.15, sigma=0.05, wl_rounds=2)


def _entity_line(e: dict) -> dict:
    """Accepts the asc-2.0.0 entity line or the older {local_id, anchor: {x, y, w, h}} shape."""
    if "id" in e:
        return e
    a = e["anchor"]
    out = {"id": e["local_id"], "type": e["type"], "anchor": [a["x"], a["y"], a["w"], a["h"]]}
    if e.get("label") is not None:
        out["label"] = e["label"]
    for k in ("param", "label_from_context"):
        if k in e:
            out[k] = e[k]
    return out


def _fact_line(f: dict) -> dict:
    return f if "fact" in f else {"fact": f["predicate"], "args": f["args"]}


def triangle_json(builder: str, *, extra_facts=(), extra_entities=(), jitter: float = 0.0,
                  labels=("A", "B", "C")) -> str:
    """A labelled triangle ABC with sides s1=AB, s2=BC, s3=CA (crop-relative anchors), as the asc-2.0.0
    JSON Lines writer reply (entity lines, then fact lines)."""
    pos = {"A": (0.2, 0.8), "B": (0.8, 0.8), "C": (0.5, 0.2)}
    ents = []
    for lid, lab, key in zip(("pA", "pB", "pC"), labels, "ABC"):
        x, y = pos[key]
        ents.append({"id": lid, "type": "point", **({"label": lab} if lab is not None else {}),
                     "anchor": [x + jitter, y, 0.02, 0.02]})
    segs = {"s1": ("pA", "pB"), "s2": ("pB", "pC"), "s3": ("pC", "pA")}
    for sid, (a, b) in segs.items():
        ax, ay = pos[a[1]]
        bx, by = pos[b[1]]
        ents.append({"id": sid, "type": "segment", "anchor": [(ax + bx) / 2 + jitter, (ay + by) / 2,
                                                             abs(ax - bx), abs(ay - by)]})
    facts = []
    for sid, (a, b) in segs.items():
        facts += [{"fact": "endpoint_of", "args": [a, sid]}, {"fact": "endpoint_of", "args": [b, sid]}]
    facts += [_fact_line(f) for f in extra_facts]
    ents += [_entity_line(e) for e in extra_entities]
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in ents + facts)


def parse(raw: str, builder: str, family: str | None = None):
    from hmb.cards.parse import parse_builder_output
    return parse_builder_output(raw, builder, family or FAM[builder], max_entities=40, max_facts=60)
