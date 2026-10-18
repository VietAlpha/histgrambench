"""EXPLICITLY FAKE providers for integration tests (never imported by production code).

Every answer is derived from generated oracle graphs (Track C, plan §12.2), never from the code under
test: a fake judge says "yes" to a statement iff the statement verbalizes a fact that holds in the
closure of the CANDIDATE's oracle graph; a fake builder emits the oracle graph and votes "true" iff a
verification statement verbalizes a fact in the closure of the oracle graph. Responses are returned
through `hmb.providers.call_model(provider_override=...)`, so the cache, ledger and fake-marking paths
of production are exercised.
"""
from __future__ import annotations

import json
import re

STMT = re.compile(r"^(s\d\d|q\d\d): (.*)$", re.M)


def _body(text: str) -> str:
    for marker in ("STATEMENTS:", "PROPOSITIONS", "QUESTIONS:", "PAIRS:"):
        if marker in text:
            text = text.split(marker)[-1]
    return text


class OracleJudge:
    """Fake judge. `truth(model_id, question, statement) -> bool | None` (None = omit)."""

    def __init__(self, truth):
        self.truth = truth
        self.sends = 0

    def send(self, req, images) -> dict:
        self.sends += 1
        answers = []
        for sid, t in STMT.findall(_body(req.text)):
            v = self.truth(req.spec.model_id, req.purpose, t.strip())
            if v is None:
                continue
            answers.append({"id": sid, ("object" if req.purpose == "Q3" else "answer"): v})
        return {"text": json.dumps({"schema": "hmb-judge-verdict-2", "answers": answers}),
                "stop_reason": "end_turn", "input_tokens": 100, "output_tokens": 20,
                "model_version_string": f"{req.spec.model_id}-FAKE"}


class OracleBuilder:
    """Fake card builder: generation returns `generation_json` (the JSON Lines reply); verification votes by
    `truth`, one `<id> <vote>` line per statement."""

    def __init__(self, generation_json: str, truth):
        self.generation_json = generation_json
        self.truth = truth
        self.sends = 0

    def send(self, req, images) -> dict:
        self.sends += 1
        if req.purpose == "card_generation":
            text = self.generation_json
        else:
            text = "\n".join(f"{sid} {'true' if self.truth(t.strip()) else 'false'}"
                              for sid, t in re.findall(r"^(s\d+)[:.)]\s*(.*)$", req.text, re.M))
        return {"text": text, "stop_reason": "end_turn", "input_tokens": 200, "output_tokens": 100,
                "model_version_string": f"{req.spec.model_id}-FAKE"}
