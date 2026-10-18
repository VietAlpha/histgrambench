"""Frozen builder prompts (§7.3, §8.3). Templates live in `prompt_templates/*.txt`; the entity-type and fact
tables are rendered verbatim from `hmb.ontology` (card classes only, asc-2.0.0), and the hard caps from the
registered form constants, so the rendered prompt (and its sha256, which enters asc_version, §8.8) changes
whenever the ontology or a cap does.

asc-2.0.0 (2026-09-26 streamlining): writers answer in JSON Lines (one entity or fact per line), so a reply cut
at `max_tokens` keeps every complete line; entity existence and `label` facts are carried by the entity lines;
verification answers are one `<id> <vote>` line per statement.

Staged generation (2026-09-26 repair): one generation turn over a dense crop could spend the whole `max_tokens`
allowance before its first line (Sonnet 5 at medium effort: 8,096 tokens, 0 lines). Each writer now answers two
bounded calls: `builder_generation_entities` (entity lines only) and `builder_generation_facts` (fact lines only,
over the writer's own entity lines, which the user text lists). A stage that ends at the limit with no complete
line is asked once more with `retry_note` (a different request, half the cap); nothing else is retried.
"""
from __future__ import annotations

import json
from pathlib import Path

from hmb.config import form_constant
from hmb.hashing import sha256_bytes
from hmb.ontology import ASC_MAJOR_MINOR_PATCH, CARD_PREDICATES, ENTITY_TYPES, ENUMS, arg_signature

PROMPT_DIR = Path(__file__).resolve().parent / "prompt_templates"
TEMPLATES = {"builder_generation_entities": "builder_generation_entities.txt",
             "builder_generation_facts": "builder_generation_facts.txt",
             "builder_verification": "builder_verification.txt"}
DERIVED_FROM_ENTITY_LINES = frozenset({"label"})       # plus every entity-class predicate


def fact_predicates() -> list[str]:
    """Predicates a writer states as fact lines: card relations and measure labels."""
    return [n for n, p in CARD_PREDICATES.items()
            if p.fclass.value != "entity" and n not in DERIVED_FROM_ENTITY_LINES]


def entity_param(etype: str) -> str | None:
    """The parameter kind of a parametric entity type (`int`, `enum:<name>`), None for unary types."""
    args = CARD_PREDICATES[etype].args
    return args[1] if len(args) > 1 else None


def entity_table_text() -> str:
    rows = []
    for t in ENTITY_TYPES:
        p = CARD_PREDICATES[t]
        par = entity_param(t)
        par_text = {None: "-", "int": "int >= 3 (side count)"}.get(par, par)
        rows.append(f"{t} | {par_text} | decided by {p.decidability.value}")
    return "\n".join(rows)


def predicate_table_text() -> str:
    rows = []
    for n in fact_predicates():
        p = CARD_PREDICATES[n]
        rows.append(f"{n} | {arg_signature(n)} | {p.decidability.value}" + (f" | {p.note}" if p.note else ""))
    return "\n".join(rows)


def enum_table_text() -> str:
    used = {k[5:] for p in CARD_PREDICATES.values() for k in p.args if k.startswith("enum:")}
    return "\n".join(f"enum:{k} = {' | '.join(v)}" for k, v in ENUMS.items() if k in used)


def _template(name: str) -> str:
    return (PROMPT_DIR / TEMPLATES[name]).read_text(encoding="utf-8")


def system_prompt(name: str) -> str:
    """Rendered frozen system prompt for a `TEMPLATES` name."""
    t = _template(name)
    conv = (PROMPT_DIR / "conventions.txt").read_text(encoding="utf-8").strip()
    return (t.replace("{CONVENTIONS}", conv)
             .replace("{ENTITY_TABLE}", entity_table_text())
             .replace("{PREDICATE_TABLE}", predicate_table_text())
             .replace("{ENUM_TABLE}", enum_table_text())
             .replace("{MAX_ENTITIES}", str(form_constant("card_max_entities")))
             .replace("{MAX_FACTS}", str(form_constant("card_max_facts")))
             .replace("{ASC_VERSION}", ASC_MAJOR_MINOR_PATCH))


def prompt_hashes() -> dict[str, str]:
    return {n: sha256_bytes(system_prompt(n).encode("utf-8")) for n in TEMPLATES}


def _ocr_block(ocr_words: list[dict]) -> str:
    rows = []
    for w in ocr_words or []:
        rows.append({"text": w.get("text"), "box": w.get("box") or w.get("bbox")})
    return json.dumps(rows, ensure_ascii=False, sort_keys=True)


def generation_user_text(ocr_words: list[dict], context_text: str | None) -> str:
    parts = ["OCR_WORDS: " + _ocr_block(ocr_words)]
    if context_text:
        parts.append("CONTEXT_TEXT (label disambiguation only):\n" + context_text)
    else:
        parts.append("CONTEXT_TEXT: (none)")
    return "\n\n".join(parts)


def entity_lines_text(entities: list[dict]) -> str:
    """The writer's own parsed entity lines, as the fact stage receives them (anchor rounded to 3 places)."""
    rows = []
    for e in entities:
        a = e["anchor"]
        d = {"id": e["local_id"], "type": e["type"], **({"label": e["label"]} if e.get("label") else {}),
             **({"param": e["param"]} if e.get("param") is not None else {}),
             "anchor": [round(float(a[k]), 3) for k in ("x", "y", "w", "h")]}
        rows.append(json.dumps(d, ensure_ascii=False))
    return "\n".join(rows)


def fact_stage_user_text(ocr_words: list[dict], context_text: str | None, entities: list[dict]) -> str:
    return generation_user_text(ocr_words, context_text) + "\n\nENTITIES:\n" + entity_lines_text(entities)


def retry_note(cap: int) -> str:
    """Appended to a stage's user text for its single retry (the stage ended at the output limit before its
    first complete line). `cap` is half the stage's registered cap, rounded up."""
    return (f"\n\nRETRY: a previous reply to this request reached the output limit before its first complete "
            f"line. This time give at most {cap} lines, the most important first, and write the first line "
            f"immediately.")


def verification_user_text(ocr_words: list[dict], statements: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{sid}: {text}" for sid, text in statements)
    return "OCR_WORDS: " + _ocr_block(ocr_words) + "\n\nSTATEMENTS:\n" + lines
