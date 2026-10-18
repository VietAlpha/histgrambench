"""Loaders for the authoritative run artifacts. Every plotted number is read here, never typed in."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
DATA = Path(__import__("os").environ.get("HMB_ROOT", REPO / "hmb_data"))
PAPER = REPO / "research-paper"
RELEASE = "histmathbench-1.0.0"

SCORE_REPORT = DATA / "reports" / f"scores_test_{RELEASE}.json"
FUNNEL_REPORT = DATA / "reports" / "funnel_report.json"
RIGHTS = DATA / "sources" / "rights_manifest.yaml"
THRESHOLDS = DATA / "thresholds.json"
MEDIA_MANIFEST = PAPER / "media" / "sample_manifest.csv"
ROSTER = REPO / "configs" / "roster.yaml"
# Offline analyses of the stored run (research-paper/analysis/*.py) and the key-retention tally (review 2026-10-03).
REVISION_NUMBERS = PAPER / "analysis" / "revision_numbers.json"
OUTLINE_NUMBERS = PAPER / "analysis" / "outline_numbers.json"
KEY_RETENTION = PAPER / "review_20261003" / "da_key_out.json"

# Short model codes used as keys in revision_numbers.json.
SHORT = {"google.gemma-4-26b-a4b": "G26", "us.meta.llama4-maverick-17b-instruct-v1_0": "MAV",
         "mistral.ministral-3-14b-instruct": "MIN", "google.gemma-4-e2b": "E2B",
         "global.anthropic.claude-haiku-4-5-20251001-v1_0": "HAI", "nvidia.nemotron-nano-12b-v2": "NEM"}


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache
def scores() -> dict:
    return _json(SCORE_REPORT)


@lru_cache
def funnel() -> dict:
    return _json(FUNNEL_REPORT)


@lru_cache
def revision() -> dict:
    return _json(REVISION_NUMBERS)


@lru_cache
def outline_numbers() -> dict:
    return _json(OUTLINE_NUMBERS)


@lru_cache
def key_retention() -> dict:
    return _json(KEY_RETENTION)


@lru_cache
def roster() -> dict:
    """Roster roles with model ids normalised to the score report's spelling (':' -> '_')."""
    r = yaml.safe_load(ROSTER.read_text(encoding="utf-8"))
    return {role: [{**m, "model_id": m["model_id"].replace(":", "_")} for m in r[role]]
            for role in ("verifier", "builder", "judge", "contestant")}


@lru_cache
def rights() -> dict[str, dict]:
    return {b["book_id"]: b for b in yaml.safe_load(RIGHTS.read_text(encoding="utf-8"))}


@lru_cache
def items(split: str = "test") -> tuple[dict, ...]:
    return tuple(_json(p) for p in sorted((DATA / "data" / split).glob("*.json")))


def scored_items() -> list[dict]:
    """Scored test items: test split, eligible (the N of the score report)."""
    out = [d for d in items("test") if d["quality"]["eligible"]]
    n = scores()["tables"]["A"]["N"]
    if len(out) != n:
        raise RuntimeError(f"{len(out)} eligible test records but the score report has N={n}")
    return out


def item_record(item_id: str) -> tuple[str, dict]:
    for split in ("test", "development"):
        p = DATA / "data" / split / f"{item_id}.json"
        if p.exists():
            return split, _json(p)
    raise FileNotFoundError(item_id)


def is_scored(item_id: str) -> bool:
    split, d = item_record(item_id)
    return split == "test" and bool(d["quality"]["eligible"])


def repository(book: dict) -> str:
    """Repository name without the corpus-folder note, e.g. 'Internet Archive (corpus/...)' -> 'Internet Archive'."""
    name = book["acquisition"]["source_repository"].split(" (")[0].strip()
    return {"owner file": "Proprietary / Archival"}.get(name, name)


def model_order(track: str = "A") -> list[str]:
    """Models in descending Track A H (the article's table order)."""
    t = scores()["tables"][track]["models"]
    return sorted(t, key=lambda m: -t[m]["headline"]["estimate"])


def thresholds():
    """Registered parameters via the project's own loader (never inlined)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from hmb.config import load_thresholds
    return load_thresholds(THRESHOLDS)


def item_artifacts(model: str, track: str, item_id: str) -> dict:
    return {
        "score": _json(DATA / "scores" / model / track / f"{item_id}.json"),
        "verdicts": _json(DATA / "verdicts" / model / track / f"{item_id}.json"),
        "pdf": DATA / "compiled" / model / track / item_id / "cand.pdf",
    }


def card(item_id: str) -> dict:
    return _json(DATA / "semantic_cards" / f"{item_id}.json")
