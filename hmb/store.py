"""Artifact layout (plan §15.7, §5.2-5.3 paths) and immutable, atomic file writes (§5.4).

Rules enforced here:
  * write_immutable: a path, once written, may only be re-written with identical bytes;
    different bytes raise ImmutableConflict (a change must produce a new id/version).
  * writes are atomic (temp file + os.replace) so a reader never sees a partial file.
  * append_jsonl is append-only with flush+fsync; rows are never rewritten or deleted.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .hashing import canonical_json, sha256_bytes

DEFAULT_ROOT_ENV = "HMB_ROOT"
_APPEND_LOCK = threading.Lock()


class ImmutableConflict(RuntimeError):
    """An immutable artifact already exists with different content."""


def safe_model_dir(model_id: str) -> str:
    """Filesystem-safe, idempotent model directory name (Bedrock ids contain ':' which Windows rejects)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id)


class FreezeViolation(RuntimeError):
    """An action would change a frozen level after the first test-book artifact exists (§17.3)."""


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_immutable(path: str | Path, data: bytes) -> str:
    """Write bytes once; return sha256. Identical re-write is a no-op; different bytes raise."""
    path = Path(path)
    digest = sha256_bytes(data)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, path)          # exclusive: fails if another writer got there first
                return digest
            except FileExistsError:
                pass
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    existing = sha256_bytes(path.read_bytes())
    if existing != digest:
        raise ImmutableConflict(f"{path} exists with sha256 {existing}, refusing {digest}")
    return digest


def write_json_immutable(path: str | Path, obj: Any, canonical: bool = False) -> str:
    data = canonical_json(obj) if canonical else (
        json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    return write_immutable(path, data)


def write_mutable(path: str | Path, data: bytes) -> str:
    """Atomic overwrite for explicitly mutable working files (reports, status views)."""
    _atomic_write(Path(path), data)
    return sha256_bytes(data)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    with _APPEND_LOCK:
        fd = os.open(path, flags, 0o644)
        try:
            os.write(fd, line)             # one write call per row; O_APPEND positions atomically
            os.fsync(fd)
        finally:
            os.close(fd)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # A torn final line from a crash is the only tolerated defect.
                raise ValueError(f"{path}:{n}: corrupt JSONL row") from e


@dataclass(frozen=True)
class Layout:
    """Artifact paths under one data root. Mirrors plan §15.7 and the §5.2/§5.3 path fields."""

    root: Path

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> "Layout":
        r = root or os.environ.get(DEFAULT_ROOT_ENV) or Path(__file__).resolve().parent.parent / "hmb_data"
        return cls(Path(r).resolve())

    def p(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    # sources (§4, §13.1)
    @property
    def rights_manifest(self) -> Path: return self.p("sources", "rights_manifest.yaml")
    def evidence_dir(self, book_id: str) -> Path: return self.p("sources", "evidence", book_id)
    @property
    def split_groups(self) -> Path: return self.p("sources", "split_groups.yaml")
    @property
    def provenance_csv(self) -> Path: return self.p("sources", "provenance.csv")
    # items (§5.2)
    def item_record(self, split: str, item_id: str) -> Path: return self.p("data", split, f"{item_id}.json")
    def page_image(self, book_id: str, page_index: int) -> Path:
        return self.p("images", "pages", book_id, f"{page_index:04d}.png")
    def crop(self, item_id: str) -> Path: return self.p("images", "crops", f"{item_id}.png")
    def crop_canonical(self, item_id: str) -> Path: return self.p("images", "crops_canonical", f"{item_id}.png")
    def context_text(self, item_id: str) -> Path: return self.p("images", "context", f"{item_id}.txt")
    @property
    def funnel(self) -> Path: return self.p("manifests", "funnel_rows.jsonl")
    # cards (§8)
    def card(self, item_id: str) -> Path: return self.p("semantic_cards", f"{item_id}.json")
    def card_votes(self, item_id: str) -> Path: return self.p("semantic_cards", "votes", f"{item_id}.json")
    def card_manifest(self, card_uid: str, item_id: str) -> Path:
        # §5.2 names manifests/cards/<card_uid>.json, but §5.1 card_uid is item-independent, so two items
        # with identical eligible fact sets would collide (C29). Keyed per item under the card_uid.
        return self.p("manifests", "cards", card_uid, f"{item_id}.json")
    # submissions (§5.3)
    def raw_output(self, model: str, track: str, item_id: str) -> Path:
        return self.p("submissions", safe_model_dir(model), track, f"{item_id}.txt")
    def normalized_tex(self, model: str, track: str, item_id: str) -> Path:
        return self.p("submissions", safe_model_dir(model), track, f"{item_id}.tex")
    def submission_record(self, model: str, track: str, item_id: str) -> Path:
        return self.p("submissions", safe_model_dir(model), track, f"{item_id}.json")
    def compile_dir(self, model: str, track: str, item_id: str) -> Path:
        return self.p("compiled", safe_model_dir(model), track, item_id)
    def cgx(self, model: str, track: str, item_id: str) -> Path: return self.p("cgx", safe_model_dir(model), track, f"{item_id}.json")
    def verdicts(self, model: str, track: str, item_id: str) -> Path:
        return self.p("verdicts", safe_model_dir(model), track, f"{item_id}.json")
    def scores(self, model: str, track: str, item_id: str) -> Path:
        return self.p("scores", safe_model_dir(model), track, f"{item_id}.json")
    def run_manifest(self, model: str) -> Path: return self.p("scores", safe_model_dir(model), "run.json")
    # manifests, thresholds, ledger (§15, §16)
    @property
    def thresholds(self) -> Path: return self.p("thresholds.json")
    @property
    def release_manifest(self) -> Path: return self.p("manifests", "release.json")
    @property
    def tombstones(self) -> Path: return self.p("manifests", "tombstones.json")
    @property
    def pricing(self) -> Path: return self.p("manifests", "pricing.yaml")
    @property
    def ledger(self) -> Path: return self.p("manifests", "ledger.jsonl")
    @property
    def human_audit_log(self) -> Path: return self.p("manifests", "human_actions.jsonl")
    @property
    def first_test_artifact_marker(self) -> Path: return self.p("manifests", "first_test_artifact.json")
    @property
    def model_call_cache(self) -> Path: return self.p("cache", "model_calls")
    @property
    def reports(self) -> Path: return self.p("reports")

    def test_artifacts_exist(self) -> bool:
        return self.first_test_artifact_marker.exists()

    def mark_first_test_artifact(self, description: str, utc: str) -> None:
        """Called by the first writer of any test-book artifact; after this no level may change."""
        if not self.first_test_artifact_marker.exists():
            write_json_immutable(self.first_test_artifact_marker,
                                 {"description": description, "utc": utc})
