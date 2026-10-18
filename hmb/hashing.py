"""Canonical JSON (RFC 8785 subset), SHA-256 helpers, content-derived identifiers (§5.1, §15.1)."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from decimal import Decimal
from pathlib import Path
from typing import Any


class CanonicalJSONError(ValueError):
    pass


def _num(x: float | int) -> str:
    if isinstance(x, bool):
        raise CanonicalJSONError("bool is not a number")
    if isinstance(x, int):
        if abs(x) > 2**53:
            raise CanonicalJSONError("integer outside IEEE-754 safe range")
        return str(x)
    if not math.isfinite(x):
        raise CanonicalJSONError("NaN/Infinity not allowed in canonical JSON")
    if x == 0:
        return "0"
    # ES6 Number.prototype.toString over Python's shortest round-trip digits.
    sign, digits, exp_raw = Decimal(repr(x)).as_tuple()
    exp = int(exp_raw)  # finite, so never 'n'/'N'/'F'
    ds = "".join(map(str, digits))
    stripped = ds.rstrip("0")
    exp += len(ds) - len(stripped)
    ds = stripped
    k = len(ds)
    n = k + exp
    if k <= n <= 21:
        s = ds + "0" * (n - k)
    elif 0 < n <= 21:
        s = ds[:n] + "." + ds[n:]
    elif -6 < n <= 0:
        s = "0." + "0" * (-n) + ds
    else:
        e = n - 1
        s = ds[0] + ("." + ds[1:] if k > 1 else "") + "e" + ("+" if e > 0 else "-") + str(abs(e))
    return ("-" if sign else "") + s


def _canon(v: Any) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return _num(v)
    if isinstance(v, str):
        return json.dumps(unicodedata.normalize("NFC", v), ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_canon(x) for x in v) + "]"
    if isinstance(v, dict):
        items = []
        for k in v:
            if not isinstance(k, str):
                raise CanonicalJSONError(f"non-string key {k!r}")
        # RFC 8785 sorts by UTF-16 code units.
        for k in sorted(v, key=lambda s: unicodedata.normalize("NFC", s).encode("utf-16-be")):
            items.append(_canon(k) + ":" + _canon(v[k]))
        return "{" + ",".join(items) + "}"
    raise CanonicalJSONError(f"unsupported type {type(v).__name__}")


def canonical_json(v: Any) -> bytes:
    """RFC 8785 canonical JSON over UTF-8 NFC strings."""
    return _canon(v).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(v: Any) -> str:
    return sha256_bytes(canonical_json(v))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def H(*parts: Any) -> str:
    """§15.1 chain concatenation `a ‖ b`: parts joined with a 0x1F unit separator.

    The plan writes `‖` without defining an encoding; a separator byte that cannot
    appear in hex digests or canonical JSON removes concatenation ambiguity.
    """
    enc = [p.encode("utf-8") if isinstance(p, str) else
           bytes(p) if isinstance(p, (bytes, bytearray)) else canonical_json(p) for p in parts]
    return sha256_bytes(b"\x1f".join(enc))


# §5.1 content-derived identifiers.
def book_id(source_file_sha256: str) -> str:
    return "bk_" + source_file_sha256[:12]


def bbox_str(bbox: list[int] | tuple[int, ...]) -> str:
    return "[" + ",".join(str(int(v)) for v in bbox) + "]"


def item_id(book: str, page_index: int, crop_bbox_px: list[int] | tuple[int, ...]) -> str:
    return "hmb_" + sha256_bytes(f"{book}:{page_index}:{bbox_str(crop_bbox_px)}".encode())[:16]


def card_uid(eligible_fact_set: list[dict], asc_version: str) -> str:
    return "card_" + sha256_bytes(canonical_json(eligible_fact_set) + asc_version.encode())[:16]
