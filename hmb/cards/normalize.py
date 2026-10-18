"""\u00a77.5 strict label normalization with a versioned confusable-glyph map.

No external terminology dictionary is used anywhere (\u00a77.5). Three views of a label string:

* ``normalize_label(t)``  the \u00a77.5 strict matching form: NFC, trim, the confusable-glyph map
                          applied to the as-printed glyphs, then Latin case folding. Used for label
                          equality (contradiction table E, oracle/grader matching).
* ``dedup_label_key(t)``  \u00a76.7 dedup comparison: case- and diacritic-insensitive (all scripts).

Alignment (§8.1, resolution C15): ``sim_label`` is 1 when the §7.5 strict-normalized labels are
equal, a neutral 0.5 when either label is absent, and -inf when both are labelled and differ. The plan's
"0.5 at edit distance 1" tier is subsumed by the §8.1 rule that differently labelled entities never
merge, so it is inactive (P and R, or A and A', are distinct points).

The map is applied before case folding so that capital I (confusable with l and 1) and lowercase i
(not confusable) stay distinct; as a consequence, after folding, L also lands in the l/1/I class.
"""
from __future__ import annotations

import unicodedata

CONFUSABLE_VERSION = "confusables-2"     # -2: prime family (C25)
NORMALIZATION_VERSION = "labelnorm-1+" + CONFUSABLE_VERSION

# Confusable classes: every glyph maps to the class representative (itself a Latin/ASCII glyph).
_CLASSES: dict[str, str] = {
    # O / 0 / Cyrillic O / Greek Omicron (upper and lower)
    "O": "O0\u041e\u039f\u2c9e",
    "o": "o\u043e\u03bf",
    # l / 1 / I / Cyrillic-Ukrainian I / Greek Iota / vertical bar
    "l": "l1I\u0406\u0399|\u01c0",
    # Latin-Greek-Cyrillic upper-case homoglyphs
    "A": "A\u0391\u0410",
    "B": "B\u0392\u0412",
    "C": "C\u0421\u03f9",
    "E": "E\u0395\u0415",
    "H": "H\u0397\u041d",
    "J": "J\u0408",
    "K": "K\u039a\u041a",
    "M": "M\u039c\u041c",
    "N": "N\u039d",
    "P": "P\u03a1\u0420",
    "S": "S\u0405",
    "T": "T\u03a4\u0422",
    "X": "X\u03a7\u0425",
    "Y": "Y\u03a5\u04ae",
    "Z": "Z\u0396",
    # lower-case homoglyphs
    "a": "a\u0430",
    "c": "c\u0441\u03f2",
    "e": "e\u0435",
    "i": "i\u0456",
    "j": "j\u0458",
    "p": "p\u0440\u03c1",
    "s": "s\u0455",
    "x": "x\u0445\u03c7",
    "y": "y\u0443",
    "v": "v\u03bd",
}

CONFUSABLE_MAP: dict[str, str] = {}
for _rep, _members in _CLASSES.items():
    for _ch in _members:
        if _ch in CONFUSABLE_MAP and CONFUSABLE_MAP[_ch] != _rep:
            raise AssertionError(f"glyph {_ch!r} in two confusable classes")
        CONFUSABLE_MAP[_ch] = _rep
# CJK variant forms: fullwidth ASCII (U+FF01-FF5E) and the ideographic space fold to ASCII.
# (CJK compatibility ideographs U+F900-FAFF are already unified by NFC.)
for _cp in range(0xFF01, 0xFF5F):
    _ascii = chr(_cp - 0xFEE0)
    CONFUSABLE_MAP[chr(_cp)] = CONFUSABLE_MAP.get(_ascii, _ascii)
CONFUSABLE_MAP["\u3000"] = " "
# Halfwidth ideographic full stop and a few CJK punctuation variants.
CONFUSABLE_MAP["\uff61"] = "\u3002"
CONFUSABLE_MAP["\u2019"] = "'"          # right single quote vs apostrophe (prime marks: A')
CONFUSABLE_MAP["\u2032"] = "'"          # prime
# Prime family (confusables-2, C25): every prime spelling folds to ASCII apostrophes, one per stroke.
CONFUSABLE_MAP[chr(0x02B9)] = "'"        # modifier letter prime
CONFUSABLE_MAP[chr(0x2033)] = "''"       # double prime
CONFUSABLE_MAP[chr(0x2034)] = "'''"      # triple prime


def _is_latin(ch: str) -> bool:
    try:
        return unicodedata.name(ch).startswith("LATIN ")
    except ValueError:
        return False


def _casefold_latin(s: str) -> str:
    return "".join(ch.casefold() if _is_latin(ch) else ch for ch in s)


def _trim(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def _conf_char(ch: str) -> str:
    return _casefold_latin(CONFUSABLE_MAP.get(ch, ch))


def normalize_label(t: str) -> str:
    """\u00a77.5 strict matching normalization (NFC, trim, confusable map, Latin case fold)."""
    s = _trim(t)
    return unicodedata.normalize("NFC", "".join(_conf_char(ch) for ch in s))


def sim_label(a: str | None, b: str | None) -> float:
    """§8.1 sim_label (C15): 1 if equal under the §7.5 strict normalization (incl. the confusable
    map); 0.5 neutral if either is unlabelled; -inf if both are labelled and differ."""
    if a is None or b is None or not _trim(a) or not _trim(b):
        return 0.5
    if normalize_label(a) == normalize_label(b):
        return 1.0
    return float("-inf")


def dedup_label_key(t: str) -> str:
    """\u00a76.7: labels compared case- and diacritic-insensitively."""
    s = unicodedata.normalize("NFD", _trim(t))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", s.casefold())


def normalization_spec() -> dict:
    """Hashable description of the normalization (enters asc_version, \u00a78.8)."""
    return {"version": NORMALIZATION_VERSION,
            "confusable_map": dict(sorted(CONFUSABLE_MAP.items())),
            "steps": ["NFC", "trim", "confusable_map(as printed)", "casefold(Latin)"]}
