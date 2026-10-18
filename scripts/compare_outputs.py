"""Compare regenerated outputs with the stored ones (reproduction levels 1 and 2).

    python scripts/compare_outputs.py STORED REGENERATED [--ignore KEY ...] [--tol 1e-9]

Both arguments may be files or directories (directories are compared file by file over the stored side).
JSON files are compared structurally; numbers may differ by at most --tol (absolute). Keys named with --ignore
(default: created, created_utc, utc, execution_context) are skipped at any depth, because they record when and
where a file was written, not what it says. Other files are compared byte for byte, after normalising CRLF to LF.
Exit status 0 means every compared file agrees.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_IGNORE = ("created", "created_utc", "utc", "execution_context")


def diff_json(a, b, path: str, ignore: set[str], tol: float, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ignore:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'stored' if k in a else 'regenerated'}")
            else:
                diff_json(a[k], b[k], f"{path}/{k}", ignore, tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_json(x, y, f"{path}[{i}]", ignore, tol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not (math.isclose(a, b, rel_tol=0.0, abs_tol=tol) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}")


def compare_file(s: Path, r: Path, ignore: set[str], tol: float) -> list[str]:
    if not r.exists():
        return [f"{r}: missing"]
    if s.suffix == ".json":
        out: list[str] = []
        diff_json(json.loads(s.read_text(encoding="utf-8")), json.loads(r.read_text(encoding="utf-8")), "", ignore, tol, out)
        return out
    a = s.read_bytes().replace(b"\r\n", b"\n")
    b = r.read_bytes().replace(b"\r\n", b"\n")
    return [] if a == b else [f"{s.name}: content differs"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stored", type=Path)
    ap.add_argument("regenerated", type=Path)
    ap.add_argument("--ignore", nargs="*", default=list(DEFAULT_IGNORE))
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--glob", default="*", help="file pattern when comparing directories")
    a = ap.parse_args()
    pairs = ([(a.stored, a.regenerated)] if a.stored.is_file() else
             [(p, a.regenerated / p.relative_to(a.stored)) for p in sorted(a.stored.rglob(a.glob)) if p.is_file()])
    bad = 0
    for s, r in pairs:
        d = compare_file(s, r, set(a.ignore), a.tol)
        print(f"{'OK  ' if not d else 'DIFF'} {s.name}" + ("" if not d else f" ({len(d)} differences)"))
        for line in d[:15]:
            print("     ", line)
        bad += bool(d)
    print(f"{len(pairs) - bad} of {len(pairs)} files agree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
