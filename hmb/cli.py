"""Command interface: `python -m hmb <command> ...`.

Each component module may expose `register_cli(subparsers)`; the modules listed in
CLI_MODULES are imported and asked to register. Commands must set `func` on their parser.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from .config import fixture_thresholds, load_thresholds, production_template, registry_hash
from .store import Layout, write_mutable

CLI_MODULES = [
    "hmb.rights", "hmb.acquisition.cli", "hmb.sampling", "hmb.manifests", "hmb.providers",
    "hmb.cards.cli", "hmb.submission.cli", "hmb.adjudication.cli", "hmb.grading.cli",
    "hmb.synthetic.roundtrip", "hmb.ledger", "hmb.gates", "hmb.stages",
    "hmb.pipeline", "hmb.batch.cli",
]


def _utf8() -> None:
    # tasks/lessons.md L10: force UTF-8 stdout on Windows.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _cmd_thresholds_init(args) -> int:
    lay = Layout.from_env(args.root)
    th = fixture_thresholds() if args.profile == "fixture" else production_template()
    out = args.out or (lay.p("thresholds.fixture.json") if args.profile == "fixture" else lay.thresholds)
    if args.profile == "production" and out.exists() and not args.force:
        print(f"refusing to overwrite {out} (use --force before any level freeze)")
        return 2
    write_mutable(out, (json.dumps(th.to_json(), indent=1, sort_keys=True) + "\n").encode())
    print(f"wrote {out} profile={args.profile} hash={th.hash}")
    return 0


def _cmd_thresholds_status(args) -> int:
    from .config import CAL, SPECS
    th = load_thresholds(args.path)
    missing = th.missing_for_production()
    need = [n for n in missing if SPECS[n].kind == CAL]
    print(json.dumps({"profile": th.profile.value, "frozen_at": th.frozen_at,
                      "registry_hash": registry_hash(), "thresholds_hash": th.hash,
                      "not_production_ready": missing, "n_not_ready": len(missing),
                      "needs_a_value_in_provenance": need,
                      "confirmed_at_plan_value_unless_overridden": [n for n in missing if n not in need]},
                     indent=1))
    return 0


def _cmd_thresholds_freeze(args) -> int:
    """Level freeze, the end of MS3 (§17.3): every calibrated parameter needs {value, calibration_artifact_id,
    sha256, date, rule}; a provisional parameter without a record is confirmed at its plan value.
    Refuses once test-book artifacts exist (FreezeViolation)."""
    from .config import ThresholdsError, level_freeze, resolve_artifact_provenance
    from .store import FreezeViolation, read_json
    lay = Layout.from_env(args.root)
    tmpl = load_thresholds(args.template or lay.thresholds)
    try:
        prov = resolve_artifact_provenance(lay, read_json(args.provenance))
        th = level_freeze(tmpl, prov, lay)
    except (ThresholdsError, FreezeViolation) as e:
        print(f"REFUSED: {e}")
        return 2
    print(f"frozen_at={th.frozen_at} thresholds_hash={th.hash}")
    return 0


def _cmd_thresholds_provisional(args) -> int:
    """Pre-freeze values for calibrated parameters (readable only inside a pre-freeze context)."""
    from .config import ThresholdsError, set_provisional
    from .store import FreezeViolation, read_json
    lay = Layout.from_env(args.root)
    try:
        th = set_provisional(lay, read_json(args.values))
    except (ThresholdsError, FreezeViolation, FileNotFoundError) as e:
        print(f"REFUSED: {e}")
        return 2
    print(f"wrote {lay.thresholds} (pre-freeze values: {sorted(read_json(args.values))}) hash={th.hash}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hmb", description="HistMathBench v2 pipeline")
    ap.add_argument("--root", default=None, help="data root (default $HMB_ROOT or ./hmb_data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("thresholds-init", help="write a fixture or production-template thresholds file")
    p.add_argument("--profile", choices=["fixture", "production"], required=True)
    p.add_argument("--out", type=lambda s: __import__("pathlib").Path(s), default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_thresholds_init)
    p = sub.add_parser("thresholds-status", help="list parameters not yet production-ready")
    p.add_argument("path")
    p.set_defaults(func=_cmd_thresholds_status)
    p = sub.add_parser("thresholds-freeze", help="level freeze (end of MS3): provenance JSON for the calibrated parameters (provisional ones default "
                                                "to their plan values; refuses after test artifacts)")
    p.add_argument("--provenance", required=True, help="{param: {value, calibration_artifact_id, sha256, date, rule}} "
                                                       "or {param: {artifact: <root-relative pre-freeze artifact>, "
                                                       "field: <dotted key>}}")
    p.add_argument("--template", default=None, help="production template (default <root>/thresholds.json)")
    p.set_defaults(func=_cmd_thresholds_freeze)
    p = sub.add_parser("thresholds-provisional", help="pre-freeze values for calibrated parameters, readable only "
                                                     "inside a pre-freeze context (--thresholds prefreeze:PILOT|DEV)")
    p.add_argument("--values", required=True, help="{param: {value, source, rule}} (calibrated parameters only)")
    p.set_defaults(func=_cmd_thresholds_provisional)
    for mod in CLI_MODULES:
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name and (e.name == mod or mod.startswith(e.name + ".")):
                continue  # component not built yet
            raise
        reg = getattr(m, "register_cli", None)
        if reg:
            reg(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    _utf8()
    ap = build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
