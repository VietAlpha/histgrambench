"""Scaffold 4: `hmb drift-probe --window open|close` (§10.9 judge drift probe; owner Opus 4).

Reads the frozen triple set (JSON list of {triple_id, fact_text, render_png, reference: bool};
render paths relative to the root or absolute), judges it with each roster judge (or --judge), and
writes manifests/drift_probe/<window>.json. At `close` it also evaluates the window against the
stored `open` answers and writes manifests/drift_probe/result.json, the drift_probe input of the run
manifest (§15.3: open_agreement, close_agreement, kappa_min, pass). Under the fixture profile the
result is engineering evidence and `pass` is None (no gate PASS is ever emitted from fixture data).
Model calls go through hmb.providers.call_model; without --allow-live only cached calls are used.
"""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from hmb.hashing import sha256_file
from hmb.store import Layout, read_json, write_json_immutable

from .protocol import evaluate_drift, judge_config_hash, own_q1_votes, run_drift_probe




def drift_probe(layout: Layout, th, *, window: str, judges, triples_path: Path, allow_live: bool,
                phase: str = "FINAL", ledger=None, provider_override=None) -> dict:
    if window not in ("open", "close"):
        raise ValueError(window)
    triples = read_json(triples_path)
    for t in triples:
        p = Path(t["render_png"])
        t["render_png"] = p if p.is_absolute() else layout.root / p
    tag = {"profile": th.profile.value, "thresholds_hash": th.hash,
           "triples_sha256": sha256_file(triples_path), "n_triples": len(triples),
           "judge_config_hash": judge_config_hash(judges, "drift-probe", th),
           "claim_status": ("engineering evidence only (fixture profile)" if not th.is_production
                            else "production")}
    base = layout.p("manifests", "drift_probe")
    op = base / "open.json"
    if window == "open":
        # unique per window (2026-10-01): the probe's cache scope carries it, so a copied cache can never answer
        # a new window's probe with an earlier window's calls
        window_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)
    else:
        if not op.exists():
            raise FileNotFoundError("close window needs manifests/drift_probe/open.json")
        opened = read_json(op)
        window_id = opened.get("window_id")      # None for an open record written before 2026-10-01
    window_tag = f"{window}|{window_id}" if window_id else window
    answers = {j.model_id: run_drift_probe(triples, j, layout=layout, th=th, ledger=ledger,
                                           allow_live=allow_live, provider_override=provider_override,
                                           phase=phase, window_tag=window_tag) for j in judges}
    # references (owner decision 2026-09-30): each judge's own development Q1 answer on the triples it voted on
    references = {j.model_id: {t: v[1] for t, v in own_q1_votes(layout, triples, j.model_id).items()} for j in judges}
    rec = {**tag, "window": window, "window_id": window_id, "reference_rule": "own-q1-vote-replay-1",
           "references": references, "answers": answers}
    write_json_immutable(base / f"{window}.json", rec)
    if window == "open":
        return rec
    if opened.get("profile") != tag["profile"]:
        raise ValueError(f"profile mismatch: thresholds are {tag['profile']!r} but {op} is {opened.get('profile')!r}")
    if opened["triples_sha256"] != tag["triples_sha256"]:
        raise ValueError("open and close were run on different triple sets")
    per = {j: evaluate_drift(references[j], opened["answers"].get(j, {}), answers[j], th) for j in answers}
    valid = all(r["valid"] for r in per.values())
    res = {**tag, "per_judge": per,
           "open_agreement": min(r["agreement_open"] for r in per.values()),
           "close_agreement": min(r["agreement_close"] for r in per.values()),
           "open_vs_close": min(r["open_vs_close"] for r in per.values()),
           "kappa_min": float(th.value("kappa_min")), "alpha_min": float(th.value("alpha_min")),
           "pass": valid if th.is_production else None,
           "engineering_valid": valid}
    write_json_immutable(base / "result.json", res)
    return res


def drift_triples(layout: Layout, th, *, seed: int) -> list[dict]:
    """§10.9 frozen triple set from judged development-set cells (checkpoint simplification 2026-09-26):
    every model-decided fact of a development-book item that the judges decided (PRESENT or ABSENT) is a
    candidate triple (its statement as the judges read it, the candidate render, the recorded verdict as
    `reference`); `drift_probe_n_triples` of them are drawn with random.Random(seed)."""
    import random

    from hmb.acquisition.drivers import development_books
    from hmb.cards.batch import candidates_dir
    from hmb.schemas import Card, VerdictFile
    from hmb.vocab import Verdict

    from .serialize import fact_statement, names_for_card
    dev = development_books(layout)
    pool = []
    vdir = layout.p("verdicts")
    for p in sorted(vdir.rglob("*.json")) if vdir.exists() else []:
        vf = VerdictFile.model_validate(read_json(p))
        cp = candidates_dir(layout) / f"{vf.item_id}.json"
        if vf.profile != th.profile.value or not cp.exists() or \
                (read_json(cp).get("source") or {}).get("book_id") not in dev:
            continue
        card = Card.model_validate(read_json(layout.card(vf.item_id)))
        facts, names = {f.fact_id: f for f in card.facts}, names_for_card(card)
        png = layout.compile_dir(vf.model_id, str(vf.track), vf.item_id) / "render.png"
        if not png.exists():
            continue
        for fv in vf.facts:
            if fv.source == "judges" and fv.verdict in (Verdict.PRESENT, Verdict.ABSENT) and fv.fact_id in facts:
                pool.append({"triple_id": f"{vf.model_id}|{vf.track}|{vf.item_id}|{fv.fact_id}",
                             "fact_text": fact_statement(facts[fv.fact_id], names),
                             "render_png": png.relative_to(layout.root).as_posix(),
                             "reference": fv.verdict == Verdict.PRESENT})
    n = int(th.value("drift_probe_n_triples"))
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))), key=lambda t: t["triple_id"])


def _cmd_drift_triples(args) -> int:
    from hmb.config import ThresholdsError, prefreeze_record, resolve_thresholds_arg
    from hmb.store import FreezeViolation, ImmutableConflict
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds_arg(args.thresholds, layout)
    except (ThresholdsError, FreezeViolation) as e:
        print(f"REFUSED: {e}")
        return 2
    if th.is_production and th.context != "DEV":
        print("REFUSED: the triple set is drawn in the pre-freeze DEV context (--thresholds prefreeze:DEV), before "
              "the level freeze")
        return 2
    triples = drift_triples(layout, th, seed=args.seed)
    if not triples:
        print("REFUSED: no judged model-decided fact on a development-book item (run the MS3 probe first)")
        return 2
    path = layout.p("manifests", "drift_probe", "triples.json")
    try:
        write_json_immutable(path, triples)
    except ImmutableConflict as e:
        print(f"REFUSED: {e}")
        return 2
    prefreeze_record(th, layout, "drift-triples", [path], inputs={"seed": args.seed})
    print(json.dumps({"triples": str(path), "n": len(triples), "sha256": sha256_file(path)}, indent=1))
    return 0


def _cmd_drift_probe(args) -> int:
    from hmb.roster import load_roster
    from hmb.grading.cli import ReportRefused, resolve_thresholds
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds(args.thresholds, layout)
    except ReportRefused as e:
        print(f"REFUSED: {e}")
        return 2
    judges = load_roster(args.roster)["judge"]
    if args.judge:
        judges = [j for j in judges if j.model_id in set(args.judge)]
    if not judges:
        print("REFUSED: no roster judges selected")
        return 2
    ledger = None
    if args.allow_live:
        from hmb.ledger import Ledger
        ledger = Ledger(layout)
    triples = Path(args.triples) if args.triples else layout.p("manifests", "drift_probe", "triples.json")
    try:
        out = drift_probe(layout, th, window=args.window, judges=judges, triples_path=triples,
                          allow_live=args.allow_live, phase=args.phase, ledger=ledger)
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 2
    print(json.dumps({k: v for k, v in out.items() if k not in ("answers", "per_judge")}, indent=1))
    return 0


def register_cli(sub) -> None:
    p = sub.add_parser("drift-triples", help="MS3: draw the frozen §10.9 triple set from judged development-set "
                                             "cells -> manifests/drift_probe/triples.json (immutable)")
    p.add_argument("--seed", type=int, required=True, help="sampling seed (recorded)")
    p.add_argument("--thresholds", default="prefreeze:DEV", help="prefreeze:DEV, or 'fixture' for engineering runs")
    p.set_defaults(func=_cmd_drift_triples)
    p = sub.add_parser("drift-probe", help="§10.9 judge drift probe at scoring-window open/close")
    p.add_argument("--window", choices=["open", "close"], required=True)
    p.add_argument("--roster", required=True)
    p.add_argument("--judge", action="append", default=None, help="restrict to these judge model ids")
    p.add_argument("--triples", default=None, help="default: manifests/drift_probe/triples.json (development-set cells, frozen with the levels)")
    p.add_argument("--thresholds", default=None,
                   help="thresholds.json path (default <root>/thresholds.json); 'fixture' only explicitly")
    p.add_argument("--phase", default="FINAL")
    p.add_argument("--allow-live", action="store_true")
    p.set_defaults(func=_cmd_drift_probe)
