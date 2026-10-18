"""CLI hooks for hmb.cards: `cards-prompts` (frozen prompt hashes) and `cards-oracle-card`."""
from __future__ import annotations

import json
from pathlib import Path


def _cmd_prompts(args) -> int:
    from .prompts import prompt_hashes
    print(json.dumps(prompt_hashes(), indent=1))
    return 0


def _cmd_oracle_card(args) -> int:
    from hmb.store import write_json_immutable
    from .oracle import oracle_card
    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    card = oracle_card(graph, profile=args.profile)
    sha = write_json_immutable(Path(args.out), card.model_dump(mode="json", by_alias=True))
    print(f"wrote {args.out} card_uid={card.card_uid} facts={len(card.facts)} sha256={sha}")
    return 0


def _cmd_cards_build(args) -> int:
    from hmb.config import PREFREEZE_CONTEXTS, ThresholdsError, prefreeze_record, resolve_thresholds_arg
    from hmb.ledger import Ledger
    from hmb.providers import LiveCallsDisabled
    from hmb.roster import load_roster
    from hmb.store import FreezeViolation, Layout
    from .batch import candidates_dir, card_gates_path, cards_build
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds_arg(args.thresholds, layout)
    except (ThresholdsError, FreezeViolation) as e:
        print(f"REFUSED: {e}")
        return 2
    only = None
    if th.is_prefreeze:           # PILOT/DEV development cards only; test (MS4) cards need the level freeze
        from hmb.acquisition.drivers import development_books
        if PREFREEZE_CONTEXTS[th.context or ""]["partition"] != "development" or args.phase != th.context:
            print(f"REFUSED: in the pre-freeze {th.context} context cards-build runs only as --phase {th.context} "
                  "on development books (PILOT/DEV); TEST cards need the level freeze")
            return 2
        only = development_books(layout)
    elif args.phase in ("PILOT", "DEV") and th.is_production and not th.frozen_at:
        print("REFUSED: unfrozen production levels: run PILOT/DEV cards with --thresholds prefreeze:<PILOT|DEV>")
        return 2
    roster = load_roster(args.roster)
    ledger = Ledger(layout)
    from .batch import card_selection_excluded, card_selection_locked
    skip = set(card_selection_excluded(layout)) if only is None else set()
    if only is None and args.phase == "TEST":
        try:
            locked = card_selection_locked(layout)
        except ValueError as e:
            print(f"REFUSED: {e}")
            return 2
        if locked is not None:              # `card-lock`: card exactly the locked test candidates, nothing else
            ids = {p.stem for p in candidates_dir(layout).glob("*.json")}
            skip = ids - locked
            print(f"card-lock: {len(locked)} locked test candidates; {len(skip)} other candidates not carded")

    def build():
        return cards_build(layout, th, builders=roster.get("builder", []), judges=roster.get("judge", []),
                           allow_live=args.allow_live, phase=args.phase, ledger=ledger,
                           max_tokens=args.max_tokens, only_books=only, skip_items=skip,
                           workers=args.workers)
    try:
        if getattr(args, "batch_config", None):
            s = _hybrid(layout, ledger, args, build)
            if s is None:
                return 4                     # stopped early; batch jobs still running: rerun to resume
        else:
            s = build()
    except LiveCallsDisabled as e:
        print(str(e))
        return 3
    built = s["built"] + s["gates_only"]
    prefreeze_record(th, layout, f"cards-build --phase {args.phase}",
                     [p for i in built for p in (layout.card(i), card_gates_path(layout, i))],
                     inputs={"candidates": sorted(i for i in built if (candidates_dir(layout) / f"{i}.json").exists())})
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in s.items()}, indent=1))
    if not s["disjointness_checked"]:
        print("WARNING: roster lists no judges; builder/judge family disjointness (§10.9) NOT checked")
    for e in s["errors"]:
        print(f"ERROR {e['item_id']}: {e['error']}")
    return 1 if s["errors"] else 0


def _hybrid(layout, ledger, args, build):
    """Run `build` (the whole cards-build pass) under the hybrid batch router (hmb.batch, 2026-09-26): a writer
    whose route is batch-capable (Kimi K2.5) is answered by batch jobs, the other writer synchronously; the pass
    is replayed from the cache until every item is built. Returns None when --max-wall-seconds stopped it."""
    from hmb.batch.cli import make_router
    from hmb.batch.jobs import BedrockBatchBackend, load_config
    from hmb.batch.scheduler import HybridScheduler
    from hmb.providers import set_router
    from .batch import candidates_dir
    cfg = load_config(args.batch_config)
    n = sum(1 for _ in candidates_dir(layout).glob("*.json")) if candidates_dir(layout).exists() else 0
    router = make_router(layout, cfg, args.policy, ledger=ledger, lane_size=n or None)
    set_router(router)
    try:
        out = HybridScheduler(router, BedrockBatchBackend(cfg), workers=1, poll_seconds=cfg.poll_seconds,
                              max_wall_seconds=args.max_wall_seconds).run({"cards-build": build})
    finally:
        set_router(None)
    res = out["results"].get("cards-build")
    if res is None:
        print(f"STOPPED EARLY: batch jobs still running ({out['parked']}); rerun the same command to resume")
        return None
    if isinstance(res, Exception):
        raise res
    return res


def _selection_inputs(layout, th):
    """(candidates [{item_id, book_id, group, phash_cluster_id, proposal_id}], development groups) as card-select reads them."""
    from hmb.acquisition.drivers import development_books
    from hmb.rights import load_manifests, r3_gate
    from hmb.sampling import split_graph
    from hmb.store import read_json
    from hmb.vocab import Profile
    from .batch import candidates_dir
    mans = [m for m in load_manifests(layout.rights_manifest).values() if r3_gate(m, Profile.production)["passed"]]
    graph = split_graph(mans, [], th)
    dev_groups = {graph["book_group"][b] for b in development_books(layout) if b in graph["book_group"]}
    cands = []
    for p in sorted(candidates_dir(layout).glob("*.json")):
        c = read_json(p)
        book = c["source"]["book_id"]
        cands.append({"item_id": c.get("id") or p.stem, "book_id": book, "group": graph["book_group"].get(book),
                      "phash_cluster_id": (c.get("quality") or {}).get("phash_cluster_id"),
                      "proposal_id": c["extraction"]["proposal_id"]})
    return cands, dev_groups


def _resolve_frozen(args, cmd):
    from hmb.config import ThresholdsError, resolve_thresholds_arg
    from hmb.store import FreezeViolation, Layout
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds_arg(args.thresholds, layout)
    except (ThresholdsError, FreezeViolation) as e:
        print(f"REFUSED: {e}")
        return layout, None
    if th.is_prefreeze:
        print(f"REFUSED: {cmd} concerns test candidates; it does not run in a pre-freeze context")
        return layout, None
    return layout, th


def _cmd_card_select(args) -> int:
    """Owner-set pre-card selection (hmb.sampling.select_for_cards) -> manifests/card_selection.json, plus an
    immutable copy manifests/card_selection/<sha12>.json and funnel rows for every candidate whose status changed."""
    from datetime import datetime, timezone

    from hmb.acquisition.funnel import append_row, latest_rows
    from hmb.hashing import sha256_json
    from hmb.sampling import select_for_cards
    from hmb.store import write_json_immutable, write_mutable
    from hmb.vocab import Exclusion
    from .batch import card_selection_lock_path, card_selection_path
    layout, th = _resolve_frozen(args, "card-select")
    if th is None:
        return 2
    if card_selection_lock_path(layout).exists():
        print(f"REFUSED: the selection is locked ({card_selection_lock_path(layout)}); a new selection needs the "
              "owner to retire that lock by hand")
        return 2
    cands, dev_groups = _selection_inputs(layout, th)
    unmapped = sorted(c["item_id"] for c in cands if c["group"] is None)
    if unmapped:
        print(f"REFUSED: {len(unmapped)} candidates belong to no R3-approved book (e.g. {unmapped[:3]})")
        return 2
    sel = select_for_cards(cands, dev_groups, args.budget, args.seed)
    doc = {"artifact": "card_selection", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "authority": args.authority, "n_candidates": len(cands),
           "n_test_selected": sum(v["selected"] for v in sel["per_group"].values() if not v["development"]),
           **sel}
    doc["sha256"] = sha256_json(doc)
    if args.dry_run:
        print(json.dumps({k: v for k, v in doc.items() if k not in ("selected", "excluded", "per_group")}, indent=1))
        return 0
    write_json_immutable(layout.p("manifests", "card_selection", f"{doc['sha256'][:12]}.json"), doc)
    write_mutable(card_selection_path(layout), (json.dumps(doc, indent=1) + "\n").encode("utf-8"))
    rows, changed = latest_rows(layout), 0
    selection_reasons = {Exclusion.precard_duplicate.value, Exclusion.selection_budget.value}
    for c in cands:
        prev = rows.get(c["proposal_id"])
        if prev is None:
            continue
        reason = sel["excluded"].get(c["item_id"])
        if reason:
            want = dict(prev, outcome="excluded", exclusion_reasons=[reason], item_id=c["item_id"],
                        is_cluster_representative=reason != Exclusion.precard_duplicate.value)
        elif set(prev.get("exclusion_reasons") or []) & selection_reasons:     # re-selected by a later selection
            want = dict(prev, outcome=None, exclusion_reasons=[], is_cluster_representative=None)
        else:
            continue
        if want != prev:
            append_row(layout, want)
            changed += 1
    print(json.dumps({"artifact": str(card_selection_path(layout)), "sha256": doc["sha256"],
                      "quota": sel["quota"], "n_test_selected": doc["n_test_selected"],
                      "excluded": {r: sum(1 for v in sel["excluded"].values() if v == r) for r in sorted(selection_reasons)},
                      "funnel_rows_appended": changed}, indent=1))
    return 0


def _cmd_card_lock(args) -> int:
    """Owner lock of the current selection -> manifests/card_selection_lock.json (immutable). Checks the selection's
    hash and immutable copy and re-derives it from the candidates with its recorded budget and seed. From then on
    `cards-build --phase TEST` cards exactly the locked test candidates and `card-select` refuses."""
    from datetime import datetime, timezone

    from hmb.hashing import sha256_json
    from hmb.sampling import select_for_cards
    from hmb.store import read_json, write_json_immutable
    from .batch import card_selection_lock_path, card_selection_path
    layout, th = _resolve_frozen(args, "card-lock")
    if th is None:
        return 2
    if card_selection_lock_path(layout).exists():
        print(f"REFUSED: already locked ({card_selection_lock_path(layout)})")
        return 2
    sp = card_selection_path(layout)
    if not sp.exists():
        print("REFUSED: no card_selection.json (run card-select first)")
        return 2
    doc = read_json(sp)
    problems = []
    if sha256_json({k: v for k, v in doc.items() if k != "sha256"}) != doc["sha256"]:
        problems.append("card_selection.json does not match its recorded sha256")
    imm = layout.p("manifests", "card_selection", f"{doc['sha256'][:12]}.json")
    if not imm.exists() or read_json(imm) != doc:
        problems.append(f"immutable copy {imm} missing or different")
    cands, dev_groups = _selection_inputs(layout, th)
    again = select_for_cards(cands, dev_groups, doc["budget"], doc["seed"])
    if again["selected"] != doc["selected"] or again["excluded"] != doc["excluded"]:
        problems.append("the candidates no longer re-derive this selection (candidates or split groups changed)")
    dev_ids = {c["item_id"] for c in cands if c["group"] in dev_groups}
    test_ids = sorted(set(doc["selected"]) - dev_ids)
    if len(test_ids) != doc["n_test_selected"]:
        problems.append(f"{len(test_ids)} test candidates selected, the selection records {doc['n_test_selected']}")
    if problems:
        print("REFUSED: " + "; ".join(problems))
        return 2
    lock = {"artifact": "card_selection_lock", "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "authority": args.authority, "selection_sha256": doc["sha256"], "budget": doc["budget"], "seed": doc["seed"],
            "rule": doc["rule"], "n_test_locked": len(test_ids),
            "n_development_group_outside_lock": len(set(doc["selected"]) & dev_ids), "test_item_ids": test_ids}
    lock["sha256"] = sha256_json(lock)
    if args.dry_run:
        print(json.dumps({k: v for k, v in lock.items() if k != "test_item_ids"}, indent=1))
        return 0
    write_json_immutable(card_selection_lock_path(layout), lock)
    print(json.dumps({"artifact": str(card_selection_lock_path(layout)),
                      **{k: v for k, v in lock.items() if k != "test_item_ids"}}, indent=1))
    return 0


def _cmd_dev_measure(args) -> int:
    """MS3 development-set measurements -> reports/dev/dev_measurements_<sha12>.json (immutable)."""
    import hashlib
    from datetime import datetime, timezone

    from hmb.config import ThresholdsError, execution_context, prefreeze_record, resolve_thresholds_arg
    from hmb.store import FreezeViolation, Layout, write_json_immutable
    from .batch import dev_measurements
    layout = Layout.from_env(args.root)
    try:
        th = resolve_thresholds_arg(args.thresholds, layout)
    except (ThresholdsError, FreezeViolation) as e:
        print(f"REFUSED: {e}")
        return 2
    if th.is_production and th.context != "DEV":
        print("REFUSED: dev-measure runs in the pre-freeze DEV context (--thresholds prefreeze:DEV) or on a fixture")
        return 2
    m = dev_measurements(layout, th)
    inputs = m.pop("inputs")
    doc = {"artifact": "development_measurements", "version": "dev-measure-2",
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "profile": th.profile.value,
           "execution_context": execution_context(th, "dev-measure", {"n_inputs": {k: len(v) for k, v in inputs.items()}}),
           "inputs": inputs, "measurements": m}
    tag = hashlib.sha256(json.dumps(doc["inputs"], sort_keys=True).encode()).hexdigest()[:12]
    out = layout.reports / "dev" / f"dev_measurements_{tag}.json"
    write_json_immutable(out, doc)
    prefreeze_record(th, layout, "dev-measure", [out], inputs={"n_carded": m["n_carded"]})
    print(json.dumps({"artifact": str(out), **{k: m[k] for k in ("n_candidates", "n_carded", "n_retained",
                                                                "retention_r_hat", "F_bar")}}, indent=1))
    return 0


def register_cli(sub) -> None:
    p = sub.add_parser("cards-build", help="build cards + card gates for every data/candidates/<item_id>.json")
    p.add_argument("--thresholds", required=True, help="thresholds.json (fixture or production), or prefreeze:PILOT|DEV "
                                                        "for development cards before the level freeze")
    p.add_argument("--roster", required=True, help="roster YAML (hmb.roster.load_roster)")
    p.add_argument("--phase", choices=["PILOT", "DEV", "TEST"], required=True,
                   help="ledger phase: PILOT (MS2), DEV (MS3), TEST (MS4)")
    p.add_argument("--allow-live", action="store_true", help="permit paid calls on cache misses")
    p.add_argument("--max-tokens", type=int, default=None, help="test override of card_builder_max_tokens")
    p.add_argument("--batch-config", default=None, help="run under the hybrid batch router (hmb.batch): batch-capable "
                                                        "writers (Kimi K2.5) are answered by batch jobs; resumable")
    p.add_argument("--policy", choices=["sync", "fixed", "window"], default="window")
    p.add_argument("--workers", type=int, default=1, help="items carded concurrently (default 1: sequential)")
    p.add_argument("--max-wall-seconds", type=float, default=None,
                   help="with --batch-config: stop after this long (exit 4; rerun to resume)")
    p.set_defaults(func=_cmd_cards_build)
    p = sub.add_parser("card-select", help="OWNER-set pre-card selection of test candidates: one scan per work, one per "
                                           "pHash cluster, seeded per-group quota under --budget (cards-build honours it)")
    p.add_argument("--thresholds", required=True, help="the level-frozen thresholds.json")
    p.add_argument("--budget", type=int, required=True, help="maximum test candidates to card (owner choice, recorded)")
    p.add_argument("--seed", type=int, required=True, help="selection seed (owner choice, recorded)")
    p.add_argument("--authority", required=True, help="who decided the budget and when (recorded)")
    p.add_argument("--dry-run", action="store_true", help="print the selection summary; write nothing")
    p.set_defaults(func=_cmd_card_select)
    p = sub.add_parser("card-lock", help="OWNER lock of the current card selection: cards-build --phase TEST then cards "
                                         "exactly its test candidates, and card-select refuses")
    p.add_argument("--thresholds", required=True, help="the level-frozen thresholds.json")
    p.add_argument("--authority", required=True, help="who locked it and when (recorded)")
    p.add_argument("--dry-run", action="store_true", help="check and print the lock; write nothing")
    p.set_defaults(func=_cmd_card_lock)
    p = sub.add_parser("dev-measure", help="MS3 development-set measurements (r̂, F̄, contradiction, family "
                                           "agreement) from stored cards; no model calls")
    p.add_argument("--thresholds", required=True, help="prefreeze:DEV (or 'fixture' for engineering runs)")
    p.set_defaults(func=_cmd_dev_measure)
    p = sub.add_parser("cards-prompts", help="print sha256 of the frozen builder prompts (§7.3/§8.8)")
    p.set_defaults(func=_cmd_prompts)
    p = sub.add_parser("cards-oracle-card", help="known-geometry oracle graph JSON -> Card JSON (all facts eligible; engineering fixtures)")
    p.add_argument("graph")
    p.add_argument("out")
    p.add_argument("--profile", choices=["fixture", "production"], default="production")
    p.set_defaults(func=_cmd_oracle_card)
