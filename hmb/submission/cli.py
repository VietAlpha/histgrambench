"""CLI: `python -m hmb [--root R] sandbox-build | sandbox-selftest | sandbox-compile <file.tex> |
font-coverage-gate --cards <dir> | determinism-rerun [--sample] | sandbox-cputime-check`."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path


def _th(args):
    from hmb.config import fixture_thresholds, load_thresholds
    return load_thresholds(args.thresholds) if args.thresholds else fixture_thresholds()


def _layout(args):
    from hmb.store import Layout
    return Layout.from_env(getattr(args, "root", None))


def _build(args) -> int:
    from .sandbox import build_image
    lock = build_image(log_path=Path(args.log) if args.log else None)
    print(json.dumps({"image_id": lock["image_id"], "manifest": lock["manifest"]}, indent=1))
    return 0


def _selftest(args) -> int:
    from .sandbox import image_selftest
    r = image_selftest(_th(args))
    print(r["output"])
    return 0 if r["pass"] else 1


def _compile(args) -> int:
    from .sandbox import compile_in_sandbox
    res = compile_in_sandbox(Path(args.tex).read_bytes(), _th(args), Path(args.out))
    print(json.dumps(res.to_json(), indent=1, ensure_ascii=False))
    return 0


def font_coverage_gate_cards(cards_dir: Path, th, out_dir: Path) -> dict:
    """§9.6 one-time gate over every card JSON in `cards_dir` (the eligible cards)."""
    from hmb.schemas import Card
    from hmb.store import read_json

    from .sandbox import font_coverage_gate, labels_from_cards
    cards = [Card.model_validate(read_json(p)) for p in sorted(Path(cards_dir).glob("*.json"))]
    rep = font_coverage_gate(labels_from_cards(cards), th, out_dir)
    rep["n_cards"] = len(cards)
    return rep


def _font_gate(args) -> int:
    lay = _layout(args)
    out = Path(args.out) if args.out else lay.reports / "font_coverage_gate"
    rep = font_coverage_gate_cards(Path(args.cards), _th(args), out)
    print(json.dumps({k: rep[k] for k in ("n_cards", "n_labels", "pass", "missing")}, ensure_ascii=False, indent=1))
    return 0 if rep["pass"] else 1


def determinism_rerun_stored(layout, th, *, sample: bool, seed: str = "hmb-determinism-v1") -> dict:
    """HarnessDeterminism@rerun (§9.6) over stored OK submissions: recompile + re-render each selected
    normalized.tex twice in the pinned image; a submission passes when both reruns reproduce the
    stored render_sha256 exactly. `sample=True` uses the fixed seeded 5% sample."""
    from hmb.schemas import SubmissionRecord
    from hmb.store import read_json

    from .sandbox import determinism_sample, harness_determinism_rerun
    stages = {}
    for p in sorted(layout.p("compiled").glob("*/*/*/submission_stage.json")):
        rec = SubmissionRecord.model_validate(read_json(p))
        if rec.outcome_code == "OK" and rec.normalized_tex:
            stages[f"{p.parts[-4]}/{rec.track}/{rec.item_id}"] = rec
    keys = determinism_sample(stages, th, seed) if sample else sorted(stages)
    rows = []
    with tempfile.TemporaryDirectory(prefix="hmb_det_") as tmp:
        for k in keys:
            rec = stages[k]
            tex = (layout.root / rec.normalized_tex).read_bytes()
            r = harness_determinism_rerun(tex, th, Path(tmp) / k.replace("/", "__"))
            ok = r["identical_render"] and r["render_sha256"][0] == rec.render_sha256
            rows.append({"submission": k, "stored_render_sha256": rec.render_sha256, **r, "pass": ok})
    n = len(rows)
    rep = {"n_ok_submissions": len(stages), "n_rerun": n, "sampled": sample, "seed": seed if sample else None,
           "sample_frac": th.value("determinism_sample_frac") if sample else 1.0,
           "HarnessDeterminism@rerun": (sum(r["pass"] for r in rows) / n) if n else None, "rows": rows}
    return rep


def _det(args) -> int:
    from hmb.store import write_mutable
    lay = _layout(args)
    rep = determinism_rerun_stored(lay, _th(args), sample=args.sample, seed=args.seed)
    out = lay.reports / "harness_determinism.json"
    write_mutable(out, (json.dumps(rep, indent=1) + "\n").encode())
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    print(f"report: {out}")
    return 0 if rep["HarnessDeterminism@rerun"] in (None, 1.0) else 1


def cputime_check_stored(layout, th, *, n: int, concurrency: int, reps: int, seed: str = "hmb-cputime-v1",
                         runner=None) -> dict:
    """C24 check (owner 2026-10-01, before lifting the serial compile lock): compile the same stored
    normalized.tex set serially and then `concurrency` at a time, `reps` rounds each, under the production
    container flags, and compare CPU seconds and outcomes per document. The sample is the `n // 2` stored
    submissions with the highest recorded CPU seconds (closest to `cpu_seconds`) plus a seeded draw of the
    rest. Bypasses `_COMPILE_LOCK` by calling the runner directly; writes nothing under `layout`."""
    import statistics
    import time
    from concurrent.futures import ThreadPoolExecutor

    from hmb.hashing import sha256_bytes
    from hmb.schemas import SubmissionRecord
    from hmb.store import read_json

    from .sandbox import classify, ensure_image, run_container
    run = runner or run_container
    if runner is None:
        ensure_image()
    stages = {}
    for p in sorted(layout.p("compiled").glob("*/*/*/submission_stage.json")):
        rec = SubmissionRecord.model_validate(read_json(p))
        if rec.normalized_tex and rec.cpu_seconds is not None:
            stages[f"{p.parts[-4]}/{rec.track}/{rec.item_id}"] = rec
    by_cpu = sorted(stages, key=lambda k: (-stages[k].cpu_seconds, k))
    top = by_cpu[:n // 2]
    rest = sorted((k for k in stages if k not in set(top)), key=lambda k: sha256_bytes(f"{seed}:{k}".encode()))
    keys = sorted(top + rest[:n - len(top)])
    texs = {k: (layout.root / stages[k].normalized_tex).read_bytes() for k in keys}

    def one(k: str, tmp: Path) -> tuple[float | None, str]:
        art = run(texs[k], th)
        res = classify(art, th, tmp / f"{k.replace('/', '__')}_{time.monotonic_ns()}")
        return res.cpu_seconds, res.outcome.value

    obs = {k: {"serial": [], "parallel": []} for k in keys}
    walls = {"serial": [], "parallel": []}
    with tempfile.TemporaryDirectory(prefix="hmb_cputime_") as td:
        tmp = Path(td)
        for _ in range(reps):
            t0 = time.monotonic()
            for k in keys:
                obs[k]["serial"].append(one(k, tmp))
            walls["serial"].append(time.monotonic() - t0)
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                for k, r in zip(keys, ex.map(lambda k: one(k, tmp), keys)):
                    obs[k]["parallel"].append(r)
            walls["parallel"].append(time.monotonic() - t0)

    cpu_limit = float(th.value("cpu_seconds"))
    rows, ratios = [], []
    for k in keys:
        s = [c for c, _ in obs[k]["serial"] if c is not None]
        q = [c for c, _ in obs[k]["parallel"] if c is not None]
        outcomes = {m: sorted({o for _, o in obs[k][m]}) for m in ("serial", "parallel")}
        ms, mq = (statistics.median(s) if s else None), (statistics.median(q) if q else None)
        ratio = (mq / ms) if ms and mq is not None else None
        if ratio is not None:
            ratios.append(ratio)
        rows.append({"submission": k, "stored_cpu_seconds": stages[k].cpu_seconds,
                     "stored_outcome": stages[k].outcome_code, "serial_cpu": s, "parallel_cpu": q,
                     "serial_median": ms, "parallel_median": mq, "ratio": ratio, "outcomes": outcomes,
                     "outcome_changed": outcomes["serial"] != outcomes["parallel"] or len(outcomes["serial"]) > 1})
    q_sorted = sorted(ratios)
    return {"n_stored": len(stages), "n": len(keys), "n_top_cpu": len(top), "concurrency": concurrency,
            "reps": reps, "seed": seed, "cpu_seconds_limit": cpu_limit,
            "ratio_median": statistics.median(q_sorted) if q_sorted else None,
            "ratio_p95": q_sorted[min(len(q_sorted) - 1, math.ceil(0.95 * len(q_sorted)) - 1)] if q_sorted else None,
            "ratio_max": q_sorted[-1] if q_sorted else None,
            "outcome_changes": sum(r["outcome_changed"] for r in rows),
            "wall_seconds": walls,
            "throughput_speedup": (sum(walls["serial"]) / sum(walls["parallel"])) if sum(walls["parallel"]) else None,
            "rows": rows}


def _cputime(args) -> int:
    from hmb.store import write_mutable
    lay = _layout(args)
    rep = cputime_check_stored(lay, _th(args), n=args.n, concurrency=args.concurrency, reps=args.reps)
    out = Path(args.out) if args.out else lay.reports / "sandbox_cputime_check.json"
    write_mutable(out, (json.dumps(rep, indent=1) + "\n").encode())
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    print(f"report: {out}")
    return 0 if rep["outcome_changes"] == 0 else 1


def register_cli(sub) -> None:
    p = sub.add_parser("sandbox-build", help="build the §9.6 sandbox image and write sandbox/image.lock.json")
    p.add_argument("--log", default=None)
    p.set_defaults(func=_build)
    for name, fn, h in (("sandbox-selftest", _selftest, "run the image self-test under production flags"),
                        ("sandbox-compile", _compile, "compile one .tex in the sandbox (engineering)"),
                        ("font-coverage-gate", _font_gate, "§9.6 font-coverage gate over a card directory"),
                        ("determinism-rerun", _det, "§9.6 HarnessDeterminism@rerun over stored submissions"),
                        ("sandbox-cputime-check", _cputime,
                         "C24: CPU seconds of stored submissions, serial vs N concurrent compiles")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--thresholds", default=None, help="thresholds file (default: fixture profile)")
        if name == "sandbox-compile":
            p.add_argument("tex")
            p.add_argument("--out", required=True)
        if name == "font-coverage-gate":
            p.add_argument("--cards", required=True, help="directory of eligible card JSON files")
            p.add_argument("--out", default=None)
        if name == "determinism-rerun":
            p.add_argument("--sample", action="store_true", help="fixed seeded 5%% sample (§9.6)")
            p.add_argument("--seed", default="hmb-determinism-v1")
        if name == "sandbox-cputime-check":
            p.add_argument("--n", type=int, default=60, help="documents (half the highest stored CPU, half seeded)")
            p.add_argument("--concurrency", type=int, default=6, help="concurrent compiles in the parallel pass")
            p.add_argument("--reps", type=int, default=3, help="serial and parallel rounds per document")
            p.add_argument("--out", default=None, help="report path (default: <root>/reports/)")
        p.set_defaults(func=fn)
