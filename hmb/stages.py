"""The operator's path as stage commands, one per milestone (checkpoint simplification 2026-09-26).

Each stage runs an ordered list of the existing commands through `hmb.cli.main`, so every refusal,
freeze guard and exit code is exactly the one the single command gives; nothing is re-implemented.
A stage stops at the first step that fails and prints that step and how to resume. Every step is
resumable (finished work is skipped or content-addressed), and steps whose immutable output already
exists are skipped here, so rerunning a stage after fixing the cause continues where it stopped.

    hmb setup    --roster R [--approve --actor NAME]      MS1 form freeze: thresholds template, roster check,
                                                           form-freeze artifact; --approve = the owner's act
    hmb pilot    --roster R [--source S] [--provisional V] [--allow-live]
                                                           MS2 pilot (prefreeze:PILOT): development books ->
                                                           gates, G8b, development cards
    hmb develop  --roster R [--source S] [--provisional V] [--allow-live]
                                                           MS3 development set (prefreeze:DEV): development
                                                           books -> gates, G8b, cards, dev-measure
    hmb freeze   --provenance prov.json                    MS3 close: level freeze (consumes the DEV artifacts,
                                                           the dev-sizing report and the owner's delta*)
    hmb build    --roster R --seed N [--allow-live]       MS4 test set: approved test books, cards, items, release
    hmb run      --roster R --access-date D --provider-terms-version V [--allow-live]
                 [--batch-config C --policy window]        MS5: scoring window, every contestant, run manifests
                                                           (--batch-config: one hybrid batch-run step, hmb.batch)
    hmb report   --roster R                                MS5: headline, diagnostics, gates, §17.6 decision

The boundaries are the ones the protocol forces: the owner approves the forms after `setup`; the pilot
and the development set read provisional levels only inside a named pre-freeze context
(`--thresholds prefreeze:PILOT|DEV`, hmb.config) and only on development books; the level freeze consumes
their artifacts; test-book acquisition, test cards, releases and scores read frozen levels; the scoring
window must bracket every contestant run. Book census and rights clearance run in parallel
(`rights-check`); `approved_sources` lists the cleared books. The single commands remain for
troubleshooting (docs/RUNBOOK.md).
"""
from __future__ import annotations

import json

from .store import Layout, read_json, safe_model_dir

TRACKS = ("A", "B")


class StageStopped(RuntimeError):
    pass


def _run(root: str | None, stage: str, steps: list[tuple[str, list[str]]]) -> int:
    """Run (label, argv) steps in order through the CLI; stop at the first nonzero exit."""
    from .cli import main
    n = len(steps)
    for k, (label, argv) in enumerate(steps, 1):
        full = (["--root", root] if root else []) + argv
        print(f"[{stage} {k}/{n}] {label}: hmb {' '.join(argv)}", flush=True)
        code = main(full)
        if code:
            print(f"[{stage}] STOPPED at step {k}/{n} ({label}, exit {code}). Fix the cause shown above "
                  f"(docs/RUNBOOK.md §6), then rerun `hmb {stage} ...`: finished work is skipped.")
            return code
    print(f"[{stage}] done ({n} steps).")
    return 0


def _frozen(lay: Layout) -> bool:
    from .config import load_thresholds
    if not lay.thresholds.exists():
        return False
    th = load_thresholds(lay.thresholds)
    return th.is_production and bool(th.frozen_at) and not th.missing_for_production()


# ------------------------------------------------------------------ stage bodies
def setup_steps(lay: Layout, a) -> list[tuple[str, list[str]]]:
    steps = []
    if not lay.thresholds.exists():
        steps.append(("production thresholds template", ["thresholds-init", "--profile", "production"]))
    steps.append(("roster check", ["roster-check", a.roster]))
    if a.approve:
        steps.append(("OWNER approval of the current forms",
                      ["form-freeze", "--record-owner-approval", "--actor", a.actor or ""]))
    else:
        steps.append(("form-freeze artifact for owner review", ["form-freeze"]))
        steps.append(("parameters the level freeze needs", ["thresholds-status", str(lay.thresholds)]))
    return steps


def _prefreeze_steps(lay: Layout, a, context: str) -> list[tuple[str, list[str]]]:
    """MS2 (PILOT) and MS3 (DEV) before the level freeze, on development books, in that pre-freeze context."""
    if _frozen(lay):
        raise StageStopped("levels are frozen: the pre-freeze development stages are closed")
    ctx = ["--thresholds", f"prefreeze:{context}"]
    live = ["--allow-live"] if a.allow_live else []
    sources = a.source or approved_sources(lay)[0]
    if not sources:
        raise StageStopped("no development book: pass --source (an R3-approved book)")
    steps = [("pre-freeze values of calibrated parameters", ["thresholds-provisional", "--values", a.provisional])] \
        if a.provisional else []
    steps += [(f"CPU gate ladder {s} (development book)", ["acq-run", s, "--corpus", a.corpus, *ctx]) for s in sources]
    steps += [("G8b gatekeeper", ["acq-g8", "--roster", a.roster, "--corpus", a.corpus, *ctx, *live]),
              ("development cards", ["cards-build", "--roster", a.roster, *ctx, "--phase", context, *live])]
    if context == "DEV":
        steps.append(("development-set measurements", ["dev-measure", *ctx]))
    return steps


def pilot_steps(lay: Layout, a) -> list[tuple[str, list[str]]]:
    return _prefreeze_steps(lay, a, "PILOT")


def develop_steps(lay: Layout, a) -> list[tuple[str, list[str]]]:
    return _prefreeze_steps(lay, a, "DEV")


def freeze_steps(lay: Layout, a) -> list[tuple[str, list[str]]]:
    return [("level freeze", ["thresholds-freeze", "--provenance", a.provenance]),
            ("confirm nothing is missing", ["thresholds-status", str(lay.thresholds)])]


def approved_sources(lay: Layout) -> tuple[list[str], list[str]]:
    """Collector source names of the books whose rights record passes R3 (production), from the
    master evidence `rights-draft` wrote. Returns (sources, problems)."""
    from .rights import load_manifests, r3_gate
    from .vocab import Profile
    out, problems = [], []
    for book, rec in sorted(load_manifests(lay.rights_manifest).items()):
        if not r3_gate(rec, Profile.production)["passed"]:
            continue
        ev = lay.evidence_dir(book) / "master.json"
        if not ev.exists():
            problems.append(f"{book}: no sources/evidence/{book}/master.json (run rights-draft on its master)")
            continue
        out.append(read_json(ev)["file_name"])
    return out, problems


def build_steps(lay: Layout, a) -> list[tuple[str, list[str]]]:
    if not _frozen(lay):
        raise StageStopped("build needs the level freeze (production acquisition and card building read "
                           "frozen levels): run `hmb freeze` first")
    sources = a.source or approved_sources(lay)[0]
    if not sources:
        problems = approved_sources(lay)[1]
        raise StageStopped("no approved book: " + ("; ".join(problems) if problems else
                                                    "no rights record passes R3 (see `hmb rights-check`)"))
    th = str(lay.thresholds)
    live = ["--allow-live"] if a.allow_live else []
    steps = [(f"CPU gate ladder {s}", ["acq-run", s, "--corpus", a.corpus, "--thresholds", th]) for s in sources]
    steps += [("G8b gatekeeper", ["acq-g8", "--roster", a.roster, "--corpus", a.corpus, "--thresholds", th,
                                  "--dataset-version", a.dataset_version, *live]),
              ("card building", ["cards-build", "--roster", a.roster, "--thresholds", th, "--phase", "TEST", *live,
                                 *(["--batch-config", a.batch_config] if getattr(a, "batch_config", None) else [])]),
              ("items, splits, book cap", ["items-finalize", "--release-version", a.dataset_version,
                                           "--seed", str(a.seed), "--thresholds", th]),
              ("funnel report", ["funnel-report"])]
    if not lay.p("releases", a.version, "release.json").exists():
        steps.append(("release manifest", ["release-build", "--version", a.version, "--roster", a.roster]))
    return steps


def _contestants(roster: str) -> list:
    """Roster contestants that run on the direct-figure tracks. A text-only contestant (DeepSeek V3.1
    on the locked roster) is ineligible for the direct-figure track (A), and the protocol defines no
    scored text-input track, so its track execution is unavailable here; the §13.6 text-only probe
    (`t3-probe`) is the only text-input condition and is a diagnostic, not a track."""
    import sys

    from .roster import accepts_images, load_roster
    out = []
    for c in load_roster(roster).get("contestant", []):
        if accepts_images(c):
            out.append(c)
        else:
            print(f"note: {c.model_id} is text-only: direct-figure track unavailable, no scored text-input "
                  "track exists (its locked roster entry is kept; `t3-probe` is the diagnostic)", file=sys.stderr)
    return out


def run_steps(lay: Layout, a) -> list[tuple[str, list[str]]]:
    th = ["--thresholds", str(lay.thresholds)]
    live = ["--allow-live"] if a.allow_live else []
    tracks = [x for t in TRACKS for x in ("--track", t)]
    probe = lay.p("manifests", "drift_probe")
    steps = []
    closed = (probe / "close.json").exists()
    if not (probe / "open.json").exists():
        steps.append(("open the scoring window", ["drift-probe", "--window", "open", "--roster", a.roster, *th, *live]))
    contestants = _contestants(a.roster)
    if not contestants:
        raise StageStopped("the roster lists no contestant")
    if not closed:
        if getattr(a, "batch_config", None):            # hybrid batch/on-demand scheduler (hmb.batch, 2026-09-26)
            steps.append(("every contestant cell (hybrid scheduler)",
                          ["batch-run", "--roster", a.roster, "--batch-config", a.batch_config, "--policy", a.policy,
                           *tracks, *th, "--access-date", a.access_date,
                           "--provider-terms-version", a.provider_terms_version, *live]))
        else:
            for c in contestants:
                steps.append((f"model run {c.model_id}",
                              ["model-run", "--roster", a.roster, "--model-id", c.model_id, *tracks, *th,
                               "--access-date", a.access_date, "--provider-terms-version", a.provider_terms_version,
                               *live]))
        steps.append(("close the scoring window", ["drift-probe", "--window", "close", "--roster", a.roster, *th,
                                                   *live]))
    for c in contestants:
        if not lay.run_manifest(c.model_id).exists():
            steps.append((f"run manifest {c.model_id}", ["model-close", "--roster", a.roster, "--model-id", c.model_id,
                                                         *tracks, *th]))
    return steps


def report_steps(lay: Layout, a) -> list[tuple[str, list[str]]]:
    from .grading.cli import release_id
    models = [safe_model_dir(c.model_id) for c in _contestants(a.roster)]
    if not models:
        raise StageStopped("the roster lists no contestant")
    tracks = [x for t in TRACKS for x in ("--track", t)]
    scores = lay.reports / f"scores_test_{release_id(lay)[0]}.json"
    steps = []
    if not scores.exists():
        steps.append(("headline, diagnostics, gate evidence", ["score-report", "--models", *models, *tracks]))
    steps.append(("gates and the §17.6 decision", ["gates-evaluate", "--scores", str(scores)]))
    return steps


STAGES = {"setup": setup_steps, "pilot": pilot_steps, "develop": develop_steps,
          "freeze": freeze_steps, "build": build_steps, "run": run_steps, "report": report_steps}


def _cmd(stage: str):
    def run(args) -> int:
        lay = Layout.from_env(args.root)
        try:
            steps = STAGES[stage](lay, args)
        except StageStopped as e:
            print(f"[{stage}] STOPPED before any step: {e}")
            return 2
        if getattr(args, "dry_run", False):
            print(json.dumps([{"step": lbl, "command": "hmb " + " ".join(argv)} for lbl, argv in steps], indent=1))
            return 0
        return _run(args.root, stage, steps)
    return run


def register_cli(sub) -> None:
    def add(name, hlp):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--dry-run", action="store_true", help="list the steps without running them")
        p.set_defaults(func=_cmd(name))
        return p
    p = add("setup", "MS1 form freeze: thresholds template, roster check, form-freeze artifact; "
                     "--approve records the owner's approval")
    p.add_argument("--roster", required=True)
    p.add_argument("--approve", action="store_true", help="OWNER act: approve the current form-freeze artifact")
    p.add_argument("--actor", default=None, help="owner name (with --approve)")
    for name, hlp in (("pilot", "MS2 pilot (prefreeze:PILOT): development books -> gates, G8b, development cards"),
                      ("develop", "MS3 development set (prefreeze:DEV): development books -> gates, G8b, cards, "
                                  "dev-measure")):
        p = add(name, hlp)
        p.add_argument("--roster", required=True)
        p.add_argument("--corpus", default="corpus")
        p.add_argument("--source", action="append", default=None, help="development books (collector sources)")
        p.add_argument("--provisional", default=None,
                       help="thresholds-provisional --values file (calibrated parameters)")
        p.add_argument("--allow-live", action="store_true")
    p = add("freeze", "MS3 close: level freeze from the development artifacts, sizing and delta*")
    p.add_argument("--provenance", required=True)
    p = add("build", "MS4 test set: approved test books -> gates, G8b, cards, items, funnel, release")
    p.add_argument("--roster", required=True)
    p.add_argument("--seed", type=int, required=True, help="split-assignment seed (owner choice, recorded)")
    p.add_argument("--version", default="1.0.0", help="benchmark release version")
    p.add_argument("--dataset-version", default="v1.0")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--source", action="append", default=None, help="restrict to these collector sources")
    p.add_argument("--allow-live", action="store_true")
    p.add_argument("--batch-config", default=None, help="card building under the hybrid batch router (batch-capable "
                                                         "writers answered by batch jobs)")
    p = add("run", "MS5: scoring window, every roster contestant on Tracks A and B, run manifests")
    p.add_argument("--roster", required=True)
    p.add_argument("--access-date", required=True)
    p.add_argument("--provider-terms-version", required=True)
    p.add_argument("--allow-live", action="store_true")
    p.add_argument("--batch-config", default=None, help="run every cell through the hybrid batch/on-demand scheduler "
                                                         "(batch-run) instead of one model-run per contestant")
    p.add_argument("--policy", choices=["sync", "fixed", "window"], default="window",
                   help="lane flush policy with --batch-config (default window)")
    p = add("report", "MS5: score report, gates and the §17.6 decision")
    p.add_argument("--roster", required=True)


__all__ = ["STAGES", "approved_sources", "StageStopped"]
