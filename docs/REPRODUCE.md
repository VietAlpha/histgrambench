# Reproducing HistGramBench

Three levels. Levels 1 and 2 are offline and make no model call; level 3 is a paid procedure that was not run for
this release. Commands are shown for a POSIX shell; on Windows use `.venv\Scripts\python.exe` and set environment
variables with `$env:HMB_ROOT = "..."` (PowerShell). The expected results below were obtained when this release was
prepared.

## 0. Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.lock
uv venv --python 3.12 .venv-release && uv pip install --python .venv-release -r scripts/requirements-release.txt
.venv/bin/python scripts/check_hashes.py          # code_hash, form_freeze_sha256, harness_version, release_hash
hf auth login                                     # the dataset is gated: request access first
hf download VietAlphaLabs/HistGramBench --repo-type dataset --local-dir ../HistGramBench
.venv-release/bin/python scripts/verify_release.py --dataset ../HistGramBench      # integrity, joins, splits, mirrors
.venv-release/bin/python scripts/materialize_root.py --dataset ../HistGramBench --out ../hgb_root --without-stored-reports
export HMB_ROOT=../hgb_root
```

`check_hashes.py` reports `harness_version` as different when your `sandbox/image.lock.json` describes a rebuilt
image; the other three must match. `--without-stored-reports` leaves the stored score report out of the root so that
level 1 regenerates it; the stored copy stays in `frozen/reports/` for comparison.

## Level 1: tables and figures from stored scores (minutes)

```bash
# in the order of the stored report's `models` list (the report records the order it was given)
M="us.meta.llama4-maverick-17b-instruct-v1_0 nvidia.nemotron-nano-12b-v2 google.gemma-4-26b-a4b \
   mistral.ministral-3-14b-instruct google.gemma-4-e2b global.anthropic.claude-haiku-4-5-20251001-v1_0"
# 1. score report (headline, intervals, bands, validity, strata, page-text effect, leave-one-family-out)
.venv/bin/python -m hmb --root $HMB_ROOT score-report --models $M --track A --track B --split test
.venv/bin/python scripts/compare_outputs.py frozen/reports/scores_test_histmathbench-1.0.0.json \
    $HMB_ROOT/reports/scores_test_histmathbench-1.0.0.json
# 2. analysis numbers (variants, strata, page-text subsets, loss decomposition, H_valid, key retention)
.venv/bin/python research-paper/analysis/compute_revision_numbers.py
.venv/bin/python research-paper/analysis/compute_outline_numbers.py
for f in research-paper/review_20261003/*.py; do .venv/bin/python "$f"; done
git diff --stat research-paper/      # expected: only research-paper/analysis/revision_numbers.json (last-bit p-values)
for f in r1_common_valid r1_fixed_den; do      # these two write into the current directory
  .venv/bin/python scripts/compare_outputs.py research-paper/review_20261003/$f.json $f.json; done
# 3. generated tables and figures
.venv/bin/python research-paper/graphics/make_figures.py fig1 fig3 fig4 fig5 fig5_recall fig6 figS1 tables
git diff --stat research-paper/graphics/*.tex research-paper/graphics/*.md
```

Step 4 (Table 3, hand-typed in the manuscript): the shares of the 114,403 key facts by predicate group come from
`answer_key_facts` filtered on `in_key` (expected 38.5, 1.35, 39.0, 5.0, 0.06, 0.3 and 15.7 percent).

Tolerance: the score report and the analysis outputs must agree exactly apart from creation timestamps and execution
context (`compare_outputs.py` default tolerance 1e-9 on numbers). In the release check, `revision_numbers.json` differed
from the stored copy only in the last binary digits of a few p-values (relative differences near 1e-16, from
floating-point summation order); every other output was identical. Re-rendered figures differ from the stored ones only
in renderer-generated element ids (SVG clip-path ids) and embedded timestamps (PDF); compare them visually.
`review_20261003/r1_common_valid.py` and `r1_fixed_den.py` write their JSON into the current directory; compare it
with the stored copy in `research-paper/review_20261003/` using `scripts/compare_outputs.py`.

The hand-typed manuscript tables (2, 3, 4, 5) and Figure 2 have no generator; docs/OUTPUT_MAP.md says where their
numbers come from. Figures 1 and 3 are regenerated with `make_figures.py fig1 fig3` (Figure 3 needs the compiled PDFs,
so materialise without `--skip compiled_pdfs`).

## Level 2: scores from stored answers, geometry and verdicts (CPU hours)

```bash
.venv/bin/python scripts/rescore_offline.py --split test --workers 7 --realign 40 --out rescore.json   # values used in the release check
```

For all 30,084 test cells this re-derives every geometry-decided verdict from the stored geometry and alignment with
the frozen code, checks the set of judge-routed facts, recomputes every item score with the frozen scorer and
compares them with the stored files; `--realign` recomputes the full alignment for a seeded sample of valid drawings
(some alignments take minutes to hours). Expected: every score and every geometry verdict equal (exit status 0).

With Docker and the sandbox image (docs/SANDBOX.md), recompile a stratified sample of stored answers, including
every invalid outcome code, and re-extract the geometry:

```bash
.venv/bin/python scripts/recompile_sample.py --per-outcome 2 --work ../recompile_scratch --out recompile.json   # 96 cells in the release check
```

What level 2 does not redo: the judges' decisions (Q1 fact votes and Q3 object choices) are stored model outputs and
enter as inputs; re-deciding them needs new model calls (level 3).

## Level 3: fresh generation and judging (paid; not run for this release)

Requires an AWS account with Amazon Bedrock access to the contestant and judge models in `configs/roster.yaml`, a
dated rate card (`configs/pricing.template.yaml`), the item images and page text (dataset `items`) and explicit
authorisation of the spend. The run of the paper cost about USD 156 in model calls for the six
contestants and their judging (ledger estimate).

1. Fill `configs/roster.yaml` (new contestants under `contestant:`; keep the judges, writers and screener),
   `configs/pricing.yaml` and, for batch inference, `configs/batch.yaml`, from the templates.
2. `python -m hmb --root ROOT roster-check --roster configs/roster.yaml`.
3. Open a scoring window with a judge drift probe, run the cells, close the window, close the model:
   `drift-probe --window open --allow-live`, `model-run ... --allow-live` (or `batch-run --allow-live`),
   `drift-probe --window close --allow-live`, `model-close --model-id <id>`; then `score-report` as in level 1.
   `--allow-live` is required for any paid call; without it every call must be answered from the local cache.
4. Keep compiles serial unless `sandbox-cputime-check` on your host shows no outcome change at the chosen concurrency.

New answers from the original six models will differ from the stored ones (provider-side model updates, one answer
per condition, temperature 0 is not a guarantee of determinism); the stored answers are the reference.
