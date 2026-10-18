# HistGramBench

Code, frozen protocol, run records and analysis for **HistGramBench**, a benchmark that asks vision-language models
to redraw printed historical mathematical diagrams (1551–1933) as TikZ programs and checks every redrawing fact by
fact on its own compiled geometry.

> **HistGramBench v2.** Benchmark data version `histmathbench-1.0.0`, release hash
> `sha256:a1bfbd4f1938236aa657dc10fc4e3f51f168b4324197d59d128087021e20283e`. This is the code and data of the paper
> *HistGramBench: Measuring Structural Recovery When Multimodal Models Redraw Historical Mathematical Diagrams as
> TikZ*. See [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md).

The Python package and command are named `hmb` (from the project's working name, HistMathBench). The names are kept
because the frozen protocol hashes the code byte for byte.

## The task in one paragraph

A model receives a figure crop (Track A) or the crop plus up to 1,200 characters of page text (Track B) and must
answer with one TikZ picture. The answer is extracted, normalised, policy-checked and compiled once with LuaLaTeX in a
network-less sandbox. The compiled page is read back as vector geometry, the figure's answer key (typed facts written
by two AI models under an agreement rule) is aligned to it, and each fact is decided on the geometry when its objects
bind, otherwise by a panel of judge models from other families. The item score is the share of checkable key facts
found present; the benchmark score averages within and then across 87 source groups.

## Two repositories

| | GitHub (this repository) | Hugging Face dataset |
|---|---|---|
| Holds | evaluation code, sandbox recipe, frozen configuration and prompts, tests, item-level scores (canonical), run reports, paper analysis scripts, tables and figures | figures and metadata, answer keys and their construction records, contestant answers, compiled drawings, geometry, verdicts, source catalogue, split groups, exclusion evidence |
| Planned id | `VietAlpha/histgrambench` (private) | `VietAlphaLabs/HistGramBench` (private, dataset) |

Each file has one canonical home. The dataset's `scores` table and its `metadata/` files are mirrors generated from
this repository and checked for equality (`scripts/verify_release.py`). Both ids are planned and not yet created.

## Installation

Two environments are used, so that the evaluation code runs on exactly the versions of the paper run:

```bash
# 1. pinned run environment (Python 3.12; the paper run used CPython 3.12 on Windows 11 and Amazon Linux 2023)
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.lock

# 2. release-tooling environment (reads the Parquet dataset)
uv venv --python 3.12 .venv-release
uv pip install --python .venv-release -r scripts/requirements-release.txt
```

Compiling answers additionally needs Docker and the pinned LuaLaTeX sandbox image (TeX Live 2025, Noto fonts);
analysis-only reproduction does not. See [docs/SANDBOX.md](docs/SANDBOX.md).

Do not clone with line-ending conversion: some source files have CRLF line endings, and the protocol hashes
(`code_hash`, `form_freeze_sha256`) are computed over the exact bytes. The repository's `.gitattributes` disables
conversion. `python scripts/check_hashes.py` recomputes the hashes and compares them with the run's values.

## Quick start (offline, no model calls)

```bash
# get the dataset (private: `hf auth login` first)
hf download VietAlphaLabs/HistGramBench --repo-type dataset --local-dir ../HistGramBench

# rebuild the run's directory layout from the released records (checks every sha256)
.venv-release/bin/python scripts/materialize_root.py --dataset ../HistGramBench --out ../hgb_root

# score one figure end to end from its stored answer (see examples/README.md)
HMB_ROOT=../hgb_root .venv/bin/python examples/minimal_example.py
```

On Windows use `.venv\Scripts\python.exe` and `set HMB_ROOT=...` (or `$env:HMB_ROOT=...` in PowerShell).

## Reproducing the paper

Three levels, described with exact commands, expected results and tolerances in
[docs/REPRODUCE.md](docs/REPRODUCE.md); the map from every table and figure to its inputs is
[docs/OUTPUT_MAP.md](docs/OUTPUT_MAP.md).

1. **Tables and figures from stored scores** (offline, minutes): regenerate the score report, the analysis numbers
   and every table, and compare with the stored copies.
2. **Scores from stored answers, geometry and verdicts** (offline, CPU hours): re-derive every geometry-decided
   verdict and every item score with the frozen code; optionally recompile a stratified sample of answers in the
   sandbox. Judge decisions are stored inputs; re-deciding them needs model calls.
3. **Fresh generation and judging** (paid, not run for this release): new contestant answers and judge calls through
   Amazon Bedrock with the frozen prompts and the drift-probe protocol.

## Evaluating a new model

The protocol for new contestants is level 3 in docs/REPRODUCE.md. Live calls are disabled unless a command is given
`--allow-live`; without it every model call must be answered from the local cache, and a miss stops the run. Three
batch subcommands (`batch-collect`, `batch-probe`, `batch-cancel`) talk to Amazon Bedrock batch jobs **without** the
`--allow-live` gate; do not run them unless you intend to submit or cancel paid jobs.

## Repository layout

```
hmb/                     evaluation code (python -m hmb); prompts in hmb/*/prompts, wrapper in hmb/submission
sandbox/                 Dockerfile and image lock of the LuaLaTeX sandbox
configs/roster.yaml      frozen model roster of the run (models, routes, decoding); *.template.yaml for new runs
frozen/                  frozen run configuration and records: thresholds, release and form-freeze manifests,
                         split groups, rights manifest, drift probes, run reports, per-model run records
results/                 item-level scores: score_cells.jsonl.gz (native, canonical) and item_scores.csv
research-paper/          analysis scripts, stored analysis outputs, table sources and figure renders of the paper
docs/                    reproduction, output map, sandbox, provenance; docs/protocol: plan, components, deviations
scripts/                 release tools: materialise, verify, rescore, recompile sample, compare outputs
examples/                minimal offline example
tests/                   test suite (pytest; `docker` tests need the sandbox image, `live` tests are deselected)
```

## Citation

Cite the paper (see [CITATION.cff](CITATION.cff) and [CITATION.bib](CITATION.bib)). No DOI or archive identifier
has been assigned yet.

## Licence

Code and dataset: Apache-2.0 ([LICENSE](LICENSE)). When reusing a figure, cite its source work (dataset `works`).
Third-party components: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Limitations

* Answer keys and judge verdicts are model-generated; no human audited them on the historical figures.
* Six low-cost contestant models, one answer per figure and track; run-to-run variation is not measured.
* The sandbox image is not distributed; a rebuilt image has a different identity, so its `harness_version` differs
  from the run's even when outcomes agree (docs/SANDBOX.md).
* Reproduction notes (host-dependent timeouts, hand-typed tables, release-manifest hashes): docs/RELEASE_NOTES.md.
