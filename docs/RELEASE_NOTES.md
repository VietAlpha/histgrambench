# HistGramBench v2: release notes

## Release

| | |
|---|---|
| Release | HistGramBench v2 |
| Benchmark data version | `histmathbench-1.0.0` (item records carry `dataset_version: v1.0`) |
| Release hash | `sha256:a1bfbd4f1938236aa657dc10fc4e3f51f168b4324197d59d128087021e20283e` (`frozen/manifests/release.json`; carried by every test answer) |
| Frozen forms | `form_freeze_sha256 255a5c7b…af84e` (`frozen/manifests/form_freeze.json`): registry, ontology, prompts, metric and estimator code |
| Frozen levels | `frozen/thresholds.json` (frozen 2026-09-28, before the test run) |
| Code | the code that scored the run; `code_hash 213b01d6…` (`python scripts/check_hashes.py`) |
| Dataset | Hugging Face dataset `VietAlphaLabs/HistGramBench` (items, answer keys, answers, verdicts, geometry, compiled drawings, source catalogue) |

The Python package is called `hmb` and the code refers to "HistMathBench", the project's working name; the names are
kept because the frozen hashes cover the code byte for byte.

## The run

| | |
|---|---|
| Contestants | Gemma 4 26B A4B, Llama 4 Maverick 17B, Ministral 3 14B, Gemma 4 E2B, Claude Haiku 4.5, Nemotron Nano 12B v2 VL (Amazon Bedrock) |
| Settings | temperature 0, top_p 1 (not sent for Claude Haiku 4.5), 8,192 output tokens, one answer per figure and track |
| Scoring window 1 | `20261001T134744Z-19efb0c8`: five contestants; judge drift probe open 0.867 / close 0.947 / open-vs-close 0.867 (required 0.8); records in `frozen/manifests/drift_probe/windows/20261001T134744Z-19efb0c8/` |
| Scoring window 2 | `20261002T144546Z-4c34b96c`: Claude Haiku 4.5; open 0.867 / close 0.867 / open-vs-close 0.984; records in `frozen/manifests/drift_probe/{open,close,result}.json` |
| Judges | Qwen3 VL 235B, Gemma 4 31B, Mistral Large 3; verdict files carry `judge_config_hash c06fd35a…` (the drift probe hashes the same judges under the label `drift-probe`, giving `f44c7335…`) |
| Compile host | one 8-vCPU x86_64 cloud VM with the pinned sandbox image `sha256:e902dd5a…`; 4 concurrent compiles, accepted after a CPU-time check showed no outcome change |
| Harness versions | `93e9e015…` for 25,060 answers; `d9a0fa81…` (this code) for 5,024: all Claude Haiku 4.5 answers and 10 Gemma 4 26B answers whose output was a lone `\documentclass` line, on which the earlier harness aborted (all 10 score 0). The only code difference is `frozen/records/wrapper_fix_20261002.diff` |

`frozen/manifests/release.json` was built before the run; its harness object (`harness_version 6009b2bb…`) and
`judge_config_hash` (`00156fb3…`, from the extractor label `cgx-1.6.0` instead of `cgx-1.6.0+geo-eval-1`) predate the
final code and the verdict files. The release hash and every item, answer-key and split identity are unaffected.

## Notes for reproduction

* **Timeouts depend on the host.** Answers that hit the 20-second CPU limit on the run's host can finish faster on
  another machine and then end in a compile error instead (4 of 4 sampled timeouts did). Both outcomes are invalid and
  score 0; only the split between `C_TIMEOUT` and `C_ERROR` in Table S4 changes.
* **Last-digit differences.** Regenerating `research-paper/analysis/revision_numbers.json` changes the last binary
  digits of a few p-values (relative differences near 1e-16); every printed number is unchanged.
* **Generated and hand-typed outputs.** Figure 2 has no generator script; Tables 2–5 are typed in the manuscript from
  the stored outputs listed in docs/OUTPUT_MAP.md.
* **Language and genre.** The paper's language and genre breakdown is inferred from work titles by
  `compute_revision_numbers.py`; the dataset exposes it as `language_title_heuristic` and `genre_title_heuristic`.
* **Synthetic-diagram key check (§5.2).** The check of keys against synthetic diagrams with known geometry was made
  during development with an earlier writer pair (Claude Sonnet 5 and Kimi K2.5); the code that computed it is not part
  of this release.
* **Release counts.** `release.json` counts test and development together (100 works, 188 split groups, 2,657 figures,
  121,013 facts); the paper's 95 works, 87 groups, 2,507 figures and 114,403 facts are the scored test set.
* **One stale test.** `tests/w2/test_build.py::test_roster_check_gatekeeper_only_three_writers_effort` fails: its
  fixture roster predates the Claude Haiku 4.5 entry (which may send only one of temperature and top_p).
