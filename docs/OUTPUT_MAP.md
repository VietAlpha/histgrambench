# Paper outputs: inputs and commands

Every table and figure of the paper (r21), the stored inputs it is computed from, and the command that regenerates
it. `ROOT` is a data root made by `scripts/materialize_root.py`; commands run from the repository root in the pinned
environment with `HMB_ROOT=ROOT`. Stored inputs are in `frozen/` (run records) and `research-paper/analysis/`,
`research-paper/review_20261003/` (analysis outputs); `make_figures.py` writes figures to
`research-paper/graphics/out/` and tables to `research-paper/graphics/*.tex|md`.

| Paper output | Content | Inputs | Regenerate |
|---|---|---|---|
| Table 1 | corpus composition | `analysis/revision_numbers.json` (`composition`), `analysis/outline_numbers.json` (`period`) | `make_figures.py tables` → `graphics/table3_composition.tex` (included by `\input`) |
| Table 2 | task conditions | page-text count from the score report | hand-typed in the manuscript (`graphics/table1_conditions.*` is an earlier draft, not used) |
| Table 3 | fact vocabulary and shares | answer keys (`answer_key_facts`, `in_key`) | hand-typed; shares recomputable from the keys (docs/REPRODUCE.md, level 1, step 5). `graphics/table2_facts.*` is an earlier draft |
| Table 4 | main results | `frozen/reports/scores_test_histmathbench-1.0.0.json` (H, bands, validity), `outline_numbers.json` (`H_valid`), `revision_numbers.json` (`loss_decomposition_A`) | hand-typed from those files (`graphics/table3_results.*` is an earlier draft) |
| Table 5 | page text, condensed | score report (`context_effect`), `revision_numbers.json` (`page_text`) | hand-typed condensation of Table S5 |
| Table 6 | stability under variants | `revision_numbers.json` (`sensitivity_A`, `item_weighted_A`, `headline_A`) | `make_figures.py tables` → `graphics/table5_stability.tex` (pasted inline) |
| Table 7 | strata | score report (`by_diagram_type`, `by_touching_cc`), `revision_numbers.json` (`strata_A`), `outline_numbers.json` (`period`) | `make_figures.py tables` → `graphics/table6_strata.tex` (pasted inline) |
| Table S1 | exclusion reasons | `frozen/reports/funnel_report.json` | `make_figures.py tables` → `graphics/tableS2_exclusions.tex` (pasted inline) |
| Table S2 | model roles and families | `configs/roster.yaml`, score report | `make_figures.py tables` → `graphics/tableS4_families.tex` (`\input`) |
| Table S3 | key retention | `review_20261003/da_key_out.json` (from `da_key.py`) | `make_figures.py tables` → `graphics/tableS5_key_retention.tex` (`\input`) |
| Table S4 | outcome codes | score report (`validity.outcome_distribution`) | `make_figures.py tables` → `graphics/tableS3_outcomes.tex` (pasted inline) |
| Table S5 | page text by subset | `revision_numbers.json` (`page_text`) | `make_figures.py tables` → `graphics/table7_page_text.tex` (pasted inline) |
| Figure 1 | sample mosaic | `research-paper/media/*.png`, `media/sample_manifest.csv`, item records | `make_figures.py fig1` → `fig1_mosaic.*` |
| Figure 2 | pipeline diagram | none recorded | `graphics/out/fig2_better.png` has no generator in the project (`paperfigs/fig2_pipeline.py` writes an earlier version, `fig2_pipeline.*`) |
| Figure 3 | worked example | `compiled/<model>/A/hmb_284751dbdd7d38c6/cand.pdf`, the item's crop | `make_figures.py fig3` → `fig3_worked_example.*` (reads the item's crop and the compiled PDFs from the data root) |
| Figure 4 | pairwise differences | score report (`tables.*.partial_order`), `frozen/thresholds.json` (δ\*) | `make_figures.py fig6` → `fig6_pairwise.*` |
| Figure 5 | outcomes per model and track | score report (`validity`) | `make_figures.py fig4` → `fig4_outcomes.*` |
| Figure 6 | class and tier recall, Q5, contradicted share | `revision_numbers.json` (`relation_tiers`, …) | `make_figures.py fig5_recall` → `fig5_class_recall.*` |
| Figure 7 | H by track | score report (`context_effect`) | `make_figures.py fig5` → `fig5_page_text.*` |
| Figure S1 | period by repository | item records, rights manifest | `make_figures.py figS1` → `figS1_corpus.*` |
| Text numbers | headline H and CI, BH bands, δ\*, τ_b, drift agreement, separated pairs | score report, `revision_numbers.json`, `outline_numbers.json`, `frozen/manifests/drift_probe/` | see the files; `analysis/compute_revision_numbers.py`, `analysis/compute_outline_numbers.py` |
| Text numbers | common-valid difference, decomposition checks, forced bounds, fixed denominator, judge timing | `review_20261003/r1_*.json`, `da_*.json` | the matching `review_20261003/*.py` script |
| Text numbers | X_i share (21–38 %), judge-family disagreement (16–24 %), 149 keys with A and a | scores (`per_class`), verdicts (votes), answer keys | hand-computed for the paper; no script: X_i share = `1 − n_star/n` of class `R` on valid Track A drawings; disagreement = judged facts whose two eligible families' Q1 votes differ; case pairs = keys whose retained `label` facts contain two labels equal after lower-casing |
| §5.2 synthetic check | 170 of 172 mappable key facts true; 76 % mappable; 51 % of true facts | none released | made during development with an earlier writer pair; not reproducible from this release (docs/RELEASE_NOTES.md) |

The stored analysis outputs are produced as follows (each script writes its JSON next to itself, so running it in a
checkout overwrites the stored copy and `git diff` shows any change; `r1_common_valid.py` and `r1_fixed_den.py` write to the
current directory instead, so compare their output with `scripts/compare_outputs.py`):

| Output | Script | Reads |
|---|---|---|
| `frozen/reports/scores_test_histmathbench-1.0.0.json` | `python -m hmb --root ROOT score-report --models <6> --track A --track B --split test` | scores, submissions, items, answer keys, verdicts |
| `frozen/reports/funnel_report.json` | `python -m hmb --root ROOT funnel-report` | stored only: the candidate and page logs it reads are not released |
| `research-paper/analysis/revision_numbers.json` | `research-paper/analysis/compute_revision_numbers.py` | score report, item records, answer keys, scores, verdicts |
| `research-paper/analysis/outline_numbers.json` | `research-paper/analysis/compute_outline_numbers.py` | the same |
| `research-paper/review_20261003/*.json` | `research-paper/review_20261003/<name>.py` (two write to the current directory) | the same |
