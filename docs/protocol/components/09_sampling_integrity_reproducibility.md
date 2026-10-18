# Component 09 — Sampling, split groups, selection reporting, contamination, pilot size, and reproducibility

Part of *HistMathBench Zero-Annotation Project Plan v2*. Canonical entry point: [`../HistMathBench_Zero_Annotation_Project_Plan_v2.md`](../HistMathBench_Zero_Annotation_Project_Plan_v2.md). This file owns **Section 13** (sampling, grouped splits, selection funnel, contamination, pilot-size logic) and **Section 15** (reproducibility, manifests, versioning, release policy, hidden-test integrity). Global notation is in the top-level plan, Section 3.4.

| Component card | |
|---|---|
| Purpose | Make held-out claims true, selection bias visible, contamination bounded, the pilot correctly sized, and every published number reproducible from hashed artifacts. |
| Inputs | Rights manifest, duplicate clusters, bibliographic fields, funnel rows, stored votes and verdicts, readiness-track census prominence indicators. |
| Mechanism | Split groups as connected components over hard edges; two-way pilot split; per-stage per-stratum funnel with two bias companions; leave-one-family-out re-derivations; prominence association and a text-only recall probe; a simulation-based sizing rule; a hash chain with a version trigger table. |
| Outputs | `sources/split_groups.yaml`, funnel report, contamination tables, `N*` and `G*`, `manifests/release.json`, run manifests, tombstones. |
| Gates | Collapse guard (13.1); slice minimums (11.11); G2 yield (17.5). |
| Retained artifacts | Everything hashed into `release_hash`. |
| Cost tier | CPU; the text-only probe is about 150 contestant calls plus judging. |
| Lifecycle | Split groups once per release; funnel per run; manifests per release and per run. |
| Human effort | None per item. The collapse-guard decision is one recorded owner choice before any contestant call. |
| Dependencies | Sections 4, 6, 8, 10, 11, 14, 16, 17. |
| Downstream decisions | Validity of every held-out claim; scale-or-stop inputs G2 and G7. |

> **Checkpoint simplification 2026-09-26.** The Track C control suite (CP3) and evaluator validation (CP5) were
> deleted and milestones MS1–MS5 replace CP1–CP11. For this component: the development split no longer supplies
> `w_p` or severity quantiles and now supplies the drift-probe triples (13.2); nothing is calibrated on Track C
> (13.2); the development probe and sizing run in MS3 (`dev-measure`, `dev-sizing`, 13.7); `controls_hash` and
> `controls/MANIFEST.sha256` leave the hash chain, so `release_hash = H(cardset, split, thresholds, judge_config,
> harness, metric)` (15.1, 15.2); the Track C partitions and the `controls/` tree leave the release (15.6, 15.7).

---

## 13. Sampling, splits, selection reporting, contamination, and pilot size

### 13.1 Split groups

Nodes are books. **Hard edges** force merging: (1) a duplicate edge — two items from different books in the same `phash_cluster_id` at the release threshold `τ_dup ≡ phash_near_hamming`, the G7d pHash clustering threshold named in `thresholds.json` (no separate `τ_dup` value exists; BUILD_STATUS C35); (2) a work-identity edge — normalized (title, author) match after script folding and transliteration, or a shared catalogue work identifier; (3) an edition or translation edge from catalogue metadata. **Soft edges** are reported, never merged: same author without title match; same publisher and series within ten years.

A **split group** is a connected component over hard edges. Assignment is at group level. Residual cross-group duplicates that survive at graded similarity drop every development-side member and keep the test-side item. `soft_edge_crossings` (development–test book pairs joined by a soft edge, with type) is published as the honest residual.

**Collapse guard.** If the largest component holds more than 50 percent of eligible items, do not force a split: either drop the lowest-value book whose removal breaks the component, or declare the pilot single-group and abandon the held-out claim. The choice is made and recorded **before any contestant call**.

> **Owner amendment, 2026-09-23 (decision D7, implemented 2026-09-23): standing rule, no manual choice.** The books
> of the collapsed component are tried in order of eligible-item count (ties by `book_id`). The first book whose
> removal splits the component into at least two groups, with none above 50 percent, is dropped. If no single drop
> works, the pilot is single-group and the held-out claim is abandoned. **Floor (decision D12, implemented
> 2026-09-23):** the component's largest-eligible-count book is never dropped; if only its removal would work, the
> pilot is single-group. The rule and its outcome are written into `sources/split_groups.yaml`
> (`collapse_guard.rule`, `action`). There is no owner decision file.

### 13.2 Pilot split: two-way

| Split | Groups | Function | Released |
|---|---|---|---|
| development | 3 split groups | pipeline debugging; retention measurement; the variance probe (13.7); cost estimation; prompt sanity; public example cards; the drift-probe triples (Section 10.9) | fully, at paper time |
| test | the remaining 7–12 groups, about 300–400 eligible items | all reported scores | frozen at the end of MS4 (release); public at paper time |
| reserved | none in the pilot | | |

There is no public-test or hidden-test split in the pilot, no submission server, and no rate limiting; those belong to V1 (13.8). A three-way split of 10–15 books would leave 3–5 test groups, for which no valid comparison can reach `p < 0.05` (Section 14.2), so the original hierarchy was internally inconsistent with its own inference method.

**Assignment (v2, 2026-09-26).** Every split group that holds a book used before the level freeze (MS2/MS3,
`sources/development_books.jsonl`) is development; the seeded draw of the remaining groups fills the development
split up to `dev_groups`, and every other group is test (`sampling.assign_splits`, `python-random-shuffle-v2`).

**Permitted uses of development books:** the functions listed above, exhaustively. **Forbidden:** selecting or adjusting any consensus rule, eligibility threshold, metric definition, cap, or judge parameter using development-book scores. They are form constants fixed at MS1 or levels frozen at the MS3 level freeze (Section 17.3); the Track C calibration that used to supply some of them was removed on 2026-09-26.

### 13.3 Selection funnel reporting and bias bounds

Stages S0–S8 are defined in Section 6.6. For every stage and stratum (split group, book, language, decade, predicted category, degradation-statistic tercile, `source_codec`), publish `n_{s,g}` and conditional retention `r_{s,g} = n_{s,g}/n_{s−1,g}`, plus cumulative survival separately.

The eligibility rate `e = n_S8 / n_S4` over confirmed target diagrams accompanies every headline in the claim sentence (Section 1). Two optional companions (simplification 2026-09-25: implemented, not mandatory, never blocking) are available (H = the headline score, `H_g` the mean score in stratum `g` among eligible items, `w_g = n_{S4,g}/n_S4` the pre-exclusion stratum share):

```text
S_adj    = Σ_g w_g H_g                        # post-stratified; assumes ignorable missingness within stratum (stated)
envelope = [ H·e , H·e + (1 − e) ]            # assumption-free, since scores lie in [0, 1]
```

A composition table of eligible versus excluded items on all pre-gate automatic metadata (with exclusion reasons) is published, labelled a bias indicator computed from unvalidated metadata, not a correction. Inverse-probability weighting is not used: its selection model would be fitted on the same metadata that drives selection.

Mandatory claim-boundary sentence: the headline is an estimate for the machine-eligible subpopulation; eligibility is correlated with legibility; degradation effects are attenuated by construction because the most degraded items are preferentially excluded.

### 13.4 Stratification policy

Book is the sampling unit. Stratify **book selection** on at most two dimensions (language family; publication period). Category, degradation, label density, and complexity are post-hoc reporting strata under the slice minimums of Section 11.11, never balancing constraints. At census, place at least two books in each (codec class × language family) cell intended for reporting, or declare the slice unreportable in advance.

### 13.5 Family overlap: builders, judges, contestants

Per eligible fact, the set of supporting builder families is a released field (Section 8.5). Leave-one-builder-family-out was removed 2026-09-26: with the locked two-writer roster and the both-writers rule it reduced to "one writer's facts alone", a set the rule never admits. **Leave-one-judge-family-out** re-decides stored verdicts without one judge family and is the G6 input. A **family-overlap matrix** (contestant families × {builder families, judge families}) is published with at least one reported configuration per contestant in which its family appears in neither role; where none exists, the table says so. Zero new model calls.

### 13.6 Contamination bound

The plausible pretraining channel for these books is the public OCR text layer, which is exactly the Track B input; a page-image-to-TikZ pairing for 1850–1930 books essentially does not exist. Per book, from the census: `hosted_by_major_aggregators` (count) and `public_fulltext_layer` (boolean), combined into a prominence tercile at book level.

- **T1 (descriptive):** headline by prominence tercile with the group-level test; effect size and interval, no stars.
- **T2 (zero cost):** the context effect of Section 11.7 by prominence tercile; if memorized text is doing work, high-prominence books show larger Track B gains, and the item pairing removes difficulty.
- **T3 (the one new spend):** a **text-only recall probe** on about 50 test items × 3 models: supply the context text with no crop, score normally. Correct reconstruction with no image is recall, not reconstruction. Reported per tercile. About 150 contestant calls plus judging. This is a diagnostic condition, not a track.

Claim boundary: the benchmark cannot exclude pretraining exposure; it bounds the association between score and prominence and reports a direct text-only recall rate.

### 13.7 Pilot-size logic

The pilot stays at roughly 400–500 eligible items from roughly 10–15 books unless the analysis below shows a smaller pilot answers the same gating questions. Items are the wrong axis for power (Section 14.3: `N_eff ≤ G/ρ`), so the plan measures before committing.

**Development probe (MS3 in Section 17; `dev-measure` and `dev-sizing`, formerly CP6/CP7):** two contestant models on about 100 development items, Track A only. From the paired differences `d_i` grouped by development book:

```text
n₀    = (N_dev − Σ_g n_g² / N_dev) / (G_dev − 1)
ρ̂     = max(0, (MSB − MSW) / (MSB + (n₀ − 1) MSW))      # one-way random-effects ANOVA on d_i by book
σ̂²_d  = sample variance of d_i
ρ_plan = 80th percentile of a book-level bootstrap distribution of ρ̂      # plan on the upper end
```

**Simulation-based sizing rule** (executable from probe artifacts only):

```text
inputs: dev paired differences d_ig; grid N ∈ {150, 250, 350, 450}; G ∈ achievable test-group counts; δ* preregistered
for (N, G): for b in 1..2000:
    draw G group effects by resampling dev book means of d
    draw n_g per group from the empirical cluster-size distribution scaled to N
    draw residuals by resampling the within-book residual pool
    d_i := δ* + u_g(i) + e_i
    reject := the Section 14.2 primary test rejects at α = 0.05
power(N, G) := mean(reject)
N* := min{N : power(N, G) ≥ 0.80};   if none, G is binding, not N
```

Decision rules: if `ρ_plan ≤ 0.05` and `σ̂_d ≤ 0.15`, a pilot of 250 items suffices for any `δ* ≥ 0.05` at `G = 12`; if `ρ_plan ≥ 0.15` at `G = 12`, no `N` reaches `δ* = 0.05` (floor about 0.069 at `σ_d = 0.20`), and extracting 450 instead of 250 buys about 7 percent precision for 80 percent more dominant cost, so the pilot is **not** enlarged; the options are to raise `δ*` and state it, or to redistribute the same `N*` across more of the ≥ 20 rights-cleared books from the census at constant item count. The redistribution changes per-book human effort (one approval per added book) and departs from the 10–15 book default, so it is an **explicit owner decision recorded in the preregistration**, never a default. `G_min = 6` test groups is a hard floor for any inferential claim.

### 13.8 What is reserved for V1

A three-way split (development / public test / hidden test) when `G ≥ 30` split groups; a submission server, per-submitter budgets, and rate limits; the hidden set drawn from books never used in the pilot, which the Phase 1 census therefore **reserves up front** (pilot test books become public at paper time). Unscored-pool items inherit their group's split and are released for development and test groups; counts only for any reserved group.

---

## 15. Reproducibility, manifests, versioning, release policy

### 15.1 Canonical hashing chain

`H` = SHA-256 over RFC 8785 canonical JSON of UTF-8 NFC strings, or over raw bytes for binary files.

```text
page_hash         = H(page_image_bytes)
item_hash         = H(crop_image_bytes ‖ extraction.pipeline_version ‖ page_hash ‖ crop_bbox)
card_hash         = H(frozen_eligible_fact_set ‖ asc_version ‖ consensus_rule_hash)
cardset_hash      = H(sorted list of (item_hash, card_hash))
split_hash        = H(split_groups.yaml)
thresholds_hash   = H(thresholds.json)                   # every gate level, tolerance, cap, τ, κ_min, α_min, phash_near_hamming (= τ_dup)
judge_config_hash = H(judge prompts ‖ verdict schema version ‖ judge model ids with snapshot dates
                      ‖ extractor version ‖ protocol constants (decision rule, chunk size, retries, §10.2 tolerances) ‖ decoding)
harness_version   = H(evaluation code commit ‖ sandbox_image_digest ‖ wrapper_version ‖ extraction_rules_version
                      ‖ rasterizer name+version, DPI, canvas)
metric_hash       = H(metric definitions ‖ metric constants ‖ aggregation rules)
release_hash      = H(cardset_hash ‖ split_hash ‖ thresholds_hash ‖ judge_config_hash ‖ harness_version ‖ metric_hash)
```

`controls_hash` (over `controls/MANIFEST.sha256`) was removed from the chain on 2026-09-26 with the control suite.

### 15.2 `manifests/release.json`

```json
{
  "benchmark_version": "histmathbench-1.0.0",
  "release_hash": "sha256:...",
  "created": "...",
  "cardset_hash": "...", "split_hash": "...", "thresholds_hash": "...",
  "thresholds_frozen_at": "...",
  "judge_config_hash": "...", "metric_hash": "...",
  "harness": {"eval_commit": "...", "container_digest": "sha256:...", "tex_distribution": "...", "rasterizer": "...", "dpi": 300, "canvas": "1024x1024"},
  "counts": {"books": 0, "split_groups": 0, "items_eligible": 0, "facts_eligible": 0},
  "funnel": "manifests/funnel_1.0.0.json",
  "tombstones": "manifests/tombstones.json"
}
```

### 15.3 Run manifest per contestant (`scores/<model>/run.json`)

`release_hash`; the model record (id, provider version, family, access date, decoding, attempts) and its hash; `prompt_sha256`; the scoring window (open, close); the drift-probe result (open agreement, close agreement, `κ_min`, pass); `verdicts_hash`; the funnel eligibility rate; `output_release`. **Plotting rule:** one leaderboard axis carries exactly one `release_hash`.

### 15.4 Version trigger table and tombstones

| Change | Bump | Why |
|---|---|---|
| card content, ontology, rules R/E, consensus rule, eligibility levels | MAJOR | every score changes |
| judge configuration (model, snapshot, prompt, extractor, protocol constants) | MAJOR | every verdict changes |
| metric definitions or caps | MAJOR | every score changes |
| compile container, TeX pin, rasterizer, DPI, wrapper, extraction rules | MAJOR | `Valid@1` and renders change |
| split assignment or split-group graph | MAJOR | denominators change |
| adding books or items without altering existing ones | MINOR | old scores remain valid on the old subset |
| adding an analysis or reporting slice | MINOR | no existing number changes |
| item retired for rights reasons | MINOR plus tombstone | see below |
| documentation, paths, non-scoring metadata | PATCH | provably unused in scoring |

`manifests/tombstones.json` records `{item_id, retired_in_version, reason, evidence}`. Ids are never reused. Previously published scores keep their `release_hash` and are never recomputed; the new version publishes scores on the reduced set alongside, both counts visible.

### 15.5 Threshold freeze artifact

All thresholds live in one `thresholds.json`, hashed into `release_hash` and timestamped at the level-freeze (Section 17.3) **before any test-book contestant call**. The published `release_hash` proves the freeze predates the scores. Any later change is a MAJOR bump with a written reason.

### 15.6 Artifact visibility (pilot)

| Artifact | Status |
|---|---|
| schemas, ontology, prompts, `thresholds.json` with freeze timestamp | public, frozen at the end of MS4 (release) |
| development crops, cards, votes, verdicts | public at paper time |
| test crops, cards, votes, verdicts | frozen at the end of MS4 (release); public at paper time |
| funnel (counts and per-stratum retention, all splits) | public |
| rights manifest, evidence bundles, provenance view, split groups | public |
| release and run manifests, all hashes | public |
| unscored pool | released for development and test groups |
| model outputs | per `output_release`; non-releasable models are marked not independently reproducible |
| submission server, rate limits | not built in the pilot |

The Track C validation and calibration partitions were removed from this table on 2026-09-26 with the control suite.

Canonical-source rule: `sources/rights_manifest.yaml` is authoritative for book-level fields; `sources/split_groups.yaml` for grouping; the item record references `book_id` and never restates book fields; `sources/provenance.csv` is a generated view, never hand-edited.

### 15.7 Repository tree

```text
histmathbench/
├── data/{development,test}/            # item records
├── images/{pages,crops,crops_canonical,context}/
├── semantic_cards/                     # frozen cards; votes/ alongside
├── sources/{rights_manifest.yaml, evidence/, split_groups.yaml, provenance.csv}
├── evaluation/{extract,compile,render,card_builder,extractor,adjudicator,metrics}/
├── prompts/  schemas/  manifests/{release.json, funnel_*.json, tombstones.json, pricing.yaml, cards/, drift_probe/}
├── releases/<version>/
├── thresholds.json  benchmark_card.md  LICENSE  README.md
```

### 15.8 Hidden-test integrity (V1)

Reserved books are never extracted in the pilot; their identity is recorded in the census and hashed. At V1, hidden cards and judge configuration remain server-side; marginals (per-stratum counts and retention) are public, per-item metadata for reserved groups is not. For a generative reconstruction task, marginals are not exploitable the way they are for multiple-choice benchmarks.
