# Component 10 — Cost model, staged inference policy, phase plan, gates, and the scale-or-stop rule

Part of *HistMathBench Zero-Annotation Project Plan v2*. Canonical entry point: [`../directives/HistMathBench_Zero_Annotation_Project_Plan_v2.md`](../directives/HistMathBench_Zero_Annotation_Project_Plan_v2.md). This file owns **Section 16** (cost model and staged inference policy) and **Section 17** (milestones with automated exits and the scale-or-stop rule). Global notation is in the top-level plan, Section 3.4.

> **Checkpoint simplification 2026-09-26** (`reports/checkpoint_simplification_20260926.md`). CP1–CP11 are replaced by
> five milestones, each answering one decision: MS1 form freeze, MS2 pilot, MS3 development set with sizing and the
> level freeze (was CP6, CP7 and CP4), MS4 test set and release (was CP8–CP9), MS5 final run and report (was
> CP10–CP11). CP3 (Track C controls) and CP5 (evaluator validation) are deleted with gates G3–G5; CP3b (census and
> rights) is a parallel readiness track with one condition before MS4. The CAL pre-freeze context, the control
> judging cost term and the three-attempt cap are gone. 7 calibrated parameters remain (was 14).

| Component card | |
|---|---|
| Purpose | Spend in the order that maximizes information before irreversible commitments, meter every model call, and decide scale-or-stop against preregistered gates. |
| Inputs | The cost ledger, funnel counts, the development set and variance probe, model-run outputs. |
| Mechanism | Symbolic cost model with scaling classes; an append-only ledger with a dated rate card; cheap-before-expensive ordering; a two-act freeze; five milestones; gates split into blocking (G1, G7), modulating (G2, G6, G8) and informational (G9); a decision table. |
| Outputs | `manifests/pricing.yaml`, the ledger, `thresholds.json` provenance records, the decision record. |
| Gates | G1, G2, G6–G9 (17.5); milestone exits MS1–MS5 and `books_ready` (17.2). |
| Retained artifacts | Ledger, milestone exit records, decision record, all hashed. |
| Cost tier | Negligible. |
| Lifecycle | Ledger continuous; phases once. |
| Human effort | Phase-level approvals only: book approval (Section 4), the sizing decision (13.7) and the final publish or stop decision. The collapse guard is a standing deterministic rule (13.1, D7/D12) and spend is tracked, not capped (16.5 item 8, D2). |
| Dependencies | Every other section. |
| Downstream decisions | Whether and how the project scales. |

---

## 16. Cost model and staged inference policy

### 16.1 Symbols

| Symbol | Meaning | Scaling class |
|---|---|---|
| `G` | pilot books extracted | per book |
| `P` | pages rendered | per page |
| `K` | proposals emitted at G1 | per proposal (CPU) |
| `K_f` | proposals reaching G8 (paid classification) | per crop |
| `N_cand` | candidates entering card building | per item |
| `N` | eligible scored items | per item |
| `r` | retention `N / N_cand` | ratio |
| `B` | answer-key writers per item (2 since 2026-09-26) | multiplier |
| `M` | contestant models | per model |
| `T` | tracks per model (2) | multiplier |
| `q` | share of valid cells with at least one `model`-class fact | ratio |
| `E[J]` | judge calls per valid cell with model facts (2 + third-judge and Q2/Q3/Q4 expectation) | multiplier |
| `c_*` | metered unit cost of one call at stage `*` | USD per call |

### 16.2 Cost equation

```text
Cost_total = Cost_onetime_engineering
           + G · c_book                                              [per book: rights pass, download]
           + P · (c_render + c_ocr + 3 c_det)                        [per page]
           + K_f · (c_cls + c_ver + c_meta)                          [per crop reaching G8]
           + N_cand · 2B · c_card                                    [per item, once: generation + verification]
           + M · T · N · ( c_gen + Valid@1 · ( c_cgx + q · E[J] · c_judge ) )   [per model: dominant recurring]
           + c_probe                                                 [drift probes, per scoring window]
           + c_textprobe                                             [about 150 contestant calls + judging]
```

`Valid@1` multiplies the judging term because invalid outcomes never reach the judge; `q` and the extractor remove judge calls for cells whose facts are all `geo`. These are the two largest savings relative to the original design, and both are measured, not assumed.

### 16.3 Scaling classes

One time: ontology, schemas, sandbox, harness, extractor. Per book: rights verification (human), download, render, OCR, detection. Per crop reaching G8: classification, verification, metadata. Per item (paid once, cached for every model and every version that keeps the ontology): card generation and verification. Per item per model per track (the only term that recurs with each new model): generation, compile, extraction, judging.

### 16.4 Ledger and rate card

Every model call appends `{stage, model_id, input_tokens, output_tokens, image_count, wall_ms, unit_price_id}` to an append-only ledger. `unit_price_id` references a dated entry in `manifests/pricing.yaml` carrying the provider's official pricing URL and retrieval date. **No provider price appears in this plan**; the pilot (MS2) meters `c_judge`, `c_card`, `c_gen` to within a factor of about 1.5 for a few dollars, and every projection thereafter uses metered values. Marginal costs are computed, not estimated:

```text
c_item  = Δ ledger cost / Δ eligible items        over the last complete extraction batch
c_book  = Δ ledger cost / Δ books                 over the last complete book
c_model = Δ ledger cost / Δ models evaluated      over the model run
```

### 16.5 Staged inference policy

1. All CPU gates (Section 6.3, G0–G7) precede the first paid call; pHash precedes the verifier.
2. The strict gatekeeper (G8b) is the only paid screening call per crop (role audit 2026-09-23); there is no category classifier.
3. Card generation and verification use strong models, paid once.
4. Outcome codes 1–15 never reach adjudication.
5. The deterministic extractor decides `geo` facts; judges are called only for cells with `model` facts or ambiguous entities (the extras channel was removed 2026-09-26).
6. Two judges on every such cell; the third only on facts the first two do not decide; no fourth vote.
7. (Deleted 2026-09-26 with Track C: there is no control judging.)
8. A preregistered aggregate ceiling `X_total` and per-phase caps are enforced by the ledger: at 80 percent of a phase cap the driver halts and requires explicit human authorization to continue. If a projection exceeds the judging budget, the model set is reduced (frontier subset preserved), never the per-cell evidence.

   > **Owner amendment, 2026-09-23: spend is tracked, not capped.** Item 8 is suspended. There is no `X_total`, no
   > per-phase caps, no 80 percent halt, no judging budget B and no `X_V1` ceiling. The ledger still meters and
   > records every call, per phase, stage and model, against the rate card. G9 still reports the projected V1 cost,
   > but only as information. See `reports/decision_audit_20260923.md`, Decision 2. Implemented 2026-09-23: the ledger
   > has no caps, halt or `ledger-authorize`; the five registry entries and the unwired budget governor are removed.
   > No milestone has a spend stop rule.

### 16.6 Order-of-magnitude anchor (all prices are placeholders, not verified)

The reviewer of record ran a symbolic anchor with assumed unit prices solely to rank line items; the ranking is stable under uniform price scaling and the absolute conclusion is not. Under those placeholders the original design's largest line item was the implied full-factorial control suite (about 36,000 judge calls), roughly four times the model study; the redesigned control suite (about 16,000 calls per version) was deleted on 2026-09-26; the extraction pipeline is a rounding error; and the dominant recurring term is `M · T · N` contestant generation plus judging. Whether API spend binds at all, versus engineering time, is decided by the pilot's (MS2) metered values, and each Phase 0 deliverable carries an engineer-day estimate so that design arguments are made on the axis that actually binds.

---

## 17. Phase plan, gates, and the scale-or-stop rule

### 17.1 Ordering principle

Cheap before expensive, development before test. The pilot and the development set run on development books only,
before any test-book artifact exists, and determine the decoding caps, the retention rate, the fact yield, `σ_d`,
`ρ`, and therefore `N`, `G`, and `M`. Book census and rights clearance run in parallel because they are librarian
work with no dependency on the pipeline. Each milestone exists because it answers one decision that no adjacent
milestone can answer without either using test data or re-running paid work on changed forms.

### 17.2 Milestones

| MS | Milestone (was) | Decision it supports | Minimum inputs | Outputs consumed downstream | Automated exit (PASS) | FAIL / NOT_RUN |
|---|---|---|---|---|---|---|
| MS1 | **Form freeze** (CP1) | the protocol forms are fixed before any empirical data that evaluates the benchmark | schemas, ontology (7.1), rules R and E, prompts, wrapper, outcome codes, metric and estimator code, every gate form with named free parameters | owner-approved `form_freeze_sha256`; the thresholds template | owner approval of the current forms; registry hash recorded (`ms1_exit`) | NOT_RUN until the owner approves; any later edit to a frozen component makes it NOT_RUN again |
| MS2 | **Pilot** (CP2) | the real pipeline runs end to end and its decoding caps and costs are known before the development set is paid for | MS1; 2 R3-approved development books; ≤ 30 crops (`hmb pilot`, `prefreeze:PILOT`) | metered `c_judge`, `c_card`, `c_gen`; pre-freeze decoding caps; the `geo` share | zero crashes; every record validates; costs metered; body invariant asserted; determinism re-run green; font coverage green (`ms2_exit`) | FAIL → fix the harness; a form change → new MS1 approval and rerun; two failed attempts → re-scope the harness |
| MS3 | **Development set, sizing and level freeze** (CP6, CP7, CP4) | the levels, `N*`, `G*` and `δ*` that govern test construction and scoring | MS2; approved development books, disjoint from test books (`hmb develop`, `prefreeze:DEV`) | `dev-measure` (retention `r̂`, `F̄`, contradiction rate, family agreement); `dev-sizing` (`σ̂_d`, `ρ̂`, `N*(G)`); the drift-probe triple set; `thresholds.json` with provenance for the 7 calibrated parameters (`hmb freeze`) | `r̂ ≥ 0.25` and `F̄ ≥ 6`; `δ* ≥ 1.1 × MDD_floor`; every parameter frozen with value, artifact id, hash, date (`ms3_exit`) | `r̂ < 0.25` or `F̄ < 6` → revise the builder (a form change, so MS1 again); `δ* < 1.1 × MDD_floor` at achievable `G` → raise `δ*` or record the owner decision on more books (13.7) |
| MS4 | **Test set and release** (CP8, CP9) | the frozen test set is large enough to run the models | frozen levels; `books_ready`; R3-approved test books (`hmb build`) | frozen cards, split groups, item records, `release.json` (release hash) | `N_cand ≥ N*/r̂ × 1.25` and G2 (yield) (`ms4_exit`) | shortfall → more pages or reserve books; never a gate or level change |
| MS5 | **Final run and report** (CP10, CP11) | scale, reduce or stop (17.6) | the release; one scoring window with drift probes (`hmb run`, `hmb report`) | scores, the score report, the gate record and the decision | G6, G7, G8 computable and the 17.6 decision is not UNDECIDABLE (`ms5_exit`) | a drift-probe failure invalidates the window (10.9); spend is tracked, not capped (D2) |

**Readiness track (was CP3b).** Candidate books, R3 approvals, evidence bundles and terms, prominence indicators and
the reserved V1 hidden books are gathered in parallel with MS1–MS3. The only condition the sequence checks is
`books_ready`, before MS4: the R3-approved test books form at least `G*` split groups after grouping, with every
pre-freeze development book excluded (the split guard of `items-finalize` refuses otherwise).

**Why these cannot merge.** MS1 fixes what MS2 may change; merging them would freeze forms before the pilot has
exposed breakage. MS2 can change forms, and MS3's measurements are valid only on final forms, so a merged
pilot-development stage would discard paid development work after every fix. MS3 must end in the level freeze, and
MS4 must not begin until it has, because test construction reads `N*` and the frozen levels. MS5 needs the release
hash fixed before any contestant call. The CP4–CP7 contradiction of the old sequence (levels, `N*`, `G*` frozen at
CP4 but chosen at CP7) is removed by placing sizing inside MS3.

**C42, fixed 2026-09-26.** The variance probe runs before the level freeze on development books: `model-run --split development --thresholds prefreeze:DEV` scores the development-book candidates whose card gates pass (no item record, release or scoring window needed; the test split is refused), and `dev-sizing --thresholds prefreeze:DEV` groups them by the split graph over the development books and records its execution context, so the level freeze can cite it. `drift-triples` draws the frozen §10.9 triple set from those judged cells. At MS4, `items-finalize` puts every split group that holds a pre-freeze development book in the development split (the seeded draw fills the rest), and the release's judge-config check reads test items only.

**Deleted.** CP3 (control generation) and CP5 (evaluator validation, V1–V11, at most three attempts) with gates
G3–G5, the CAL/VAL partitions, the card arm, `s*`, the severity calibration and `w_p`: see component 08.

### 17.3 Two-act freeze

- **Form freeze (MS1):** every gate's functional form, estimator, denominator, and decision rule is fixed and hashed with free parameters named (for example, "item eligibility requires `|eligible strict relation facts| ≥ r_min`").
- **Level freeze (end of MS3):** every calibrated parameter receives a number computed from development books only (or, for `δ*`, an owner choice made before any test-book contestant call), with a provenance record; every provisional parameter is confirmed at its plan starting value (provenance: the frozen form registry) unless a measured value replaces it (simplification 2026-09-25).
- **Pre-freeze contexts (amendment 2026-09-25; renamed 2026-09-26).** MS2 and MS3 run before the level freeze and read provisional levels. After the MS1 approval they run in a named pre-freeze context of the production profile (`prefreeze:PILOT`, `prefreeze:DEV`), on development books only. Only that context's listed provisional parameters are readable (plan starting values, or recorded pre-freeze values for calibrated ones); every run records its stage, partition, the values read and the registry and form-freeze identity, and a book used there is a development book that may never enter the test split. The level freeze can take a value from a pre-freeze development artifact by path and hash. Test-book acquisition, test cards, releases and scores read frozen levels only.
- **Invariant:** no numeric level changes after the first test-book artifact exists. Any change is a new benchmark version with a written reason. The prohibition on relaxing thresholds covers yield shortfalls as well as model ranks.

### 17.4 Human effort by milestone

MS1 protocol authoring and the owner's approval (one time); the readiness track: one approval per book plus terms recording (per book); MS3 approving `δ*` and the sizing decision; MS5 the publish or stop decision. Nothing per item, per model, or per cell.

### 17.5 Preregistered gates (replace the original success criteria)

Levels marked `[0b]` are named free parameters set at the level freeze (end of MS3); the values shown are starting points.

| ID | Type | Metric | Threshold rule | Data |
|---|---|---|---|---|
| **G1** zero per-item human labour | blocking | `human_touch_rate` = items with any recorded human content intervention / `N` | exactly 0.000 | pipeline audit log |
| **G2** yield | modulating | `N`, `F̄`, `min_g n_g` | `N ≥ N*` and `F̄ ≥ 8 [0b]` and at least `G_min = 6` test groups each with ≥ 10 items | Section 8.7 output, before any model runs |
| **G6** ranking stability | modulating | minimum over judge families of Kendall `τ_b` between the headline ordering and the leave-one-judge-family-out ordering, recomputed from stored votes (component 07, 14.5) | `τ_b ≥ 0.80`; requires `M ≥ 4` and ≥ 3 judge families | model run (`score-report`) |
| **G7** separation | blocking | the all-pairs Track A comparison of the headline `H` (IM p-values, BH at `q = 0.10`; component 07, 14.4) | at least one eligible model pair is separated with `|d̄| ≥ δ* [0b]` and, for that pair, `δ* ≥ 1.1 × MDD_floor(G, ρ̂, σ̂_d)` computed from the run (D21) | all `N`, common denominator, `G ≥ G_min` (`score-report`) |
| **G8** context effect | modulating | the pooled context effect `ΔH = H_B − H_A` averaged over models (11.7), IM interval over group means | passes if the CI excludes 0 (effect) **or** lies entirely within `±δ*` (informative null); fails only if it straddles `±δ*` | all `N` (`score-report`) |
| **G9** marginal cost | informational (never passes or fails) | `c_item`, `c_book`, `c_model`; projected V1 cost at 2,500 items, 60 books, `M` models | none: the projection is reported (no `X_V1`, D2) | ledger |

> **Owner amendment, 2026-09-23 (D2, implemented): G9 is informational.** There is no `X_V1` ceiling, so G9 reports
> the projection and never passes or fails. The 17.6 rule "SCALE when at least 3 of the 4 modulating gates pass" is
> applied literally with a G9 that cannot pass: SCALE needs G2, G6 and G8 all to pass. **Decision D11 (2026-09-23):**
> the literal, conservative rule is kept; "at least 2 of the 3 remaining" was rejected (BUILD_STATUS C38).

Deleted from the original criteria: "at least one interpretable research finding" (unfalsifiable disjunction) and "a corpus can be constructed with no per-item annotation" (tautological; replaced by G1, which audits the constraint). Added diagnostic, not a gate: the ceiling-effect flag (14.6), computed by `score-report`. All gate inputs are read from the score report by `gates-evaluate --scores`; none is assembled by hand (2026-09-25). G3–G5 (evaluator validation on Track C) were deleted on 2026-09-26; the numbers G1, G2, G6–G9 are kept so earlier records stay readable.

### 17.6 Scale-or-stop decision table

| Blocking (G1, G7) | Modulating (G2, G6, G8, G9) | Decision |
|---|---|---|
| all pass | ≥ 3 of 4 pass (G9 cannot pass, so G2, G6 and G8 all pass; D11) | **SCALE** to V1 (2,000–3,000 items, 40–80 books, three-way split) |
| all pass | ≤ 2 of 4 pass | **SCALE-REDUCED**: fix the failing modulator first. G2 → more books at constant items per book. G6 → freeze one judge configuration and publish single-judge with the instability reported. G8 → Track B becomes a V1 analysis, not a V1 track requirement. G9 is informational and has no remediation; its projected cost is reported with the decision (D2) |
| G7 fails, ceiling-effect flag set | any | **SCALE-REDUCED with widened eligibility**: the instrument works, the sample is over-selected; do not conclude models are equivalent |
| G7 fails, no ceiling effect, `δ* ≥ 1.1 × MDD_floor` for every eligible pair | any | **STOP**: models do not separate at a difference the design could detect; the floor is printed beside the null |
| G1 fails | any | **STOP immediately**: the zero-annotation premise is violated |

### 17.7 Rejected alternatives

Enlarging the pilot to 800–1,000 items (items are the wrong axis); a human spot-check phase; any control suite, full-factorial or reduced (component 08, 12.5); a Bayesian hierarchical comparison model; a staffing Gantt; `Compile@k` or best-of-k in the pilot; interim analyses with early stopping (alpha spending for a run that costs days); running Track B on only a subset of models (saving small at `M ≤ 6`, and Track B pairing on all models is core structure; revisited only if the ledger shows judging binds).
