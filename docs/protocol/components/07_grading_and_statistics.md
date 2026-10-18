# Component 07 — Grading, metrics, and the statistical analysis plan

Part of *HistMathBench Zero-Annotation Project Plan v2*. Canonical entry point: [`../HistMathBench_Zero_Annotation_Project_Plan_v2.md`](../HistMathBench_Zero_Annotation_Project_Plan_v2.md). This file owns **Section 11** (grading and metrics) and **Section 14** (statistical analysis plan). Global notation is in the top-level plan, Section 3.4.

> **Simplification 2026-09-25** (`reports/simplification_audit_and_execution_20260925.md`). This file was rewritten to
> one scoring path: a single item score with no weights, caps, optimistic bound or not-scored state; one headline
> (the group-balanced mean) with one interval method (the Ibragimov–Müller group t); three mandatory diagnostic
> families. The class-weighted item score with `cap_Y`, the `[pessimistic, optimistic]` interval and `W` rule, the
> list-wise drop, `H_item`/`H_fact` as co-headlines, the composite-weight family, `Semantic@Valid`/`CommonValid`,
> the decomposition, `S^P`, precision, `LabelRecall_equiv`, the sign-flip companion, the item bootstrap, `σ_eval`
> suppression and the gate-sensitivity re-score were removed. Section numbers are kept; removed sections say so.

> **Checkpoint simplification 2026-09-26.** The Track C control suite (CP3) and evaluator validation (CP5, V1–V11)
> were deleted, with gates G3–G5. For this file: the Track C known-answer precision (11.5) and the Track C secondary
> analyses (14.7) are gone; the evaluator numbers that enter decisions are the drift probe and G6 only (14.5); the
> judge reliability numbers of 11.4 are reported from the final run itself; pilot sizing happens in MS3 with
> `dev-sizing` (formerly CP7, 14.3).

| Component card | |
|---|---|
| Purpose | Turn stored verdicts and outcome codes into one headline score with one interval, three diagnostic families, and the gate inputs, with deterministic code and no free parameters. |
| Inputs | Frozen cards, verdict files, outcome codes, split groups. |
| Mechanism | Recall over the eligible strict facts; indeterminate counts as not recovered; group-balanced mean; Ibragimov–Müller group t. |
| Outputs | Per-cell scores; per model and track: headline and interval, validity, per-class diagnostics; the all-pairs partial order; the G2/G6/G7/G8 evidence (`score-report`). |
| Gates | Supplies G2, G6, G7, G8 (Section 17.5). |
| Retained artifacts | Score files, the score report, `metric_hash`. |
| Cost tier | CPU only. |
| Human effort | None. |

---

## 11. Grading and metrics

### 11.0 Symbols

| Symbol | Definition |
|---|---|
| `i` | a scored item of the frozen eligible set of one split (test for reported numbers) |
| `g(i)` | its split group (Section 13.1); `G` = number of groups; `n_g` = items in group `g` |
| `m`, `t` | contestant model, track ∈ {A, B}; a cell is `(i, m, t)` |
| `F_i` | the item's eligible facts (Section 8.4); classes E, R, L, all strict (asc-2.0.0: layout is not a card class) |
| `o(i,m,t)` | outcome code (Section 9.5); `g_i = 1[o = OK]` |
| `v(f)` | verdict ∈ {PRESENT, ABSENT, CONTRADICTED, INDETERMINATE, EXEMPT} (Section 10.4) |

**Invariant.** Every denominator is a function of the frozen card and the outcome code only, never of the model, the
judge or the other models in the run. Every eligible item is scored for every model; nothing is dropped.

### 11.1 Fact universe

The fact universe of an item is its eligible minimal generating set `F_i` (Section 8.2). Category membership comes
from the ontology (Section 7.1), so entity existence is scored once, in E.

### 11.2 Verdict handling (fixed rules)

`EXEMPT` facts leave both the numerator and the denominator (a deterministic consequence of an entity miss already
counted). `ABSENT` and `CONTRADICTED` count the same (not recovered); `CONTRADICTED` is also counted for Q5.
`INDETERMINATE` (the evaluator could not decide, including a failed judge call) counts as **not recovered**. There is
no optimistic bound, no uncertainty width and no not-scored state: a cell with an evaluator failure is scored like
any other and flagged (`evaluator_failure`); the failure rate is reported per model (Section 11.4). Counting an
undecided fact as a miss never rewards output that confuses the evaluator and keeps every denominator fixed.

### 11.3 Item score (the one-page specification)

```text
scorable_i = #{ f ∈ F_i : f is strict (E, R, L) and v(f) ≠ EXEMPT }
S_i(m,t)   = g_i(m,t) · #{ f scorable : v(f) = PRESENT } / scorable_i
```

Plain recall over the item's strict facts: every fact weighs the same, there is no free parameter. Layout facts
never entered `S_i` (coarse position statements, combinatorially numerous: one cube specification produced 180 layout
facts out of 330, BUILD_STATUS C27); since asc-2.0.0 (2026-09-26) they are not on the card at all, so no layout
diagnostic exists and reflection is not scored (disclosed).

Deliberate amplification: one missing point costs its entity fact, its relations and its label. Destroying more
mathematics costs more; this is stated on the benchmark card.

**Worked examples** (a card with 3 point facts and 2 incidence facts, 5 strict facts; `tests/w4/test_grading.py`,
`test_case_*`):

| case | verdicts | `S_i` |
|---|---|---|
| 1 all facts present | 5 PRESENT | 5/5 = 1.0 |
| 2 one absent fact | 4 PRESENT, 1 ABSENT | 4/5 = 0.8 |
| 3 one contradicted fact | 4 PRESENT, 1 CONTRADICTED | 4/5 = 0.8 (and counted for Q5) |
| 4 one indeterminate fact | 4 PRESENT, 1 INDETERMINATE | 4/5 = 0.8 |
| 5 compile or extraction failure | any outcome ≠ OK | 0, with the card's own counts |
| 6 no eligible facts | — | the item is not in the scored set (the §8.7 gate excludes it) |

An EXEMPT fact shrinks both sides: with `e1` ABSENT and `on(p1, c)` EXEMPT, 2 PRESENT of 4 scorable give 0.5, not 2/5.

### 11.4 Mandatory diagnostic families (per model and track)

1. **Validity:** the outcome-code distribution, `Valid@1` and the derived rates of Section 9.5, over all eligible items.
2. **Per-class recall:** `R^c` for c ∈ {E, R, L} (item-macro mean of `g_i · PRESENT/scorable` in the class, over
   items with at least `m_c` scorable facts in the class), with the contradicted and indeterminate shares of each
   class; the label error counts (transcription, attachment, missing; labels are matched strictly, Section 7.5);
   `Q5(m) = P(item has ≥ 1 CONTRADICTED relation | o = OK and R^E ≥ 0.8)`, the plausible-but-wrong rate; and the
   evaluator-failure rate (share of valid cells in which a judge call failed).

The extras family (`MaterialExtraItemRate`) was removed 2026-09-26 with the extras channel (component 06).
Everything else is optional and never blocks a run.

### 11.5 Precision — removed 2026-09-25

Candidate-fact precision is not computed on historical items (the extras channel, its only precision-side number,
was removed 2026-09-26). Known-answer precision/recall/F1 was a Track C quantity; it was removed with the control
suite on 2026-09-26.

### 11.6 Compile-gated performance

`H` (Section 11.8) already counts invalid outputs as 0, so it is the compile-gated number. `Valid@1` is reported
beside it. The decomposition identity and `Semantic@Valid`/`Semantic@CommonValid` were removed.

### 11.7 Context effect (Track B minus Track A)

Both tracks share the card (Section 7.3). Per model: `ΔH = H_B − H_A` and `ΔValid@1`. The G8 input is the pooled
effect: per item, the difference `S_i(m,B) − S_i(m,A)` averaged over models; its group-balanced mean with the IM
interval (Section 14.2). The context effect therefore includes compile effects; `ΔValid@1` shows how much.

### 11.8 Aggregation: the headline

```text
H(m,t) = (1/G) Σ_g (1/n_g) Σ_{i ∈ g} S_i(m,t)          # mean over split groups of the mean item score
```

Groups count equally because the group, not the item, is the sampling unit (Section 14.1); the interval of
Section 14.2 is exact for this quantity. With equal group sizes `H` equals the plain item mean. Worked example
(`test_case_7`): groups with item scores {1.0, 0.6, 0.8}, {0.4}, {0.6, 0.6} give group means 0.8, 0.4, 0.6 and
`H = 0.6`, although the plain item mean is 0.667.

The selection funnel and the eligibility rate `e` (Section 13.3) accompany every headline in the claim sentence.

### 11.9 Composite weights — removed 2026-09-25

There is no class-weighted composite, so there is no composite family and no flip fraction.

### 11.10 Cost metrics

From the ledger (Section 16): cost per scored item per model and per marginal book, inputs to G9 (informational).

### 11.11 Slice reporting and suppression

A slice is published with scores only if `n_items ≥ 30`, `n_groups ≥ 3` and `n_facts ≥ 150`; otherwise counts only.
Language, decade and publisher slices are book-level constants, labelled descriptive.

### 11.12 Edge cases

| Situation | Behaviour |
|---|---|
| no eligible facts | item not in the scored set (Section 8.7); count published |
| blank render | `o = V_EMPTY`, `S_i = 0`, no judge call |
| every fact indeterminate | `S_i = 0` from those facts; the indeterminate share and failure rate are reported |
| compile failure, unsafe output, timeout, protocol violation | `S_i = 0` under its own outcome code |
| judges disagree after three votes, or a judge call fails | INDETERMINATE, counted as not recovered |
| a class with fewer than `m_c` scorable facts | omitted from that class's diagnostic row; still inside `S_i` if strict |
| item in the unscored pool | never scored; reported in the funnel |

### 11.13 Double-counting audit

(a) `inside(O,c)` is dropped when `center_of(O,c)` is kept. (b) `collinear` sub-tuples and argument permutations
collapse under canonicalization. (c) `(text, entity)` label pairs are unique. (d) Entity existence is retained on
purpose (documented amplification). (e) No class weights exist. (f) Facts flagged derived never enter a denominator.

### 11.14 Required report per model and track (`score-report`)

The headline `H` with its IM interval, `G` and the group sizes; the three diagnostic families of Section 11.4; the
secondary partial order (Section 14.4); the context effect when both tracks are run; the leave-one-judge-family-out
orderings (G6); and the `gate_evidence` block that `gates-evaluate --scores` reads. The report is always for one
split (test by default).

---

## 14. Statistical analysis plan

### 14.1 Unit of inference

Items nested in books are not independent, and editions or shared plates make books not independent either. The
cluster is the **split group** `g` (Section 13.1). Every interval and test in the report is computed on group means.

### 14.2 One method: the Ibragimov–Müller group t

For any per-item quantity `x_i` (a model's `S_i`, or a paired difference `S_i(m₁) − S_i(m₂)`):

```text
x̄_g  = mean of x_i over the items of group g,  g = 1..G
est  = (1/G) Σ_g x̄_g
CI   = est ± t_{0.975, G−1} · sd(x̄_1..x̄_G) / √G ;   p from the same t distribution
```

This is valid with a small number of groups and conservative at the 5 percent level under heterogeneous group
variances ([Ibragimov & Müller, JBES 28(4), 2010](https://www.princeton.edu/~umueller/tstat.pdf)). Because the paired
difference uses the same items and groups as the two headlines, its estimate equals `H(m₁) − H(m₂)`.
`G ≥ G_min = 6` test groups are required for an inferential claim; below that, point estimates and counts only.
Assumptions: groups are independent; each group mean is roughly normal or the groups are symmetric enough for the
t reference (a small-G method cannot check this); groups count equally. Failure modes: very few groups give very
wide intervals (honest, not wrong); one dominant group is prevented by the 25 percent book cap and the collapse guard.

**Worked example** (`test_case_8`): on the six items above, model b scores 0.2 below model a on every item of the
first group and the same elsewhere, so the group differences are (0.2, 0, 0): estimate 0.0667, sd 0.1155, CI
`0.0667 ± 4.303 · 0.1155/√3 = [−0.220, 0.354]`. Treating the six items as independent would give a far narrower,
wrong interval.

Prohibited as the primary method at pilot scale: the pairs cluster bootstrap (over-rejects with few clusters,
[Cameron, Gelbach & Miller 2008](https://www.nber.org/system/files/working_papers/t0344/t0344.pdf)), mixed-effects
models, bootstrap rank intervals, and any item-level independent-observation interval.

### 14.3 Minimum detectable difference (design check)

With `ρ` the intra-group correlation of the item-level paired difference (one-way ANOVA estimator, Section 13.7) and
`σ_d` its SD:

```text
MDD_floor = (t_{0.975, G−1} + t_{0.80, G−1}) · σ_d · √(ρ / G)
```

`score-report` computes `ρ̂`, `σ̂_d` and `MDD_floor` for every Track A model pair from the run itself; a pair counts
toward G7 only if `δ* ≥ 1.1 × MDD_floor` for that pair. The same formula (with `N`) sizes the pilot in MS3
(`dev-sizing`, Section 13.7).

### 14.4 The partial order and separation (G7)

- All `M(M−1)/2` pairs are tested with the IM method of 14.2 and Benjamini–Hochberg control at `q = 0.10` over the
  full family (an uncomputable pair stays in the family with p = 1). The published result is a partial order in
  which unseparated models share a rank band. A pair is separated iff BH rejects. No total-order leaderboard is
  published at pilot scale.
- **G7 (separation; owner decision D21, 2026-09-26)** passes when at least one eligible Track A model pair is
  separated with `|d̄| ≥ δ*` (and `δ* ≥ 1.1 × MDD_floor` for that pair, 14.3). No model pair is named in advance;
  a pass supports "the benchmark separates at least one model pair", not a claim about a particular pair beyond
  what the partial order reports.

### 14.5 Evaluator numbers that enter decisions

The drift probe (Section 10.9) and the leave-one-judge-family-out ranking stability (G6: minimum Kendall `τ_b`
between the full ordering and each ordering recomputed from the stored votes without one judge family, M ≥ 4
models, ≥ 3 judge families). The indeterminate shares and the evaluator-failure rate of Section 11.4 are reported
from the same run and do not gate. The control-based numbers (V2, V4, V6, V7 on Track C) and the real-item
relations (V9) were removed on 2026-09-26. `σ_eval` rank suppression and the interval-only rule were removed.

### 14.6 Guards against easy-item selection

The headline is always reported on the frozen eligible set with the funnel and the eligibility rate. The
**ceiling-effect flag** is computed by `score-report`: the top model's `H > 0.90` and the mean within-group SD of its
item scores `< 0.10`. It decides between SCALE-REDUCED-with-widened-eligibility and STOP when G7 fails (Section 17.6).
The gate-sensitivity re-score was removed (the consensus rule has no threshold to shift); leave-one-builder-family-out
(`rederive-report`) remains an optional robustness output.

### 14.7 Exploratory analyses (no inference, no stars)

Score by raw degradation statistic; score by `touching_cc`; score and context effect by prominence tercile; language,
decade and publisher tables with group counts; the T3 text-only probe (Section 13.6). (The Track C secondary analyses were removed on 2026-09-26.) The post-stratified `S_adj` and the selection envelope of Section 13.3 are optional.
