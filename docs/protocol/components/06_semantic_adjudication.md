# Component 06 — Semantic adjudication: deterministic extraction, model judges, verdicts, escalation, extras

Part of *HistMathBench Zero-Annotation Project Plan v2*. Canonical entry point: [`../HistMathBench_Zero_Annotation_Project_Plan_v2.md`](../HistMathBench_Zero_Annotation_Project_Plan_v2.md). This file owns **Section 10**. Global notation is in the top-level plan, Section 3.4.

| Component card | |
|---|---|
| Purpose | Decide, for each eligible card fact and each valid candidate, whether the candidate asserts it, with no human; judge reliability is reported from the final run itself. |
| Inputs | Frozen card (Section 8.5), compiled PDF and canonical PNG (Section 9), canonical crop (for source-support questions only). |
| Mechanism | Deterministic geometry and text extraction from the compiled PDF for `geo` facts; two model judges from distinct families on every cell for `model` facts; a fixed majority rule, with a third judge only on facts the first two do not decide. |
| Outputs | Per-fact verdicts, candidate-fact record (CGX), bindings. |
| Gates | None of its own. The judge drift probe (10.9) can invalidate a scoring window. Extractor correctness is covered by offline software tests, including `hmb cgx-roundtrip` (component 08, §12), which is not a gate. |
| Retained artifacts | CGX record, every judge call's raw JSON, verdict file, drift-probe results. |
| Cost tier | Per item per model per track: CPU for extraction; 2 judge calls per chunk of ≤ 25 `model`-class facts, a third only for undecided facts, plus Q3 when an entity is ambiguous. |
| Lifecycle | Per cell; extractor and judge configuration one time per evaluator version. |
| Human effort | None. |
| Dependencies | Section 7.1 (decidability), Section 8.5 (fact record), Section 9 (artifacts), Section 11 (verdict flow), component 08 §12 (evaluator assurance), Section 15 (judge config hash, scoring window). |
| Downstream decisions | All recall metrics; judge-family stability analyses. |

> **Simplification 2026-09-25** (`reports/simplification_audit_and_execution_20260925.md`): the log-likelihood-ratio
> vote aggregation with fitted η/θ/ρ and the log 9 band, the in-band planted controls, Q4 label equivalence, the 3 %
> extractor audit, the geo-share extractor cut, the indeterminacy cap and the legibility not-scored rule were removed.
> Every question uses one majority rule (10.5). Every eligible cell is scored (component 07, Section 11.2).

> **Streamlining 2026-09-26** (`reports/answer_key_judge_streamlining_execution_20260926.md`; supersedes the text
> below where they differ). Q2 source support and the extras channel (10.7, the precision-side vocabulary of 10.4,
> `MaterialExtraItemRate`) are removed: they never entered the score and drove about 87 % of judge calls, and with a
> two-writer card of measured coverage well below 1 a "material extra" is mostly card incompleteness. Judge answers
> carry no `evidence` string; replies are read answer by answer (a malformed or cut-off entry costs only its own
> statement; a missing answer is not a vote); a slot is attempted once and, when it yields no usable answer, goes
> once to a judge of another family (no identical-prompt retries). Cards carry no layout facts (component 04), so
> no layout fact is judged. CONTRADICTED is kept: it is deterministic, costs no call and feeds Q5.

> **Checkpoint simplification 2026-09-26** (supersedes the text below where they differ). The Track C control suite
> (CP3) and evaluator validation on VAL (CP5, V1–V11) were deleted. The extractor has no validation gate: its
> correctness is covered by offline software tests, including the round trip `hmb cgx-roundtrip` (formerly "V1x DEV":
> generated known-geometry TikZ through the sandbox, CGX compared with the generating geometry; no model call, not a
> gate, sets no level). Judge reliability is reported from the final run (INDETERMINATE and evaluator-failure rates,
> per-class diagnostics, leave-one-judge-family-out G6). A bias shared by all three judge families on `model`-class
> predicates is not measured against known truth (disclosed limitation). The drift-probe triples now come from
> development-set judged cells (10.9).

---

## 10. Semantic adjudication

### 10.1 Routing by decidability

Every card fact carries `decidability ∈ {geo, model}` from the ontology (Section 7.1). `geo` facts are decided by the deterministic extractor of Section 10.2 without a model call; `model` facts (curve identity, solids, marked equality, semantic label equivalence, `region`) are decided by model judges under Section 10.5. The full extractor is always used; a failing offline extractor test (including `hmb cgx-roundtrip`) is an evaluator defect to fix, not a routing change. The smoke set reports the `geo` share for cost planning only.

### 10.2 Deterministic candidate extraction (the CGX record)

Extraction works on the **compiled PDF**, not the TikZ source: source parsing cannot resolve `\foreach`, `calc`, `intersections`, macros or `pic` without running TeX, and the PDF is the evaluated TeX.

1. **Paths.** Parse the content stream into line segments and cubic Béziers with the current transformation applied. PGF emits circles and arcs as cubic Bézier quadrants; fit a circle to each closed four-Bézier run and accept as `circle` when the radial residual is below tolerance; otherwise keep a generic curve (routes to `model`).
2. **Labels.** Extract text with per-glyph bounding boxes (LuaLaTeX with Unicode fonts produces the ToUnicode maps this needs; confirmed in the Phase 0 smoke test, Section 9.8).
3. **Candidate facts.** Over the extracted primitives, emit the positively asserted facts only: existence per primitive, endpoints, and every `geo` relation that holds within tolerance. Canonicalize exactly as Section 8.2 (collinear sets, spanning trees for parallel classes, transitive reduction of incidence) so the candidate universe does not inflate with primitive count.
4. **Tolerances**, relative to `D` = render bounding-box diagonal and `r` = circle radius: incidence `ε_on = 0.010·D`; tangency `|dist(center, line) − r| ≤ 0.015·r` and exactly one intersection within `ε_on`; parallel ≤ 1.5°; perpendicular 90° ± 1.5°; collinear `ε_on`. Scale-relative tolerances make invariance to uniform scaling automatic. Values are preregistered (their confirmation on Track C, V4, was removed on 2026-09-26).

**Assurance, no gate.** The Track C validation gate (V1x ≥ 0.98 on controls) and the real-item relations of the former Section 12.6 were removed on 2026-09-26. The extractor's correctness is covered by offline software tests, including the round trip `hmb cgx-roundtrip`: generated known-geometry TikZ is compiled in the sandbox and the CGX record is compared with the geometry that produced it (component 08, §12). It is engineering evidence, not a gate, and supports no benchmark claim. There is no real-item extractor audit (disclosed limitation). The extractor is versioned into `judge_config_hash`.

The CGX record is stored per cell and is the **candidate-fact universe**: a function of the submission and a frozen extractor version, reproducible bit for bit.

### 10.3 Card-to-candidate alignment and the four judge questions

**Alignment** (deterministic, before any relation is checked): anchor on labels (a card point labelled `A` binds to the extracted point nearest the extracted glyph `A` within `0.06·D`); propagate along structure (a circle with a bound centre, a segment with two bound endpoints); resolve the remainder by exact subgraph matching on type and incidence signature when ≤ 15 entities remain unmatched, otherwise Hungarian assignment on structural similarity; if the matching is not unique, emit `ALIGNMENT_AMBIGUOUS` and route the affected facts to Q3. Relations are judged on structurally aligned entities, so a mislabelled point loses label credit but does not cascade into relation misses; the strict variant (labels must match for a relation to count) is a published sensitivity. Bindings are stored for audit.

| Q | Question | Inputs | Deliberately excluded | When |
|---|---|---|---|---|
| **Q1 RECALL** | Is fact `f` asserted by this render? | canonical candidate render; fact text; the rendering-convention note (Section 7.6) | **the crop**, the context, the abstention list, the TikZ source, other facts' verdicts | every valid cell, `model`-class facts only |
| **Q3 ALIGNMENT** | Which rendered object corresponds to card entity `e`? | candidate render; card entity list with types | crop, relations | only on `ALIGNMENT_AMBIGUOUS` |

The crop is withheld from Q1 because every card fact was derived from the crop; showing it supplies the answer to a different question. Q1 input is one image, which also lowers cost.

### 10.4 Verdict vocabulary

Recall side, per card fact, exactly one value:

| Value | Meaning | Source |
|---|---|---|
| `PRESENT` | the render asserts the fact | extractor (geo) or aggregated judges (model) |
| `ABSENT` | the render does not assert it and asserts nothing incompatible | same |
| `CONTRADICTED` | the render asserts something incompatible under table E (e.g. tangent → secant) | same |
| `INDETERMINATE` | no two judges agreed after the third vote, or a judge call failed; counts as not recovered | judges only |
| `EXEMPT` | argument entity absent from the render and already counted as an entity miss by the extractor (its entity fact is geo-decided and `ABSENT`), or anchor-B fact with a non-unique binding | deterministic |

There is no precision-side vocabulary (removed 2026-09-26 with the extras channel). `CONTRADICTED` and `ABSENT` score identically (Section 11.2) but are reported separately because the contradicted-relation rate is the operational answer to "plausible but mathematically changed" (top-level Section 2.3, Q5). There is no separate abstention channel; abstention is the value `INDETERMINATE`.

### 10.5 Vote aggregation: one majority rule

For every judge question (Q1, Q3), a decision needs **at least two judges from distinct families giving the same
answer, and more of them than the opposite answer**. With at most three votes this is "the first two agree, or two
out of three":

```text
ask two judges (distinct families)
if both give the same answer            -> that answer
else (they disagree, or one answer is missing)
    ask a third judge (a third family)  -> the answer given by two of the three, if any
otherwise                               -> INDETERMINATE (Q1) / unresolved (Q3)
```

Q1: yes → `PRESENT`; no → `ABSENT`, or `CONTRADICTED` when the render asserts an E-incompatible fact. There are no fitted judge reliabilities, no weights and no band. The end-to-end
validation on controls (V2, V4, V6, V7) was removed on 2026-09-26; judge reliability is reported from the final run
(INDETERMINATE and evaluator-failure rates, per-class diagnostics, leave-one-judge-family-out G6), and a bias shared
by all three judge families is not measured against known truth (disclosed limitation). Judge self-reported
confidence is never used.

### 10.6 Judge protocol, escalation, stop condition, cost

1. **Two judges from distinct families evaluate every valid cell** that has at least one `model`-class fact (Q1 over
   all such facts, in chunks of ≤ 25). Cells with only `geo` facts and no ambiguous entity need no judge call.
2. **A third judge (third family) is asked only about the facts the first two did not decide.** No fact is voted on
   a fourth time; no cell is re-judged after freeze.
3. Q3 follows the same rule on the entities routed to it.

Expected calls per cell with `n` model-class facts: `2·⌈n/25⌉ + P₃·⌈n_D/25⌉ + (Q3: 0–3)`,
where `P₃` and `n_D` are the probability and number of undecided facts after two votes. All quantities are metered
by the ledger (Section 16); prices are never assumed in the plan text. Judge decoding is temperature 0 with a pinned
configuration. (The repeatability measurement V6 was removed on 2026-09-26.)

### 10.7 Extras — removed 2026-09-26

The excusal ladder and the Q2 source-support question are removed (see the streamlining note above). A render's
content beyond the card is not judged. Card coverage is not measured (Section 8.6; `R_card(s)` was removed on 2026-09-26).

### 10.8 Operational cases (preregistered)

| Case | Rule |
|---|---|
| Malformed or non-schema judge JSON | answer-level reading: valid entries count, invalid or missing ones are not votes (third judge); a reply with no usable answer goes once to a judge of another family (no identical-prompt retry); if both fail, the undecided facts of that call are `INDETERMINATE` and the cell records `evaluator_failure = judge_failure`; the cell is still scored and the failure rate is published |
| Judge refusal | same as malformed, recorded as `judge_refusal` |
| Fewer answers returned than facts sent | a missing answer is not a vote: the third judge is asked (10.5) |
| Fact list too long | deterministic chunks of ≤ 25 facts by `fact_id` order; chunking is part of the frozen judge config |
| Blank render (`V_EMPTY`) | never reaches the judge; handled by the outcome code (Section 9.5) |
| Q3 fails | the ambiguous entities stay ambiguous: their facts are `INDETERMINATE`; every extractor verdict stands |
| Zero eligible facts in a class | the cell contributes nothing to that class; count published |

### 10.9 Family independence, self-judging, and drift

- **Builder–judge disjointness.** Primary configuration: `families(card builders) ∩ families(judges) = ∅`. With fewer than four accessible families, rotate per item: the family that built item `i`'s card does not judge item `i`. A declared secondary configuration deliberately violates disjointness so that the confirmation effect is measured, not assumed away.
- **Judge–contestant.** No judge family judges its own family's output in the primary configuration. If a family must appear on both sides, the leaderboard is published with and without that family's judge and the rank change reported.
- **Leave-one-judge-family-out** requires at least three judge families; this is a requirement on the model-access plan, not an optional analysis. It is recomputed from the stored votes with no new calls (the remaining votes decide by majority, a tie is `INDETERMINATE`) and is the G6 ranking-stability input. Leave-one-builder-family-out is in Section 13.5.
- **Judge drift.** Scores are produced inside one declared **scoring window**. A frozen set of 100 (card fact, candidate render, reference verdict) triples drawn from development-set judged cells, with the verdict recorded there as the reference, is written by `drift-triples --seed N` in the pre-freeze DEV context (`drift_probe_n_triples` = 100 judge-decided facts
drawn from the MS3 probe cells) and stored immutably at `manifests/drift_probe/triples.json` (it was drawn from the
Track C validation partition before 2026-09-26) and judged at window open and at window close; agreement with the recorded development verdicts must be ≥ `κ_min` at both ends and open-versus-close agreement ≥ `α_min` (both form constants, 0.8). Failure invalidates the window: re-score or bump the version. Cost about 200 calls per window. Where providers offer dated snapshots, they are pinned.

### 10.10 Rejected alternatives

Log-likelihood-ratio aggregation with fitted judge sensitivity/specificity and an error-correlation deflation (removed 2026-09-25: three calibrated inputs, a fitting pipeline and planted controls to estimate them, for a decision that coincides with the majority rule whenever two agreeing votes are strong enough to leave the band, which the V7 agreement bar requires in practice; where the two rules would differ, the majority rule asks the third judge instead of weighting by estimates transferred from synthetic controls; reliability is validated end to end instead); running the card builders on every candidate render to obtain a symmetric candidate card (multiplies the dominant cost by the builder count and still leaves a stochastic denominator; retained only implicitly through Q3); unsupervised latent-class estimation over judge votes (reliability is observed on controls); calibrating self-reported confidence (no longer used); a Bayesian hierarchical model of item, model and judge effects (not decision-relevant at pilot scale); symbolic re-derivation of constructions (unnecessary: the card asserts a fact set, not a theorem); giving the judge the TikZ source (reintroduces style bias and lets the judge credit semantics never rendered); self-consistency sampling from one judge (correlated within family); scoring judge explanations; any per-item human spot-check. (The control-based validation that several entries above rely on was removed on 2026-09-26; see the checkpoint simplification note.)
