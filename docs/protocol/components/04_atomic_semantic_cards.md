# Component 04 — Atomic Semantic Cards: ontology, generation, alignment, normalization, consensus, coverage, eligibility

Part of *HistMathBench Zero-Annotation Project Plan v2*. Canonical entry point: [`../HistMathBench_Zero_Annotation_Project_Plan_v2.md`](../HistMathBench_Zero_Annotation_Project_Plan_v2.md). This file owns **Section 7** (ontology and machine generation) and **Section 8** (alignment, logical normalization, consensus, coverage, abstention, eligibility). Global notation is in the top-level plan, Section 3.4.

| Component card | |
|---|---|
| Purpose | Produce, for each crop, one frozen, canonical, minimal set of testable facts, without any human reading a diagram. Card precision and coverage are not measured (disclosed limitation, 8.6). |
| Inputs | Canonical crop, page OCR, Track B context (label disambiguation only), the closed ontology. |
| Mechanism | Two writers in two model families (amended 2026-09-26); line-oriented generation with hard caps; deterministic entity alignment; closure and minimal generating set; chunked cross-verification (each writer checks every union fact in calls of at most 20 statements; its own facts count unless it vetoes them); one fixed rule (both writers support, none rejects); numeric item gates. |
| Outputs | Frozen card (`facts`), per-fact votes with families, item eligibility. |
| Gates | Fact eligibility (Section 8.4); item gates (Section 8.7). |
| Retained artifacts | Raw builder outputs, union card, verification votes, alignment log, MGS, eligibility decisions, all hashed under `asc_version`. |
| Cost tier | Per item, paid once: 2 staged generation calls (at most 4 with the single retries) plus ⌈statements / 20⌉ verification calls per writer; deterministic code otherwise. |
| Lifecycle | Per item, one time (cards are frozen). Ontology and thresholds one time per evaluator version. |
| Human effort | None per item. One-time ontology and rule authoring. |
| Dependencies | Section 6 (inputs), Section 10 (judge consumes card and decidability), Section 11 (denominators), Section 13 (funnel, split groups). |
| Downstream decisions | Every recall denominator; item eligibility; what the benchmark claims coverage is. |

> **Simplification 2026-09-25** (`reports/simplification_audit_and_execution_20260925.md`): the family-clustered
> log-likelihood-ratio consensus (α/β per builder and category, prior π_k, τ_k from a Clopper–Pearson floor on CAL,
> a governing rule per category) is replaced by the plan's own declared fallback rule, now the only rule (8.4); the
> planted controls of the verification pass are removed (8.3); the alignment weights are form constants (8.1); the
> structure-coverage proxy SC is recorded, not a gate (8.6, 8.7).

> **Streamlining 2026-09-26, `asc-2.0.0`** (`reports/answer_key_judge_streamlining_execution_20260926.md`;
> supersedes the text below where they differ). (1) Cards carry the strict classes E, R, L only; layout (Y) stays
> in the predicate table for the extractor and Γ but is never on a card. (2) Writers answer in JSON Lines: entity
> lines (type, label, parameter, anchor) and then fact lines (relations and `measure_label`); existence and label
> facts are derived from the entity lines; hard caps of 40 entity and 60 fact lines; every complete line survives
> a reply cut at `max_tokens`. (3) Verification is chunked: every writer checks every union fact in calls of at most 20 statements; a
> writer supports the closure of its own generation unless it votes false on it (self-veto), one `<id> <vote>` line
> each. (4) Consensus is the fixed two-writer rule: both writers support (generated or voted true), neither votes
> false, plus the structural conditions; `min_true_families`, LOBFO, card instability, SC and UCR are removed.
> (5) The model-call cache key is the request itself; `asc_version` no longer hashes gate levels.

> **Repair 2026-09-26** (`reports/answer_key_judge_repairs_execution_20260926.md`; supersedes (2) where they differ).
> Generation is two staged calls per writer: an entity call (entity lines only), then a fact call that is given the
> writer's own entity lines (fact lines only). A stage cut at `max_tokens` with no complete line is asked once more
> with a narrower request (at most half the cap); nothing else is retried. The caps and the verification chunking are
> unchanged. Card precision against Γ (12.3) is measured on facts that map to oracle entities, with the mapped share
> `M_card` reported and gated beside it.

> **Checkpoint simplification 2026-09-26** (supersedes the text below and the last sentence of the Repair callout
> where they differ). The Track C control suite (CP3) and evaluator validation (CP5, V1–V11) were deleted, so
> nothing in this component is measured against synthetic truth any more: card precision `P_card(s)`, `M_card`,
> coverage `R_card(s)`, `card_vs_oracle`, `card_map_coverage_floor` and the oracle-candidate ceiling are gone (8.4,
> 8.6); the item degradation ceiling `s*` and its exclusion `degradation_above_ceiling` are gone (8.7); the
> §12.5 mutation vocabulary no longer tests `allowed_variation` (7.6); the V10 solid-family criterion is gone (7.2).
> Card precision and coverage on historical items are a disclosed limitation, as they always were on real items.

---

## 7. Atomic Semantic Card ontology and machine generation

### 7.1 Closed predicate table (`asc-1.0.0`)

Builders may emit only these predicates (asc-2.0.0: the E, R and L rows; layout is extractor-internal). Columns: **arity**; **category** (the unique scoring class E/R/L/Y of Section 11); **class** (`strict` = mathematically load-bearing, `coarse` = layout); **anchor** (`A` = score freely, `B` = score only when all arguments are uniquely bound, Section 10.3); **decidability** (`geo` = decided from extracted geometry of the compiled candidate, `model` = decided by the judge; Section 10.1).

**Entities (category E, strict, anchor A).** `point(e)`, `segment(e)`, `line(e)`, `ray(e)`, `circle(e)`, `arc(e)`, `ellipse(e)`, `polygon(e, n)`, `region(e)`, `axis(e, o)` with `o ∈ {horizontal, vertical, oblique}`, `arrow(e)`, `angle_marker(e)`, `right_angle_marked(e)`, `length_tick(e)` — all `geo` except `region` (`model`); `curve(e, f)` with `f ∈ {line, parabola, hyperbola, ellipse_arc, periodic, monotone_other, unspecified}` (`model`); `solid(e, k)` with `k ∈ {cube, prism, pyramid, cylinder, cone, sphere, polyhedron_other}` (`model`).

**Relations (category R).**

| predicate | arity | class | anchor | decidability | note |
|---|---|---|---|---|---|
| `on(p, x)` | 2 | strict | A | geo | point incident to a 1-D entity or curve |
| `endpoint_of(p, x)` | 2 | strict | A | geo | `x ∈ {segment, ray, arc}` |
| `vertex_of(p, x)` | 2 | strict | A | geo | `x ∈ {polygon, angle_marker, solid}` |
| `edge_of(s, x)` | 2 | strict | A | geo | `x ∈ {polygon, solid}` |
| `face_of(g, s)` | 2 | strict | B | model | |
| `center_of(p, x)` | 2 | strict | A | geo | `x ∈ {circle, arc, ellipse}` |
| `arc_of(a, c)` | 2 | strict | A | geo | |
| `intersect_at(x, y, p)` | 3 | strict | A | geo | symmetric in `(x, y)` |
| `tangent_to(x, y, p)` | 3 | strict | A | geo | `p` may be `unknown` |
| `parallel(x, y)` | 2 | strict | A | geo | symmetric and transitive |
| `perpendicular(x, y)` | 2 | strict | A | geo | symmetric |
| `collinear(S)` | set ≥ 3 | strict | A | geo | one maximal set per class |
| `concyclic(S)` | set ≥ 4 | strict | B | geo | |
| `between(p, q, r)` | 3 | strict | A | geo | canonical `id(p) < id(r)` |
| `equal_length(x, y)` | 2 | strict | A | model | only when tick-marked or stated |
| `equal_angle(x, y)` | 2 | strict | A | model | only when marked or stated |
| `contains(x, y)` | 2 | strict | A | geo | closed curve or region containment |
| `cross_section_of(g, s)` | 2 | strict | B | model | |
| `hidden_edge(s)` | 1 | strict | B | model | dashed-line convention |
| `asymptote_of(l, c)` | 2 | strict | B | model | |

Derived, never primitive, never separately scored (produced by the rule set R): `radius_of`, `diameter_of`, `chord_of`, `secant_to`, `inside` when it follows from `center_of`, pairwise `parallel` beyond a spanning set, `collinear` sub-tuples.

**Labels (category L, strict).** `label(t, e)` with `t` the label string as printed (`geo` for presence of the string in the compiled PDF; `model` for semantic equivalence when strings differ, Section 10.3 Q4); `measure_label(t, e, kind)` with `kind ∈ {length, angle, area, coordinate, name}` (`model`).

**Layout (category Y, coarse, anchor A, geo).** `left_of(a, b)`, `above(a, b)` (antisymmetric, one canonical direction), `inside(p, x)`, `outside(p, x)`, `same_side(p, q, l)`, `opposite_sides(p, q, l)`, `order_along(x, S)`, `orientation(x, o)`.

Total: 47 predicates. Closed plane-geometry languages of comparable size exist as precedent (Inter-GPS uses 73 predicate and attribute symbols; PGDP5K uses 16 shape classes, 5 positional relations, 22 symbol types, 6 text types; [arXiv:2105.04165](https://arxiv.org/abs/2105.04165), [arXiv:2205.09947](https://arxiv.org/abs/2205.09947); verified by the reviewer of record from the papers' abstracts and tables).

### 7.2 Bounding expressiveness

- Analytic properties of curves (monotonicity, concavity, extrema, period) are excluded from v1.0 beyond the `curve(e, f)` family label.
- Numeric and metric facts (coordinates, degree measures, lengths) are excluded; coordinates enter only as alignment keys (Section 7.4) and never as scored facts.
- Solids are included minimally, all anchor B. (The per-family validation criterion V10 that could remove them was deleted on 2026-09-26.)
- Adding a predicate is a minor `asc` version bump. Builders cannot introduce predicates at run time; out-of-vocabulary facts are dropped and the **OOV rate per builder** is published (a rising OOV rate signals that the ontology needs a revision).

### 7.3 Builder inputs: one crop-only card per item

Every builder receives the **canonical crop** and the page OCR words with boxes. Track B context text is supplied to builders **only as a label-disambiguation resource**: it may resolve or veto an ambiguous label reading (`context_assisted: true` on that label fact), and it may trigger contradiction detection, but **it may never introduce a fact that the crop does not support**. Consequently there is one card per item, one recall denominator, and the grader's evidence is constant across Tracks A and B; context belongs to the contestant, not the scorer. A sensitivity re-score excluding `context_assisted` facts is published.

Builders: **N = 3**, in **3 distinct declared model families**, family map published in the card manifest. *(Owner amendment 2026-09-23, roster decision: reduced from 4 builders in at least 3 families for streamlining; disclosed as a plan deviation. Implemented in `roster.builder_rule_problems`, the one rule shared by `roster-check` and `cards-build`.)* Builder families are disjoint from judge families in the primary configuration (Section 10.9).

### 7.4 Builder output schema

```jsonc
{
  "asc_version": "asc-1.0.0",
  "builder_id": "builder_b2",
  "builder_family": "fam_x",
  "entities": [
    { "local_id": "e1", "type": "point", "label": "A",
      "anchor": {"x": 0.31, "y": 0.62, "w": 0.02, "h": 0.02},   // crop-relative; ALIGNMENT ONLY, never scored
      "label_certain": true }
  ],
  "facts": [ {"predicate": "on", "args": ["e1", "e7"]} ],        // closed vocabulary only
  "abstentions": [ {"scope": "label", "entity": "e4", "reason": "glyph illegible"} ],
  "oov_dropped": [ "<verbatim out-of-vocabulary strings>" ]
}
```

Every fact must be a self-contained proposition readable without the crop (no "the intended angle", no "the same point as before"); facts failing this closure requirement are abstained at build time, because the recall question in Section 10.3 is asked without the crop.

### 7.5 Label strings and normalization

`label.text` stores the string as printed. Normalization for matching is **strict** and deterministic: Unicode NFC, whitespace trimmed, case folded for Latin scripts, and a versioned confusable-glyph map (O/0/О, l/1/I, Latin–Greek–Cyrillic homoglyphs, CJK variant forms). **No external terminology dictionary is used anywhere in the V1 pipeline** (neither in building cards nor in judging); this removes a leakage surface and a source of language-dependent leniency identified in review. Semantic label equivalence beyond strict matching is a judge question (Section 10.3, Q4) reported as a secondary metric.

### 7.6 `allowed_variation` is a protocol constant

The invariances are properties of the benchmark, not of a diagram. The frozen list: global translation; uniform scaling; line width; font family; colour; label offset within ε; canvas padding; DPI within the pinned band; source refactoring that yields the same render. Reflection is not in the list. Since asc-2.0.0 no card fact is chirality-bearing (layout is not a card class), so reflection is neither tested nor scored (component 07, disclosed), and the contestant prompts (v2, 2026-09-26) no longer call it disallowed. The list was the semantics-preserving vocabulary of the §12.5 mutation algebra; that algebra was removed on 2026-09-26 (checkpoint simplification), so the list is no longer tested by mutations.

---

## 8. Alignment, normalization, consensus, coverage, and eligibility

### 8.1 Deterministic entity alignment with abstention

Builder cards `K_1 … K_3` are aligned to a single accumulating canonical entity set in a **frozen builder order** recorded in the manifest (transitivity holds by construction; the builder-order permutation check on controls, published as card instability, was removed on 2026-09-26).

```text
CANON = []
for b in frozen_builder_order:
    cost[u][v] = -score(u, v) for u in K_b.entities, v in CANON with compatible types
    match = hungarian(cost)
    for (u, v) in match:
        if score(u, v) < THETA_MATCH:                       continue          # u becomes a new slot
        if margin(u, v) < DELTA_MARGIN:  mark_ambiguous(u, v); continue
        if automorphic_tie(u, v):        mark_ambiguous(u, v); continue
        merge(u -> v)
    for u unmatched: CANON.append(new_slot(u))
```

`score(u, v) = w_L · sim_label + w_G · sim_geom + w_R · sim_rel`, where `sim_label` is 1 for equal normalized labels, 0.5 at edit distance 1 under the confusable map, a hard −∞ if both are labelled and unrelated, and a neutral 0.5 if either is unlabelled; `sim_geom = exp(−d/σ)` with `d` the anchor-centroid distance (IoU for extended entities); `sim_rel` is the Jaccard similarity of typed 1-WL colour-refinement signatures after 2 rounds over the fact graph. The weights, `THETA_MATCH`, `DELTA_MARGIN`, `σ` are form constants (`align_*` in the registry: 1.0, 1.0, 0.5, 0.9, 0.15, 0.05); no fitting procedure exists. Their end-to-end validation by card precision and coverage on the controls (V1c) was removed with the control suite on 2026-09-26.

Abstention rules: below threshold, no merge; margin below `DELTA_MARGIN` on either side marks both entities `ambiguous`; a nontrivial automorphism swapping `u` with another same-type unlabelled entity within `σ` marks it `ambiguous` unless geometry alone clears both thresholds; two differently labelled entities never merge. **Any fact with an `ambiguous` argument is ineligible regardless of votes.**

### 8.2 Canonical form, entailment, contradictions, and the minimal generating set

Canonical form: symmetric predicates store arguments in id order; antisymmetric layout predicates store one direction; set arguments are sorted maximal sets.

Entailment rule set **R** (finite, range-restricted so that forward chaining terminates):

```text
endpoint_of(P,X)                          -> on(P,X)
vertex_of(P,G)                            -> on(P,G)
edge_of(S,G) & endpoint_of(P,S)           -> vertex_of(P,G)
center_of(P,C) & circle(C)                -> inside(P,C)
tangent_to(L,C,P) & P != unknown          -> on(P,L) & on(P,C)
intersect_at(X,Y,P)                       -> on(P,X) & on(P,Y)
between(P,Q,R)                            -> collinear({P,Q,R})
collinear(S) & S' ⊂ S & |S'| >= 3         -> collinear(S')        # keep only the maximal set
parallel(X,Y) & parallel(Y,Z)             -> parallel(X,Z)        # keep a spanning tree
perpendicular(X,Y) & parallel(Y,Z)        -> perpendicular(X,Z)
center_of(O,C) & endpoint_of(O,S) & endpoint_of(P,S) & on(P,C) -> radius_of(S,C)   # derived only
```

Exclusion table **E** (mutually exclusive patterns): `inside(P,X) xor on(P,X) xor outside(P,X)`; `parallel(X,Y) xor intersect_at(X,Y,_)`; `tangent_to(L,C,_) xor two distinct intersect_at(L,C,·)`; `left_of(A,B) xor left_of(B,A)`; `above(A,B) xor above(B,A)`; `same_side(P,Q,L) xor opposite_sides(P,Q,L)`; two different normalized labels on one entity.

```text
C = forward_chain(F, R)
V = pairs in C matching E                     # contradictions; every participating fact is ineligible, both sides recorded
key(f) = (predicate_rank[f.pred], -arity(f), sorted(f.args))
M = C
for f in sort(C, key, descending):
    if f in forward_chain(M \ {f}, R): M = M \ {f}
return M                                      # the canonical minimal generating set (MGS)
```

Entity-existence facts are retained deliberately (they are the existence layer). Only MGS facts are voted on, stored, judged, and counted. `contradiction_rate(item) = |facts in V| / |M|`. Removing entailed facts also lowers per-run judge cost.

### 8.3 Verification pass

Silence is not a vote. After alignment and MGS reduction, the union `U = MGS(∪_b MGS(K_b))` is rendered as natural-language statements (in a seeded order per item and builder) and each builder `b` returns `v_{b,f} ∈ {true, false, abstain}` for every `f ∈ U` given the crop and OCR. The pass turns silence into explicit votes, so a fact generated by one builder can be confirmed by the others and vetoed by any of them. There are no planted controls: they only estimated builder operating characteristics for the log-likelihood-ratio rule, which no longer exists.

### 8.4 Consensus: one fixed rule

**Fact `f` is eligible iff all hold:** (1) `f ∈ M`; (2) builders from ≥ 2 distinct families voted true
(`min_true_families`); (3) no family voted false (a family votes false if any of its builders does); (4) `f`
participates in no E-violation; (5) every argument is a non-ambiguous canonical entity; (6) if `anchor = B`, every
argument is uniquely bindable.

With three builders in three families, (2)–(3) mean "three true, or two true and one abstain". This is the rule the
plan preregistered as its declared fallback; it is now the only rule, for every category. It has one parameter and
needs no calibration data. Its precision is neither tuned nor measured: the synthetic check of card precision
against oracle graphs (`P_card(s)`, V1c) and its scope responses were removed with the control suite on 2026-09-26,
and on historical items card precision was never measured. It is a disclosed limitation (8.6).

Nothing here uses a builder's self-reported confidence. Unsupervised latent-class estimation (Dawid–Skene) is not
used: builder families are correlated, which is why the rule counts families, not builders.

### 8.5 Frozen fact record

```yaml
fact_id: r_004                 # stable at freeze, on every fact including labels and layout
class: relation                # entity | relation | label | layout
predicate: on
args: [p_A, c_1]
strictness: strict
decidability: geo              # geo | model
anchor: A
context_assisted: false
supporting_families: [fam_x, fam_z]
votes: {builder_b1: true, builder_b2: abstain, builder_b3: true}
eligible: true
```

Per-fact votes and families are released (asc-2.0.0: `generated` when the fact is in the closure of the writer's
own generation, else its verification vote; leave-one-builder-family-out was removed with the two-writer rule).

### 8.6 Coverage is measured separately from correctness

Correctness of the eligible set is `P(fact true | eligible)`; coverage is `|eligible MGS| / |true MGS|`. Neither is measured. On historical items they never were; the synthetic measurements on the control suite (`P_card(s)`, `R_card(s)`, V1c) and the oracle-candidate ceiling were removed on 2026-09-26 (checkpoint simplification). Both are disclosed limitations (Section 18.2). `SC` and `UCR` were removed on 2026-09-26 as recorded numbers with no gate, decision rule or consumer.

The paper reports recall as *recall against the frozen card*, never as recall against the diagram.

### 8.7 Item eligibility gates (numeric, frozen from development books)

An item enters the scored set iff all hold:

- rights and provenance complete (R3), not a lost duplicate representative (Section 6.7);
- `|eligible entity facts| ≥ 3` and `|eligible strict relation facts| ≥ 3`;
- `contradiction_rate ≤ 0.10`;
- `|ambiguous entities| / |canonical entities| ≤ 0.20`.

The former degradation ceiling (every degradation statistic at or below `s*`, fitted on the card arm of the control suite) was removed on 2026-09-26 with the controls; no item is excluded for degradation. Starting values above are preregistered in the form-freeze and confirmed (or replaced with a measured value) in the level freeze that closes MS3 (Section 17.3). Failing items go to the unscored pool with a closed-enum reason (Section 6.6) and inherit their book's split. Exclusions are published cross-tabulated by degradation statistic, language, and book. No single book may contribute more than 25 percent of the eligible set; excess items from a book are moved to the unscored pool by a deterministic rule (lowest `eligible_mgs_count` first) recorded in the manifest.

### 8.8 Versioning and cost

`asc-<major>.<minor>.<patch>+<hash>` where the hash covers the predicate table, R, E, the normalization and confusable maps, alignment weights and thresholds, the frozen builder order and family map, builder model ids, prompts and decoding settings, the consensus rule parameter, and item-gate levels. Any change is a new version; cards are rebuilt or explicitly labelled.

Cost for a 500-item pilot: about 1,500 generation and 1,500 verification calls, paid once; alignment, closure, MGS, and coverage proxies are milliseconds of CPU. Human effort per item, per model, per book: zero.

### 8.9 Rejected alternatives

A geometric theorem prover for entailment (unbounded, semantically mismatched to printed conventions); unsupervised Dawid–Skene as the production rule; an LLM arbiter for ambiguous alignments (another correlated model inside the consensus it referees); dual cards built with and without context (doubles cost, creates two denominators, changes no pilot decision); soft truth values carried into scoring; metric facts in the card; seven or more builders (correlation, not count, is the binding constraint); a learned alignment model; per-item `allowed_variation` authoring; probability calibration maps and calibration-error gates (the precision floor is verified directly instead); the family-clustered log-likelihood-ratio consensus with per-category thresholds fitted on CAL (removed 2026-09-25: four calibrated inputs and a fitting pipeline whose protection, card precision, is measured directly by V1c; it also imported a synthetic-to-real transfer assumption into every eligibility decision). The direct verification by V1c cited above was itself removed on 2026-09-26 (checkpoint simplification); card precision is now a disclosed limitation (8.6).
