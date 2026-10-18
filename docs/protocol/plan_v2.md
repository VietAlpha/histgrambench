# HistMathBench: Zero-Annotation Project Plan v2

**Working title:** HistMathBench
**Version:** Project plan v2.0 (replaces v1.0 of 20 September 2026)
**Date:** 21 September 2026
**Status:** Proposed pilot design, ready for Phase 0

This document is the canonical entry point. It contains the executive summary and claim boundary, the objective and tracks, the zero-annotation contract, the data partitions, the global notation, the cross-component architecture, the risks, the reporting requirements, and the next actions. Sections 4–17 are held in the component files under [`HistMathBench_v2_components/`](HistMathBench_v2_components/) and are linked from the table of contents below. Section numbers are stable across all files. The top-level plan plus the nine component files form one complete plan; no other document is required to implement or audit it.

## Table of contents

| § | Title | Location |
|---|---|---|
| 1 | Executive summary and claim boundary | this file |
| 2 | Objective, tasks, tracks, inputs, outputs, non-goals | this file |
| 3 | Zero-annotation contract, data partitions, notation, architecture | this file |
| 4 | Source, rights, provenance, licensing | [02_data_rights_provenance.md](HistMathBench_v2_components/02_data_rights_provenance.md) |
| 5 | Dataset item schema and immutable artifacts | [02_data_rights_provenance.md](HistMathBench_v2_components/02_data_rights_provenance.md) |
| 6 | Automated acquisition, detection, filtering, deduplication, context, metadata | [03_acquisition_pipeline.md](HistMathBench_v2_components/03_acquisition_pipeline.md) |
| 7 | Atomic Semantic Card ontology and machine generation | [04_atomic_semantic_cards.md](HistMathBench_v2_components/04_atomic_semantic_cards.md) |
| 8 | Entity alignment, logical normalization, consensus, coverage, abstention, eligibility | [04_atomic_semantic_cards.md](HistMathBench_v2_components/04_atomic_semantic_cards.md) |
| 9 | Contestant protocol, TikZ extraction, security, compilation, rendering | [05_submission_compilation.md](HistMathBench_v2_components/05_submission_compilation.md) |
| 10 | Semantic adjudication: deterministic checks, model judges, escalation, extras | [06_semantic_adjudication.md](HistMathBench_v2_components/06_semantic_adjudication.md) |
| 11 | Grading and metrics | [07_grading_and_statistics.md](HistMathBench_v2_components/07_grading_and_statistics.md) |
| 12 | Evaluator assurance (Track C deleted 2026-09-26) | [08_evaluator_validation.md](HistMathBench_v2_components/08_evaluator_validation.md) |
| 13 | Sampling, split groups, selection reporting, contamination, pilot size | [09_sampling_integrity_reproducibility.md](HistMathBench_v2_components/09_sampling_integrity_reproducibility.md) |
| 14 | Statistical analysis plan | [07_grading_and_statistics.md](HistMathBench_v2_components/07_grading_and_statistics.md) |
| 15 | Reproducibility, manifests, versioning, release policy | [09_sampling_integrity_reproducibility.md](HistMathBench_v2_components/09_sampling_integrity_reproducibility.md) |
| 16 | Cost model and staged inference policy | [10_execution_and_roi.md](HistMathBench_v2_components/10_execution_and_roi.md) |
| 17 | Phase plan, automated gates, scale-or-stop rule | [10_execution_and_roi.md](HistMathBench_v2_components/10_execution_and_roi.md) |
| 18 | Risks, residual limitations, claims the paper must not make | this file |
| 19 | Required paper and benchmark-card reporting | this file |
| 20 | Immediate next actions in dependency order | this file |

---

## 1. Executive summary and claim boundary

HistMathBench tests whether multimodal models can reconstruct degraded historical mathematical diagrams as executable TikZ. The input is a crop from a historical page scan; the output is TikZ that compiles into a clean diagram preserving the mathematics the original author intended: points, lines, circles, incidences, intersections, tangencies, containment, labels, and coarse layout, but not stains, fading, warp, or broken ink.

The design constraint is zero per-item human work. There is no gold TikZ, no manually restored reference image, no human semantic annotation, no human grading, and no manual rescue of uncertain items. Instead, a frozen pair of answer-key writers in two distinct model families (§7.3, amended 2026-09-26) converts each crop into an **Atomic Semantic Card**: a closed-vocabulary, logically minimal set of testable entity, relation and label facts. A fact enters scoring only when both writers support it (one wrote it and the other confirmed it, or both wrote it) and neither rejects it; its precision on historical crops is not certified (the card gates and the development-set measurements report what is observable). Candidate TikZ is compiled in a pinned sandbox, its geometry and text are extracted deterministically from the compiled PDF, and the facts that geometry cannot decide are put to two model judges from distinct families (a third when they disagree); their reliability is reported from the run itself (agreement, indeterminate and evaluator-failure rates, leave-one-judge-family-out stability). Deterministic code turns verdicts into scores.

> **Simplification 2026-09-25** (`reports/simplification_audit_and_execution_20260925.md`). One item score (recall
> over the eligible strict facts), one headline (the group-balanced mean) with one interval method (the
> Ibragimov–Müller group t), three mandatory diagnostic families, a fixed card consensus rule, one majority rule for
> every judge question, and gate evidence computed by `score-report`. Removed: fitted builder and judge
> reliabilities (α/β, π_k, τ_k, η/θ/ρ) and the planted controls that estimated them, the class-weighted score and
> `cap_Y`, the optimistic bound, the not-scored state and list-wise drop, the composite-weight family, Q4, the
> extractor audit, the SC gate. The estimand below is the group-balanced mean.

> **Checkpoint simplification 2026-09-26** (`reports/checkpoint_simplification_20260926.md`). The Track C control
> suite (CP3) and the evaluator-validation checkpoint (CP5, criteria V1–V11, gates G3–G5) are deleted, with the
> card arm, `s*`, the severity calibration, the mutation algebra and 18 registry parameters. A synthetic control
> has a valid oracle for its own render, but it cannot certify answer keys on historical scans, and on the judge
> side it re-tested the deterministic extractor that decides 36 of the 46 predicates. The program is now five
> milestones (Section 17 in component 10): MS1 form freeze, MS2 pilot, MS3 development set with sizing and the
> level freeze, MS4 test set and release, MS5 final run and report; book census and rights clearance is a
> parallel readiness track. Where the text below still describes Track C, CAL/VAL, V1–V11 or `s*`, this note
> supersedes it.

**What v2 changes.** The v1 design was methodologically incomplete in ways that would have made its pilot decision unfalsifiable: its precision denominator did not exist, its card coverage was never measured, its three-way book split left too few clusters for its own inference method, its thresholds were to be preregistered before any data existed, and its evaluator validation never ran the card builders. v2 repairs each with an automatic mechanism and removes complexity elsewhere: one crop-only card per item; a deterministic geometry extractor that defines the candidate universe and removes most judge calls; a fact-level headline with no free weights; a two-way split with a group-level t-test; a two-act threshold freeze; and a control suite that validates card construction, extraction, and judging end to end with held-out generator families.

**Claim boundary (formal estimand).** Let `H` be the population of crops extracted from approved books that survive the deterministic gates and the target-diagram verifier (before the card gate); `G(i) ∈ {0,1}` the item-eligibility indicator; `ρ = P(G = 1 | i ∈ H)` the eligibility rate; `V_e` the frozen evaluator version; `S(m, i; V_e)` the per-item score of model `m`; `g` a split group. The primary estimand is the group-balanced mean

```text
Θ(m; V_e) = E_g [ E[ S(m, i; V_e) | G(i) = 1, i ∈ g ] ]
```

estimated by the mean over test split groups of the mean eligible-item score, with the Ibragimov–Müller interval over those group means (component 07, Sections 11.8 and 14.2). Groups count equally because the group is the sampling unit; with equal group sizes this is the plain item mean. `Θ` is not an estimate over `H` (selection is reported, not removed) and is not evaluator-free (every number is conditional on `V_e`). The mandatory claim template:

> Under evaluator version `V_e`, model `m` attains `Θ̂(m)` on the subset of 1850–1930 printed mathematical diagrams for which an automated consensus specification could be constructed (eligible fraction `ρ̂`). No claim is made about diagrams outside that subset, about recovery of any canonical or author-intended figure, or about performance under a different evaluator. Reference specifications and judgments are produced by a frozen, versioned automated pipeline whose `geo` verdicts are deterministic and software-tested and whose model-judge reliability is reported from the run itself; neither the answer keys nor the judges are certified against known truth on historical items.

**Pilot.** About 400–500 eligible items from about 10–15 public-domain books, evaluated with about 5–6 models chosen by a preregistered inclusion rule, with a measured sizing rule that can shrink the pilot but never enlarges it by default. Scaling to 2,000–3,000 items from 40–80 books happens only if the preregistered gates of Section 17 pass.

---

## 2. Objective, tasks, tracks, inputs, outputs, non-goals

### 2.1 Primary task

Given an automatically detected crop of a historical mathematical diagram, generate executable TikZ that reconstructs the intended mathematical figure.

```text
book (rights-cleared) → page scan → deterministic gates → diagram crop (+ context)
    → contestant model → TikZ → sandboxed compile → canonical render + extracted geometry
    → deterministic verdicts (geo facts) + calibrated model judges (model facts) → scores
```

### 2.2 Primary research question (two parts)

1. Can a fully automated semantic evaluator, built without per-item human annotation, reliably distinguish correct reconstructions from controlled mathematical errors while ignoring cosmetic variation? (Answered only in part since 2026-09-26: `geo` verdicts come from deterministic code tested offline against known geometry; model-judge reliability is reported from the final run, not measured against known answers; card accuracy on historical crops is not certified. See component 08.)
2. Conditional on (1), how do current multimodal models perform at executable semantic reconstruction of degraded historical diagrams under that evaluator?

### 2.3 Secondary questions and their status at pilot scale

| Q | Question | Status | Reason |
|---|---|---|---|
| Q1 | Which models best preserve entities, relations, labels, layout? | secondary, comparative under `V_e` | names the evaluator in every sentence |
| Q2 | Which degradation types are associated with the largest losses? | exploratory only | eligibility is correlated with degradation; no causal version exists since Track C was deleted (2026-09-26) |
| Q3 | How much does nearby text help? | secondary, causal | within-item manipulation with a fixed reference (Section 11.7) |
| Q4 | Variation across books, decades, languages, publishers? | deferred to V1 | collinear with book at 10–15 books |
| Q5 | How often do outputs look plausible while changing the mathematics? | secondary, restated | the contradicted-relation rate among candidates with entity recall ≥ 0.8 (Section 11.4) |
| Q6 | Evaluator reliability against controlled errors? | co-primary | the only question with a known answer key |

### 2.4 Non-goals

No new reconstruction model, encoder, or architecture; no reproduction of paper texture or print defects; no single TikZ coding style; no canonical coordinates; no exact source-code matching; no claim that machine-generated cards are human gold; no new target domains, output formats, historical periods, tracks, or languages in V1.

### 2.5 Tracks

- **Track A (headline): crop → TikZ.** Input is the canonical crop with captions, figure numbers, and body prose automatically removed (Section 6.4); text printed inside the diagram remains. It isolates reconstruction from prose outside the crop.
- **Track B: crop + source context → TikZ.** The same crop plus the automatically extracted context window (stripped caption prepended), in the source language. Paired with Track A on the same items and the same card, so the context effect is a pure model effect (Section 11.7).
- **Track C: deleted 2026-09-26.** The known-answer control suite and its secondary analyses are removed (see the checkpoint-simplification note above). The harness label `C` survives only for generated known-geometry renders in the offline software tests; it is never a benchmark track and never shares an axis with Tracks A and B.

### 2.6 Diagram taxonomy (reporting variable only)

Plane geometry (triangles, polygons, circles, tangents, secants, constructions, coordinate geometry); solid geometry (prisms, pyramids, cylinders, cones, spheres, cross-sections); functions and curves (Cartesian plots, conics, elementary function plots); trigonometry and measurement; plus `other` and `unknown`. Category is never an eligibility criterion; eligibility comes solely from the deterministic gates and the TikZ-representability verifier. Statistical charts, maps, decorative illustrations, astronomy charts, engineering drawings, and isolated formulas are deferred. The package allowlist must cover the idiomatic TikZ libraries of every listed family, and `Valid@1` is reported by family as the check.

### 2.7 Model set (preregistered inclusion rule)

> **Locked contestant roster (owner decision 2026-09-25).** Claude Sonnet 4.5
> (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`, Bedrock Runtime Converse), Llama 4 Maverick 17B-128E Instruct
> (`us.meta.llama4-maverick-17b-instruct-v1:0`, Converse), Qwen3 VL 235B-A22B Instruct
> (`qwen.qwen3-vl-235b-a22b-instruct`, Bedrock Mantle Chat Completions at `https://bedrock-mantle.us-east-1.api.aws/v1`),
> all three with image input and on the direct-figure track; DeepSeek V3.1 (`deepseek.v3.1`, same Mantle endpoint)
> is text only, never on the direct-figure track, and keeps its locked entry with no scored text-input track (the
> §13.6 T3 probe is a diagnostic). Standard service tier, us-east-1, no fallback. The direct Anthropic, OpenAI and
> Gemini connectors are removed (D14). No model pair is named in advance: the 2026-09-25 primary contrast
> (Claude Sonnet 4.5 − Qwen3 VL) is superseded by owner decision D21 (2026-09-26); G7 reads the all-pairs comparison.

Frozen before the run, on a recorded date `D`: (a) all frontier general-purpose multimodal models with public API access on `D`; (b) open-weight multimodal models meeting a stated reproducibility bar; (c) at least one non-multimodal-LLM pipeline baseline; (d) at least one deliberate floor baseline. Target `M = 5–6`, of which the frontier subset (a)+(b) has at least four members (needed for rank-stability statistics). The separation gate is evaluated on the frontier subset only; (c) and (d) are calibration anchors. Per model, record whether raw outputs are releasable; non-releasable models are marked not independently reproducible. Every result records model id, provider version, family, access date, decoding settings, prompt hash, image preprocessing, attempts, and any provider feature that could affect output. One output per item per track.

> **Owner amendment, 2026-09-23 (roster decision `reports/decision_model_roster_20260923.md`; implemented 2026-09-23).**
> - **Model set.** 14 open-weight or low-cost image models plus one floor, all on Amazon Bedrock, instead of `M = 5–6`
>   and instead of all frontier models (a), by owner choice (cost). The frontier subset used for rank statistics is
>   these 14 main contestants. This deviation is disclosed.
> - **Text-only floor (c)/(d).** The floor model (DeepSeek V3.1) accepts text only. A roster entry with
>   `input_mode: text_only` is never sent an image: on both tracks it receives the frozen no-image prompt
>   `reconstruction-T3-v2` followed by the item's context text (stripped caption and nearby prose), so its Track A and
>   Track B inputs coincide. Its records carry that `prompt_id` and `image_encoding.format = none`. It partly fills the
>   "non-multimodal pipeline" baseline (c) and serves as the floor (d).
> - **Effort.** Each roster model carries a pinned reasoning `effort` (`medium` or `high`; gatekeeper and graders high,
>   answer-key writers medium). It is one value per role and model family, is part of every model-call cache key, and is
>   recorded with every result (`decoding_params.effort`, the card manifest, the judge configuration and the run
>   manifest). `roster-check` refuses a builder, judge or gatekeeper without it and never fills a default.
> - **Transport.** Models served only on Bedrock's OpenAI-compatible `bedrock-mantle` endpoint (Gemma 4) use the
>   `bedrock_mantle` provider (Chat Completions) with the model card's endpoint URL pinned in the roster; they are never
>   routed through the standard `bedrock-runtime` Converse API.

### 2.8 Adjacent work and the novelty statement

Prior work covers modern figure-to-TikZ synthesis with human-authored program references (DeTikZify with the DaTikZv2 dataset of over 360,000 human-created TikZ graphics, [arXiv:2405.15306](https://arxiv.org/abs/2405.15306); DaVinci with TikZ30K, ICLR 2026, verified from the venue record only), text-conditioned mathematical diagram generation (MathemaTikZ, ACM L@S 2025, verified from the venue record only; Math-Vision Diagrams, 2,920 items built with an LLM ensemble plus subject-matter-expert curation, [arXiv:2608.08964](https://arxiv.org/abs/2608.08964)), and primitive-level analysis of historical astronomical diagrams with human annotation (303 manuscript diagrams hand-annotated with over 3,000 primitives, [arXiv:2403.08721](https://arxiv.org/abs/2403.08721); 948 diagrams across seven linguistic traditions with 10,940 annotated text regions, [arXiv:2606.15886](https://arxiv.org/abs/2606.15886)).

HistMathBench differs in three respects that must be claimed together: **printed** historical mathematics rather than manuscript astronomy; **executable program output evaluated on mathematical semantics** rather than primitive extraction or image similarity; and a **reference specification constructed with no per-item human annotation**, where every adjacent resource relies on human-authored, hand-annotated, or expert-curated references. Book-held-out evaluation is a correctness requirement, not a contribution. The list above must be re-swept against 2026 venues before the novelty paragraph is frozen.

---

## 3. Zero-annotation contract, data partitions, notation, architecture

### 3.1 What humans may and may not do

Permitted, and the only human work in the plan: set protocol-level policy (ontology, rules, thresholds' forms, prompts); approve source books and record repository terms (one glance at a title page, one manifest per book); configure credentials and the rate card (spend is tracked by the ledger, not capped); make the sizing and publish-or-stop decisions. The collapse guard is a standing deterministic rule (13.1), not a human decision. Forbidden: creating or editing any card, fact, label, verdict, or score; repairing, rescuing, or re-cropping any individual item; adjusting any threshold after the level freeze. Gate G1 (`human_touch_rate = 0.000`) audits this from the pipeline log.

### 3.2 Lifecycle of every mechanism

| Mechanism | Frequency | Automated per item |
|---|---|---|
| ontology, rules, prompts, sandbox, extractor, generator, thresholds' forms | one time per evaluator version | — |
| rights manifest, evidence bundle, source-format choice | per book | — |
| page render, OCR, proposers, gate ladder, funnel, statistics, dedup | per page and per proposal | yes |
| card generation, verification, alignment, closure, consensus, eligibility | per item, once, cached forever | yes |
| prompt, extraction, compile, canonicalization, extraction record | per item per model per track | yes |
| judge calls (two per cell with model facts; a third on undecided facts) | per cell | yes |
| scoring, aggregation, inference | per run | yes |
| drift probes | per scoring window | yes |

### 3.3 Data partitions

| Partition | Content | Used for | Never used for |
|---|---|---|---|
| protocol-development data | the MS2 pilot (≤ 30 crops from 2 development books) | software debugging, metering, decoding caps | any claim |
| development data | the MS3 development set (development books only, disjoint from test books at the book level) | retention, facts per item, contradiction and agreement, the variance probe and sizing, measured replacements for provisional levels, the drift-probe triple set | any reported score or claim |
| public development data | the development split groups' items, cards, votes, verdicts | public examples and debugging by others | any reported score |
| public test data | none as a separate split in the pilot; the test split becomes public at paper time | — | — |
| hidden test data | none in the pilot; V1 hidden books reserved at census and never extracted before V1 | V1 | — |

Rule: no item is used both to set a threshold and to make the final claim about that threshold; test books never contribute to form design, levels or sizing.

### 3.4 Global notation

| Symbol | Meaning |
|---|---|
| `b`, `g`, `i` | book, split group (Section 13.1), item; `N` eligible items; `G` test split groups |
| `F_i` | item `i`'s frozen eligible minimal generating fact set; classes `c ∈ {E, R, L}` (entity, relation, label; layout left the card vocabulary with asc-2.0.0, 2026-09-26); each fact has `decidability ∈ {geo, model}` |
| `m`, `t` | contestant model, track ∈ {A, B}; a cell is `(i, m, t)`; `M` models |
| `o(i,m,t)` | outcome code (Section 9.5); `g_i = 1[o = OK]` |
| `v(f)` | verdict ∈ {PRESENT, ABSENT, CONTRADICTED, INDETERMINATE, EXEMPT} |
| `S_i(m,t)` | item score (Section 11.3): PRESENT / scorable over the strict facts; `H(m,t)` the headline, the mean over split groups of the mean item score |
| `V_e` | evaluator version = the release hash chain of Section 15.1 |
| `s ∈ {1,2,3}` | degradation severity on the shared real/synthetic quantile scale (Sections 6.5, 12.4) |

### 3.5 Cross-component architecture

```text
Section 4  rights gate R3 ──► Section 6 render, proposers, CPU gate ladder G0–G7, funnel, statistics, pHash
                                   │ paid: strict gatekeeper (G8b only, 2026-09-23)
                                   ▼
Section 7–8  3 builders (3 families) → align → closure/MGS → verification pass
             → fixed rule (≥ 2 families true, none false) → frozen card (crop-only) ── coverage instruments
                                   │
Section 13   split groups; dev (3) / test (7–12); funnel report
                                   │
Section 9    prompt → extraction (byte-identity invariant) → LuaLaTeX sandbox → outcome code → canonical PNG + PDF
                                   │ OK only
Section 10   PDF → CGX geometry/text extraction → deterministic verdicts (geo) + alignment
             → Q1 two judges (model facts) → third judge only if they do not agree → verdicts
                                   │
Section 11   recall over the eligible strict facts (indeterminate = not recovered); group-balanced headline
Section 14   Ibragimov–Müller group t on split-group means; BH partial order (secondary)
Section 12   evaluator assurance: offline tests of the deterministic path; run-level judge reliability
Section 15   hash chain, release manifest, version triggers
Section 16–17 ledger, two-act freeze, milestones MS1–MS5, gates G1, G2, G6–G9, decision table
```

### 3.6 Endpoint hierarchy (frozen at Phase 0a)

| Tier | Quantity | Inference at pilot |
|---|---|---|
| primary (one number) | Track A headline `H` under `V_e` with its Ibragimov–Müller interval | yes |
| mandatory diagnostics (two families) | validity (outcome codes, `Valid@1`); per-class recall with contradicted and indeterminate shares, Q5 and the evaluator-failure rate (the extras family was removed 2026-09-26) | reported beside every headline |
| secondary | context effect (Section 11.7), the BH partial order (on Track A it binds gate G7, D21), leave-one-judge-family-out stability (G6) | yes, with multiplicity stated |
| optional / exploratory | leave-one-builder-family-out, Q2 by raw degradation statistic, `touching_cc`, prominence tables, language and decade tables, T3 probe, `S_adj` and envelope | no inference, no stars, never blocking |

---

## 18. Risks, residual limitations, and claims the paper must not make

### 18.1 Risk register

| Risk | Consequence | Mitigation (section) |
|---|---|---|
| Cards omit facts or select the obvious ones | recall inflated; separation compressed | structure proxy `SC`, facts per item `F̄`, ceiling-effect flag (8.6, 14.6); card coverage is not measured against known truth (disclosed) |
| Builders share a misconception | false eligible fact every contestant inherits | both writer families must support and none reject, contradiction and ambiguity gates (8.4, 8.7); residual shared error is a disclosed limitation |
| Builder or judge family shares priors with a contestant | family-level advantage | disjointness or rotation, overlap matrix, leave-one-family-out (10.9, 13.5) |
| Judge reads the crop instead of the render | recall inflated | crop withheld from Q1; blank-render probe M5 (10.3, 12.6) |
| Precision denominator judge-dependent | precision not reproducible | deterministic extraction; precision only for geo classes (10.2, 11.5) |
| Eligibility gates remove the most degraded items | headline biased up; degradation effect attenuated | per-stratum funnel, eligibility rate in the claim sentence, composition table, claim boundary (13.3) |
| Editions, translations, shared plates across splits | frozen cards leak | split groups over hard edges; collapse guard (13.1) |
| Too few clusters for inference | pilot decision unfalsifiable | two-way split, `G_min = 6`, group t-test, MDD floor printed (13.2, 14.2–14.3) |
| Thresholds set after seeing data | preregistration hollow | two-act freeze, `thresholds.json` hashed before any test artifact (17.3, 15.5) |
| Levels tuned on test data | optimistic results | levels, sizing and δ* set on development books only and frozen before any test artifact (§17.3 FreezeViolation); book-level split guard |
| Hosted judge drifts | scores silently non-comparable | scoring window, drift probe, snapshot pinning (10.9) |
| Raster paste-in, missing glyphs, hidden repair | headline claims wrong | post-compile artifact checks, `\tracinglostchars=3`, byte-identity invariant (9.4–9.6) |
| Caption text inside the Track A crop | Track A becomes a context condition | caption strip routed to Track B (6.4) |
| Rights challenge after release | history rewritten | tombstones; old release hashes preserved (15.4) |
| Cost exceeds budget | unbounded spend | ledger metering of every call against the dated rate card, spend reporting, G9 cost projection as information; no ceiling or halt (owner decision, 16.5) |

### 18.2 Residual limitations that no mechanism removes

Answer-key accuracy on historical crops is not certified, and model-judge accuracy is not measured against known answers: a bias shared by all three judge families on the 10 model-decided predicates would not be detected (Track C and V1–V11 were deleted 2026-09-26). `geo` verdicts are deterministic and covered by offline software tests. There is no real-item extractor audit and no in-band planted-control estimate (removed 2026-09-25). Selection on unobserved semantic difficulty cannot be measured. Pretraining exposure cannot be excluded, only bounded. A "raster" drawn as many tiny paths cannot be detected. The source-country copyright status of the books is not determined.

### 18.3 Claims the paper must not make

That the evaluator is validated on real historical items in the correctness sense; that cards are accurate for real items; that any number is a property of "historical diagrams" rather than of the eligible subset under `V_e`; any known-answer reliability claim for the judges; any causal degradation claim from Q2; any language, decade, or publisher effect at pilot scale; any total-order leaderboard; any rights claim beyond the sentence in Section 4.7.

---

## 19. Required paper and benchmark-card reporting

Books, pages, proposals, and items at every funnel stage with per-stratum retention; the eligibility rate; the eligible-versus-excluded composition table; the crop-border statistics (`border_ink`, `touching_cc`) and scores by `touching_cc`; the per-item disclosure that items are Paddle's preprocessed crops (Paddle is the only region source, so no detector miss-rate bound exists); source languages, dates, repositories, codecs, and the rights sentence of Section 4.7; the split-group graph and soft-edge crossings; the ontology version and OOV rates; builder families, the consensus rule, `SC` distribution, the development-set retention, facts per item and contradiction rate; the `geo` versus model-decided share of scorable facts; judge families, inter-judge agreement and indeterminate rates, drift-probe results, evaluator-failure rates; the disclosed limitation that neither cards nor judges are certified against known truth on historical items; the exact model set, inclusion rule, prompt hashes, decoding, `output_release`; the headline and the three diagnostic families of Section 11.4 for every model and track; the BH partial order with `Ĝ` and, per pair, `ρ̂`, `σ̂_d` and the MDD floor; the context effect; leave-one-judge-family-out stability with the family-overlap matrix; prominence tables and the text-only recall rate; the ceiling-effect flag; exclusions and failure cases; `release_hash`, `thresholds_frozen_at`; compute and monetary cost from the ledger with the dated rate card; the mandatory claim template of Section 1 and the selection sentence of Section 13.3.

---

## 20. Immediate next actions (dependency order)

1. **MS1 form freeze.** The schemas, ontology, rules R and E, prompts and wrapper, outcome codes, verdict vocabulary, metric code (11), estimator (14) and gate forms (17.5) exist; review them, hash and commit (`form-freeze`, then the owner's approval).
2. **MS2 pilot.** Run ≤ 30 crops from 2 development books through every stage (`hmb pilot`); meter `c_judge`, `c_card`, `c_gen`; set the decoding caps; confirm the engine and font facts of Section 9.10 and the `geo` share of Section 10.1. A form change here returns to MS1.
3. **Readiness track (concurrent).** Identify candidate books; clear development and test books under R3 with evidence bundles and terms; record prominence indicators; reserve V1 hidden books. The one condition checked before MS4 is `books_ready`: the cleared test books form at least `G*` split groups. (The L4 data licence is decided: Apache-2.0, in `release.json`.)
4. **MS3 development set, sizing and level freeze.** Build cards for about 80–100 development crops (`hmb develop`); measure retention, facts per item, contradiction rate and family agreement (`dev-measure`); run two contestants on the same development items and compute `ρ̂`, `σ̂_d` and the sizing simulation (`dev-sizing`); the owner commits `δ*`, `N*`, `G*`; give the 7 calibrated parameters values with provenance; provisional parameters are confirmed at their plan values unless measured; freeze the drift-probe triple set; write `thresholds.json` (`hmb freeze`). There is no `X_total`: spend is tracked, not capped.
5. **MS4 test set and release.** Extract the test books to `N_cand = N*/r̂ × 1.25`; build and freeze cards; assign split groups; check the G2 yield; write the release manifest (`hmb build`).
6. **MS5 final run and report** inside one scoring window with drift probes (`hmb run`), then the score report and the decision table of Section 17.6 (`hmb report`).