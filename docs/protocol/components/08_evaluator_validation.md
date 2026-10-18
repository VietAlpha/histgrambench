# Component 08 — Evaluator assurance (Section 12)

Part of *HistMathBench Zero-Annotation Project Plan v2*. Canonical entry point: [`../directives/HistMathBench_Zero_Annotation_Project_Plan_v2.md`](../directives/HistMathBench_Zero_Annotation_Project_Plan_v2.md). This file owns **Section 12**. Global notation is in the top-level plan, Section 3.4.

> **Checkpoint simplification 2026-09-26** (`reports/checkpoint_simplification_20260926.md`). This component used to
> define Track C: synthetic known-answer controls with a calibration/validation partition, degradation calibrated to
> development crops, a derived-diff mutation algebra, the card arm that fitted `s*`, and the preregistered criteria
> V1–V11 decided once per evaluator version in at most three attempts (checkpoints CP3 and CP5, gates G3–G5). All of
> it is deleted. The earlier text is in git history (commit `b07b9c9` and before) and is not current.

| Component card | |
|---|---|
| Purpose | State what evidence supports the scoring path now that no known-answer control suite exists, and what the paper may and may not claim about evaluator reliability. |
| Inputs | The deterministic path (compile, CGX extraction, alignment, `geo` verdicts, aggregation); the judge votes stored by every scored cell. |
| Mechanism | Offline software tests of the deterministic path, including the extractor round trip on generated known geometry; run-level reliability reports for the model judges; the §10.9 drift probe. |
| Outputs | No gate and no level. Reported with the results: judge agreement, indeterminate and evaluator-failure rates per class, leave-one-judge-family-out stability (G6), drift-probe agreement, the `geo` versus model-decided share of scorable facts. |
| Gates | None of its own. G6 (component 10) reads the leave-one-judge-family-out stability. |
| Lifecycle | Software tests run with every code change; reliability reports come from each final run. |
| Human effort | None. |
| Dependencies | Sections 7.1 (decidability), 9, 10, 11, 17. |

---

## 12. Evaluator assurance

### 12.1 Why the control suite was deleted

A generated diagram has a valid oracle for its own render: its facts are computed from the specification that
produced it. That establishes less than the pilot needed.

- **Answer keys.** The card arm ran the writers on synthetic renders with synthetic degradations and turned the
  precision curve into `s*`, an admission ceiling applied to historical crops. Historical scans are not synthetic
  degradations of clean renders, so that transfer had no support, and the plan already forbade claiming card
  accuracy on real items. The ceiling is removed; items are admitted by the card gates of Section 8.7 alone.
- **Judging.** Judges only ever see the contestant's clean render, never the scan, so controls were in-domain for
  adjudication. But 36 of the 46 predicates are `geo`-decided by the deterministic extractor (Section 10.1). For
  them the mutation and invariance criteria re-measured deterministic code, which offline tests cover without
  paid calls. The part only controls could measure was judge accuracy on the 10 model-decided predicates.
  That residual is now a disclosed limitation (12.4). It is not a separate stage.
- **Real-item relations (M1–M5), repeatability (V6) and agreement (V7)** had no production runner. They are removed
  with the criteria table.

### 12.2 What covers the deterministic path

The compile harness, CGX extraction, alignment, `geo` verdicts, EXEMPT routing and aggregation are deterministic
functions of the candidate PDF and the frozen card. Their correctness is a software property, established by the
offline suite (`pytest tests`), including:

- the **extractor round trip** (`hmb cgx-roundtrip`, formerly "V1x DEV"): generated known-geometry TikZ goes through
  the production sandbox and CGX and is compared with the geometry that produced it, per predicate, with no model
  call (recorded evidence: `reports/v1x_dev_10seeds.txt`, P = R = 1.000 on 80 renders, 2026-09-22);
- the vertical path and model-run integration tests (valid candidate scores 1, a relabelled candidate loses exactly
  its derived-diff facts, a compile failure scores 0, reruns make no new calls);
- the §7.6 invariance properties of the shared predicate semantics (`tests/w5`).

These are engineering checks. They set no level, are not gates, and are never reported as benchmark evidence.

### 12.3 What is reported about the model judges

Every model-decided fact is judged by two families and, when they do not decide it, by a third (Section 10.5).
From the stored votes, with no extra calls, the final report publishes per predicate class: two-judge agreement,
the indeterminate share, the evaluator-failure rate, and leave-one-judge-family-out ranking stability (G6). The
judge drift probe (Section 10.9) brackets the scoring window. Its frozen triple set is drawn from judged
development-set cells, with the verdict recorded there as the reference. It is frozen at the level freeze.

### 12.4 What the paper may conclude

Permitted: `geo` verdicts are a deterministic, versioned function of the compiled candidate, tested offline against
known geometry; model-decided facts carry the stated agreement, indeterminate and failure rates; rankings are
stable (or not) when any one judge family is left out; the judges did (or did not) drift within the scoring window.

Not permitted: that the evaluator is validated against known answers on historical items; that answer keys are
accurate for real items; any known-answer accuracy figure for the judges. A bias shared by all three judge
families on a model-decided predicate would not be detected; the share of scorable facts it could affect is
reported with the results.

### 12.5 Rejected alternatives

Keeping a reduced control or mutation set inside the pilot or a validation milestone. It would still measure
answer keys on synthetic input only, and for judges it would test a narrow slice (model-decided predicates on
generated drawings) under generator conventions rather than historical ones, at the cost of a partition,
preregistration and attempt accounting. Human spot checks are excluded by the zero-annotation contract.
