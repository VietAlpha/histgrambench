# Component 03 — Automated acquisition, detection, filtering, deduplication, context, and metadata

Part of *HistMathBench Zero-Annotation Project Plan v2*. Canonical entry point: [`../HistMathBench_Zero_Annotation_Project_Plan_v2.md`](../HistMathBench_Zero_Annotation_Project_Plan_v2.md). This file owns **Section 6**. Global notation is in the top-level plan, Section 3.4.

| Component card | |
|---|---|
| Purpose | Convert approved books into candidate diagram crops, context windows, and deterministic metadata, while recording every rejection so selection bias can be measured. |
| Inputs | Books that passed gate R3 (Section 4.5). |
| Mechanism | Page render; three heterogeneous region proposers; a CPU-only gate ladder ahead of any paid call; deterministic image statistics computed for every proposal; caption stripping and panel splitting; pHash clustering; OCR once per page. |
| Outputs | Crops, canonical crops, context text and image, `automatic_metadata`, the append-only selection funnel, `phash_cluster_id`. |
| Gates | G0–G8 (Section 6.3). |
| Retained artifacts | Funnel rows for every proposal; page images; crops; OCR; statistics. |
| Cost tier | Per page (render, OCR, detection) and per crop (one gatekeeper call, G8b) only after the CPU ladder. |
| Lifecycle | Per book, one time. |
| Human effort | None per item. One-time authoring of the figure-number regex list; one per-book choice of the grayscale source when a repository offers several. |
| Dependencies | Section 4 (gate R3), Section 7 (consumes crops and OCR), Section 9.7 (canonical image pipeline), Section 13 (funnel and clusters). |
| Downstream decisions | Candidate pool size; Track A input definition; every selection-bias statistic. |

---

## 6. Acquisition and filtering pipeline

> **Owner amendment, 2026-09-23 (implemented 2026-09-23): Paddle's saved crops are the items.**
> Decisions D3 and D4 in `reports/decision_audit_20260923.md` supersede the region-proposal parts of this section.
> - **§6.2 is superseded.** PaddleOCR-VL (collector output) is the only region source. Its saved `image` and `chart`
>   crops are used as they are. There is no non-learned proposer, no third proposer, no box fusion, no match condition and no
>   detector miss-rate lower bound. A region Paddle missed is not in the benchmark.
> - **§6.1:** masters are still rendered for the page gates G0a–G0d and for the item's page asset, but items are not
>   cut from that render. Each item's box is Paddle's `block_bbox` in Paddle's preprocessed frame, labelled as such.
> - **Disclosure on every item** (`extraction.source_crop`): the image was rotated upright where Paddle's orientation
>   classifier turned the page (1,388 of 30,838 pages), dewarped (nonlinear, not saved, not invertible), resampled,
>   re-compressed as lossy JPEG, and bounded by Paddle's box with no margin, so labels just outside it may be cut
>   off. The record keeps the collector file path and its sha256. **The crop cannot be regenerated from the master.**
>   This conflicts with §6.9, which rejects dewarping before cropping; the conflict is accepted and disclosed.
> - **§6.3:** G1 is removed (no second proposer to agree with). G2–G7d run on the crop in Paddle's frame.
>   **§6.3.1 is a measurement, not a gate (decision D10, implemented 2026-09-23).** A Paddle crop cannot grow, because
>   the flattened page it came from was never saved, so the border test cannot separate a tight complete figure from
>   a cut-off one. G7a records `border_ink[4]`, `touching_cc` and the offending edges (`growth_steps` is always 0) and
>   never excludes; `border_truncation` is no longer emitted. The funnel report publishes the border statistics and
>   the score report stratifies scores by `touching_cc`. Truncation by Paddle's box is covered by the disclosure above.
> - **§6.4:** OCR text is Paddle's text blocks, which share the crop frame; the context window is built from them.
>   No `context_image` is produced, because the flattened page it would be cut from was never saved.
> - **§6.5:** image statistics describe the processed crop, not the source page. The page x-height and physical
>   DPI are measured on the master render and carried into Paddle's frame by the ratio of the long sides.
> - **§6.6:** funnel rows drop `proposer_boxes`, `iou_pairwise`, `max_edge_delta_frac` and `n_proposers_agreeing`
>   and add `box_frame` and `source_crop`. Stage S3 ("proposals matched") is vacuous and equals S2.
> - **Decision D13 (implemented 2026-09-23):** Paddle's `chart` crops are items alongside its `image` crops. Charts
>   that are not target diagrams are rejected downstream by the G8b gatekeeper's target-type test, as for any crop.
>
> **Owner amendment, 2026-09-23 (model-role audit `reports/audit_model_roles_20260923.md`; implemented 2026-09-23):
> the strict gatekeeper is the only AI screening step.**
> - **G8a, G8c and G8d are cut.** There is no category classifier (sorter), no metadata ensemble (tagger) and no
>   embedding duplicate spotter. `category_pred`, `n_agreeing` and the G8d metadata are no longer produced, so results
>   cannot be broken down by the §2.6 diagram type (whether the gatekeeper should name the type is an open owner
>   option), and cross-book reprints are not detected (disclosed limitation).
> - **G8b is the strict gatekeeper, alone.** One call per crop asks whether the crop is a single mathematical diagram of
>   a target type (the §2.6 families: plane geometry, solid geometry, functions and curves, trigonometry and
>   measurement, or another figure built from the same elements; not a statistical chart, map, decorative
>   illustration, astronomy chart, engineering drawing, isolated formula, table or text), whether it is readable, and
>   whether it could be faithfully redrawn in TikZ. The prompt says "when in doubt, reject". The answer is one JSON
>   object `{"target_type", "readable", "tikz_representable"}`; the crop passes only when all three are literally
>   `true`, and any other answer, including an unparseable one, is `verifier_reject`. The "different family from the
>   classifier" rule is gone, so one model's view decides eligibility; this selection effect is disclosed and the
>   rejection rate is published per book.
>
> **Checkpoint simplification 2026-09-26.** The Track C control suite (CP3) was deleted. For this section: the
> ladder thresholds are frozen on the Phase 0 engineering set only (6.3); §6.5 keeps the raw image statistics but
> drops the severity bins, the severity calibration and their validation on synthetic degradations; the item
> exclusion `degradation_above_ceiling` no longer exists (6.6). Development books are measured in milestone MS3
> (formerly CP6).

The organizing principle: **wherever the original design resolved uncertainty by "two models agree", one non-learned check that fails differently is added.** Model agreement measures consistency; only a deterministic test measures the property.

### 6.1 Page acquisition and rendering

For every book passing R3: download the master file, record `source_file_sha256`, `page_count`, `source_codec`, `source_bitdepth`, `source_colorspace`, `scanning_agent` where exposed. When a repository offers both bitonal and grayscale scans of the same book, take grayscale (a per-book choice made at census).

Render policy, pinned: grayscale 8-bit, no auto-contrast, no sharpening, no despeckle, lossless PNG, renderer name and version recorded. Resolution is **measured**, not claimed: `eff_res_xheight_px` = the page's modal text x-height in pixels from the connected-component height histogram.

Page gates (once per page):

| Gate | Test | Pass |
|---|---|---|
| G0a | file opens; page count plausible; checksum recorded; R3 satisfied | all true |
| G0b | page renders; `eff_res_xheight_px ≥ 12`; a figure-only page (Paddle item blocks, no text block) is recorded `figure_only` and passes (amendment 2026-09-25, C21: no exclusion for having no text; its crops meet every later gate) | measured |
| G0c | page not blank: `ink_frac_page > 0.002` | |
| G0d | page OCR once; script detection; skew estimate; records `ocr_status` | never excludes |

### 6.2 Region proposal and box fusion

At least **three proposers that differ in training corpus and model class**, of which **at least one is non-learned**: a classical connected-component and whitespace-projection region proposer. Two prompt variants of one vision-language model are not two proposers. Learned proposers trained on layout corpora will produce multi-panel boxes by design and were largely trained on born-digital pages ([DocLayNet, arXiv:2206.01062](https://arxiv.org/abs/2206.01062)); the non-learned proposer over-proposes on noisy pages rather than under-proposing on faint ones, which is the complementary failure the ensemble needs.

Match condition for two proposals (both required): `IoU ≥ 0.60` **and** per-edge disagreement `|e_i^(1) − e_i^(2)| ≤ 0.10 · d_i` for all four edges, where `d_i` is the width (for left and right edges) or height (for top and bottom edges) of the union box. The per-edge test detects truncation; IoU alone does not on elongated diagrams.

Fusion: crop = union of matched boxes, dilated by `max(8 px, 0.02 × union diagonal)`, then shrunk to the bounding box of ink components overlapping the union, excluding components whose height is within ±25 percent of the page's modal text-line height. `n_proposers_agreeing`, pairwise IoU, `max_edge_delta_frac`, and `box_source` are recorded in the funnel. The former scalar `crop_confidence` is deleted; its inputs are published instead.

**Detector miss-rate lower bound.** For proposers 1 and 2 over a page set with `n1`, `n2` distinct regions and `m12` matched regions:

```text
N_hat        = n1 · n2 / m12                 (pages with m12 = 0 excluded and counted)
missed       = N_hat − (n1 + n2 − m12)
miss_rate_lb = missed / N_hat
```

Positive correlation between proposers biases `N_hat` downward, so this is an honest lower bound on the joint miss rate. Compute for all proposer pairs; divergence among the three values is the diagnostic for dependence. Reported per book and pooled.

### 6.3 Gate ladder

Every gate records `{gate_id, version, passed, measured_value, threshold}` for every proposal that reaches it. `terminal_gate` is the first failure or `passed`. Gates G1–G7 are CPU-only and precede the first paid call.

| Gate | Test | Pass or action |
|---|---|---|
| G1 | fuse boxes (6.2); compute the **full image-statistics vector (6.5) and persist it** | match condition met; vector stored for every proposal including all later failures |
| G2 | geometry | short side ≥ 0.7 cm physical; area ≤ 0.60 of page; aspect ∈ [1/6, 6] |
| G3 | ink coverage | `0.005 ≤ ink_frac ≤ 0.45` |
| G4 | solid or halftone block | reject if one component covers ≥ 0.90 of crop area, or `ink_frac > 0.45` with fewer than 5 components |
| G5 | line art vs photograph | bimodality ratio (mass within ±10 percent of the two Otsu modes over total) ≥ 0.75 |
| G6 | structure present | `max_cc_skeleton_len / crop_diagonal ≥ 0.35` or ≥ 3 components with skeleton length ≥ 0.15 × diagonal |
| G7a | border truncation (6.3.1) | bounded growth, else exclude |
| G7b | panel split, caption strip, prose gate (6.4) | split / strip / exclude |
| G7c | text block | reject if OCR word boxes cover ≥ 0.35 of crop ink **and** median word length ≥ 4 |
| G7d | pHash clustering (6.7) | 64-bit DCT pHash; exact duplicate = Hamming 0; near duplicate = Hamming ≤ 6; union-find gives `phash_cluster_id`; **cluster only, do not drop** |
| G8b | strict gatekeeper (paid, the only AI screening step; 2026-09-23): target type, readable, TikZ-representable; since 2026-09-25 the same answer names one descriptive `diagram_type` (`geometry`, `graph_plot`, `table_grid`, `number_line_scale`, `relational_schematic`, `mixed_other`) recorded for survivors only, never part of the decision | all three `true`; else `verifier_reject` (reject when in doubt) |
| G9 | → card builders (Section 7) | |

Maps and halftone plates fail G5; isolated formulas and ornaments fail G6; both deterministically and before any paid call. Diagram category is a **reporting variable only** and never an eligibility criterion; `other` and `unknown` are valid values.

**6.3.1 Border-truncation test** (on the dilated crop, after fusion, before any model call):

```text
binarize crop (Sauvola, window 25 px at 300-dpi-equivalent, k = 0.2)
for each edge: border_ink[edge] = ink pixels in the outermost 2-px strip / strip pixels
touching_cc = any 8-connected ink component intersecting the crop boundary with area >= 0.01 * total ink
if max(border_ink) > 0.02 or touching_cc:
    grow the crop by 0.10 * diagonal on the offending edge(s); re-test; at most 3 steps;
    stop if growth would overlap a detected text block or leave the page
if still failing: exclude, reason = border_truncation
```

Store `border_ink[4]`, `growth_steps`, `touching_cc` for every proposal.

Every threshold in this ladder is a preregistered design choice frozen on the Phase 0 engineering set, never retuned on the historical pool (Section 17.3).

### 6.4 Panels, captions, prose, OCR, and the context window

Three deterministic rules, applied in order at G7b on the binarized crop and page-level OCR word boxes (OCR runs once per page, not per crop).

1. **Panel split** (projection profile, recursive, depth ≤ 2): a gap band of zero ink at least `0.04 × max(crop_w, crop_h)` wide that partitions ink into two sides each holding ≥ 0.20 of total ink splits the crop; each panel becomes its own proposal with `panel_of` set; the parent is retired with reason `split_into_panels`; more than 4 panels excludes with `multi_panel_unresolved`.
2. **Caption and figure-number strip**: a horizontal band of OCR words wholly within the top or bottom 25 percent of the crop, with median character height within ±25 percent of the page's modal body-text height, and either ≥ 8 characters or matching the versioned figure-number pattern (`Fig.`, `Figure`, `Pl.`, `圖`, `图`, `第…圖`, `図`, `Hình`, `Abb.`, `Рис.` and variants; list authored once at Phase 0 per source language). The band is removed from the crop, `caption_stripped` is set, and the text is **appended to the Track B context**. Track A is therefore "crop only, with captions, figure numbers, and body prose automatically removed"; text printed inside the diagram (point labels, axis labels) remains part of Track A.
3. **Residual prose gate**: after stripping, a contiguous OCR run of ≥ 12 characters inside the crop that does not match the mathematical-label pattern (`^[A-Za-z]'{0,2}$ | ^[A-Za-z]_?\d{1,2}$ | ^[0-9]{1,3}(°|′|″)?$ | ^[Ͱ-Ͽ]$ | ^[A-Z]{2,3}$` plus a CJK single-character class) excludes the proposal with `prose_in_crop`. A gate, not a flag.

When `ocr_status == unavailable`, rules 2 and 3 fall back to their geometric components (band position and modal text height). **An item is never excluded solely because OCR failed.** Per page the pipeline records `ocr_engine`, `ocr_model_version`, `script_detected`, `ocr_mean_word_conf`, `ocr_word_count`, `ocr_status ∈ {ok, low_confidence, unavailable}`; the `ocr_status` distribution by language is published.

**Context window** (`ctxwin-1`, preregistered): text blocks whose boxes fall within ±1.5 × crop height on the same page, plus any block on the same or facing page containing a reference matching the figure number recovered by rule 2, concatenated in reading order, capped at 1,200 characters, truncated at block boundaries; the stripped caption is prepended. `context_char_count` and `context_block_count` are stored; the fraction of items with zero context is published and Section 11.7 reports context gain conditional on non-empty context. `context_image` is the page region spanning the crop and the selected blocks.

The crop supplied to card builders and to the judge is canonicalized by the Section 9.7 pipeline (`crop_canonical_image`). Crop assets are never mounted into the compile container.

### 6.5 Deterministic image statistics

All computed on the 8-bit grayscale crop and its Sauvola ink mask at G1, with pinned library versions (`stats_version`):

| Field | Definition | Degradation indexed |
|---|---|---|
| `eff_res_xheight_px` | page modal text x-height in px | low_resolution |
| `contrast_rms` | RMS contrast of the crop | fading, low_contrast |
| `ink_frac` | ink pixels / crop pixels | fading (and gate) |
| `stroke_w_mean`, `stroke_w_cv` | 2 × mean distance-transform value at skeleton pixels; its coefficient of variation | uneven printing pressure |
| `frag_index` | (# ink components with area < 0.005 × total ink) / total ink × 1000 | broken_strokes |
| `endpoint_density` | skeleton endpoints per unit skeleton length | broken_strokes (primary) |
| `midtone_frac` | pixels between the two Otsu modes / ink pixels | bleed_through, fading |
| `speckle_mad` | median absolute deviation over the paper region | grain, noise |
| `blur_norm` | variance of Laplacian ÷ `contrast_rms`² | blur |
| `skew_deg` | dominant orientation by projection-profile variance maximization, page level | skew |
| `label_contact_cc` | # components merging a text-height component with a long-stroke component | label_geometry_contact |

The raw statistic is always retained and **degradation analyses use the raw statistic**. The corpus-relative severity bins, their calibration against synthetic degradations (the former Section 12.4) and the monotonicity check on those degradations were removed on 2026-09-26 (checkpoint simplification): no synthetic degradation exists any more, so the statistics are descriptive and are not validated against a generating parameter. Bitonal sources leave `contrast_rms`, `blur_norm`, `midtone_frac`, `speckle_mad` undefined; such items form their own stratum and `source_codec` is a reported covariate in every degradation or language table.

### 6.6 The selection funnel

One append-only row per **proposal**, written incrementally as gates execute, never deleted.

```text
proposal_id, parent_proposal_id, book_id, page_index
box_source, box_xyxy_page_px, proposer_boxes[], iou_pairwise{}, max_edge_delta_frac, n_proposers_agreeing
img_stats{}                    # full Section 6.5 vector, computed at G1 for every row
border_ink[4], growth_steps, touching_cc
ocr_status, ocr_mean_word_conf, ocr_word_cover, caption_stripped, caption_text, prose_chars_in_crop
phash64, phash_cluster_id, cluster_size, is_cluster_representative
gates[]                        # ordered {gate_id, version, passed, measured_value, threshold}
terminal_gate, exclusion_reasons[]
model_calls[]                  # {stage, model_id, n_calls, input_tokens, output_tokens, unit_price_id}
category_pred, n_agreeing, verifier_pred
outcome                        # excluded | unscored_pool | eligible
item_id, pipeline_version, config_hash
```

Closed exclusion vocabulary (free text is forbidden):

```text
page_low_resolution, page_render_fail, page_blank,
box_too_small, box_too_large, aspect_extreme,
ink_too_low, ink_too_high, solid_block, halftone, photo_grayscale,
no_long_strokes, text_block, prose_in_crop,
border_truncation, multi_panel_unresolved, split_into_panels,
detector_disagreement, category_not_target, verifier_reject,
duplicate_representative_lost,
card_insufficient_facts, card_low_support, card_contradiction_rate, card_ambiguous_entities,
card_structure_coverage,
rights_unresolved
```

`degradation_above_ceiling` was removed with the degradation ceiling `s*` (checkpoint simplification 2026-09-26).

Funnel stages for reporting: S0 books cleared; S1 pages rendered; S2 proposals emitted; S3 proposals matched; S4 crops passing G8b (**the reference denominator for eligibility rate**); S5 cluster representatives; S6 cards built; S7 at least one eligible fact; S8 item eligible.

Required statistics, each with its denominator named: per-gate conditional pass rate (denominator = rows reaching the gate) and cumulative survival, reported separately; survival by book, flagging any book deviating by more than 2× from the median; exclusion rate by `category_pred` over rows reaching G8; a decile table of exclusion rate for each image statistic over all G1 rows, plus a book-clustered logistic model `logit P(excluded) ~ image statistics + book`; survival by language, decade, repository, `scanning_agent`, `source_codec`; the miss-rate lower bound of 6.2; model calls and cost by stage. The standardized mean difference between included items and all G1 proposals is published for every statistic; any `|SMD| > 0.2` is named as a limitation on the benchmark card and every degradation result is published with the included range of that statistic.

### 6.7 Duplicates and near-duplicates

At G7d the pipeline **clusters only**. After cards are frozen (Section 8), two items in one cluster are redundant if the Jaccard similarity of their eligible fact sets after entity canonicalization is ≥ 0.90 (labels compared case- and diacritic-insensitively). The representative is the item with the larger eligible count; ties go to lower `frag_index`, then lexicographically smallest `item_id`. Non-representatives are excluded with `duplicate_representative_lost`. Every member of a `phash_cluster_id`, whether or not it survives, is a hard edge in split-group construction (Section 13.1). "Unresolved duplicate" means a clustered proposal for which this rule has not executed; under this ordering it cannot occur and is treated as a pipeline bug if it does.

### 6.8 Candidate definition and pool sizing

"Candidate" = a proposal surviving G7d (post-dedup). "Plausible candidate" = a proposal surviving G8b. Pool sizing is not asserted: run the full ladder on the **development books first**, measure per-stage retention `r̂` through Section 8's item gates, and set the required plausible-candidate count as `N_cand = N* / r̂ × 1.25` where `N*` is the pilot size chosen in Section 13.7 and 1.25 is a preregistered safety factor. A shortfall is recovered only by extracting more pages or adding books from the census reserve, never by changing a frozen gate level.

### 6.9 Cost and rejected alternatives

The ladder is net cost-negative relative to the original design: most proposals are rejected before the first paid call, pHash runs before the verifier, and degradation is described by CPU-computed image statistics instead of a model-emitted severity. Rejected: a learned target-diagram classifier trained on the pilot corpus (forbidden, and it would import the selection bias into the gate that creates it); super-resolution or dewarping before cropping (alters the artifact under study; since the owner amendment of
2026-09-23 the items are Paddle's dewarped crops, so this rejection now stands only for the pipeline's own processing:
no further super-resolution, dewarping or re-cropping is applied, and the collector's dewarping is disclosed per item
rather than avoided); a third vision-language model or any second proposer (Paddle is the only region source, decisions D3/D4); calibration curves for category confidence (category is reporting-only); inverse-probability weighting of scores (would be fitted on the same unvalidated metadata that drives selection; the post-stratified estimate and envelope of Section 13.3 are used instead).
