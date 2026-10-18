# Component 02 — Source rights, provenance, item schema, and immutable artifacts

Part of *HistMathBench Zero-Annotation Project Plan v2*. Canonical entry point: [`../HistMathBench_Zero_Annotation_Project_Plan_v2.md`](../HistMathBench_Zero_Annotation_Project_Plan_v2.md). This file owns **Section 4** (source, rights, provenance, licensing) and **Section 5** (dataset item schema and immutable artifacts). Global notation is defined in the top-level plan, Section 3.4.

| Component card | |
|---|---|
| Purpose | Decide which books may enter, record why, and make every downstream artifact content-addressed so a frozen release is verifiable. |
| Inputs | Repository downloads, catalog records, repository terms pages, the title-page render of each book. |
| Mechanism | Per-book rights manifest with a value-checked extraction gate; hashed evidence bundle; content-derived identifiers; normalized item and submission records. |
| Outputs | `sources/rights_manifest.yaml`, `sources/evidence/<book_id>/`, item records, submission records, release manifest entries. |
| Gates | R3 extraction gate (Section 4.5). |
| Retained artifacts | Everything in Section 5.4, all hashed. |
| Cost tier | Negligible compute. |
| Lifecycle | Per book, one time. Nothing per item, per model, or recurring. |
| Human effort | One approval per book (title-page glance plus recording the repository terms). This is source curation, which the zero-annotation contract permits. |
| Dependencies | Section 6 (acquisition populates the extraction block), Section 8 (card hashes), Section 9 (submission fields), Section 13 (split groups), Section 15 (release manifest). |
| Downstream decisions | Whether a book is extracted at all; what the benchmark card may claim about rights; whether a release is reproducible. |

> **Checkpoint simplification 2026-09-26.** The Track C control suite (CP3) and evaluator validation (CP5) were
> deleted. For this component: the item record no longer carries `degradation` severity bins or
> `out_of_control_support` (5.2), and the release manifest has no controls entry (Section 15.2). Synthetic
> known-geometry fixtures exist only in offline tests and carry no rights burden (4.4).

---

## 4. Source, rights, provenance, and licensing model

### 4.1 Historical window and the rule that is actually applied

The V1 core window is first publication **1850–1930**.

The only rights determination the project computes by rule is **United States copyright status**, and it rests on two verified facts:

- The U.S. Copyright Office states that all works published in the United States before 1 January 1931 are in the public domain, and Circular 15A applies the same 95-year arithmetic to works already under statutory protection ([copyright.gov overview](https://www.copyright.gov/what-is-copyright/); [Circular 15A](https://www.copyright.gov/circs/circ15a.pdf)).
- For **foreign** works, restoration under the Uruguay Round Agreements Act gives a restored work "the remainder of the term of copyright that the work would have enjoyed if the work had never entered the public domain in the United States", and the U.S. term for works published before 1978 is 95 years from first publication ([Circular 38B](https://www.copyright.gov/circs/circ38b.pdf)). A foreign work first published before 1931 therefore has no remaining U.S. term in 2026 even if it was restored. The Cornell public-domain chart (version current as of 1 January 2026) reaches the same conclusion for works first published outside the United States before 1931 ([Cornell chart](https://guides.library.cornell.edu/copyright/publicdomain)).

Rule `us-pd-pre1931-v1`: `us_status = public_domain` if and only if the evidenced first-publication year is at most 1930 **and** the work was published (not unpublished). Everything else is `not_determined`.

What the project does **not** determine, and records as `not_determined` by default:

- copyright status in the source country (term rules there are author-death-based, and the project will not attempt to establish author death dates);
- status in the country of release or of the hosting server;
- any inference from a repository's own rights label, which is a record of what the repository displayed, not a determination.

A **post-1930 work** may enter only with an affirmative, human-recorded basis (an open licence or government-work status). A computed public-domain basis is never available for it. The 1938 seed work named in the original handoff is therefore excluded from V1 by rule.

### 4.2 Source priorities

Prioritize diagram-dense mathematics: plane and solid geometry, trigonometry, analytic and coordinate geometry, construction geometry, elementary calculus and curve texts, arithmetic with geometric measurement, and mathematical dictionaries with concentrated diagrams. Initial languages may include Chinese, Japanese, Vietnamese, English, French, German, Russian, and Latin. Chinese and Vietnamese material may be sampled more heavily, subject to the split-group and slice rules in Section 13.

At census time, rank candidate books on **five** criteria: (1) rights cleanliness under Section 4.1; (2) repository terms under Section 4.3 (a repository that imposes no downstream restriction outranks one that does, at equal quality); (3) diagram density; (4) grayscale or colour source scans over bitonal (Section 6.1); (5) download reliability.

### 4.3 Repository terms are a separate layer from copyright

The copyright status of a 1925 text says nothing about the contractual terms accepted when downloading a scan of it. The manifest records the terms **verbatim and hashed**, plus three human-set fields:

```yaml
terms:
  terms_url: "..."
  terms_snapshot: evidence/<book_id>/terms.html
  terms_snapshot_sha256: "..."
  terms_snapshot_utc: "..."
  repository_rights_statement_verbatim: "..."
  redistribution_permitted: yes | no | unknown
  commercial_reuse_permitted: yes | no | unknown
  ai_use_clause: none | academic_only | licensed | unknown
  required_attribution_string: null | "..."
```

The reviewer of record verified two examples that show why this layer must exist: Gallica's conditions of use require a specified attribution string, charge for commercial reuse, and carry an explicit clause on use in AI and data-mining projects that is free only for non-profit academic research ([Gallica conditions](https://gallica.bnf.fr/edit/und/conditions-dutilisation-des-contenus-de-gallica)); the Internet Archive states that it "does not make guarantees as to the copyright status of items on archive.org" ([archive.org rights help](https://help.archive.org/help/rights/)). Neither fact is a legal determination by this project; both are reasons to record terms rather than assume them.

`release_mode ∈ {full_public, server_only, link_only, excluded}` is set per book by a human from these fields. A book with `redistribution_permitted: no` cannot be `full_public`.

### 4.4 Licensing layers

| # | Layer | Governs | Recorded in |
|---|---|---|---|
| L1 | Underlying work copyright | The historical text | `determination` block, per jurisdiction |
| L2 | Digitisation and repository terms | The scan, any database right asserted by the institution, the download contract | `terms` block, per book |
| L3 | Derived text (OCR, context) | Machine-derived text from L2 material | Inherits L2; flagged when L2 has an AI clause |
| L4 | Machine-generated cards and metadata | Project-created annotations | One project-wide data licence, decided at census exit (Section 17) |
| L5 | Code, schemas, prompts | Project software | Apache-2.0 |
| L6 | Model outputs | Contestant responses | Per-submission `output_release` and `provider_terms_version` |

> **Owner decision D9, 2026-09-23:** the L4 data licence is **Apache-2.0**. `release.json` carries it in
> `data_licence`, next to L5 and a pointer stating that scans and crops follow the per-book L1–L3 terms. The owner's
> statement that all source works are public domain does not replace the per-book R3 determination.

Two standing rules: **the release licence for scans and crops is the intersection of L1–L3 across all included books, not a project choice**; and **no layer inherits another's permissions**. The known-geometry fixtures of the offline tests are synthetic, are never benchmark items and carry no rights burden. External terminology dictionaries are **not used anywhere in the V1 pipeline** (Section 7.5), so no reference-resource layer exists.

### 4.5 Book-level rights manifest and the extraction gate

One record per book, written once, approved by a human, hashed.

```yaml
book_id: bk_<12 hex>                 # sha256 of the master download, truncated
work_id: wk_<id>                     # shared across editions, translations, reprints of one work
display_title: "..."
author: "..."
publisher: "..."
publisher_id: pub_<id>
edition_statement: "..."
is_reprint_of_earlier_work: yes | no | unknown
publication_year: 1925
publication_place: "..."
first_publication_country: CN
language: zh

acquisition:
  source_repository: "..."
  source_url: "..."
  download_utc: "..."
  source_file_sha256: "..."
  source_file_bytes: 0
  page_count: 0
  source_codec: jpeg | jp2 | tiff_g4 | png | pdf_mixed
  source_bitdepth: 1 | 8 | 24
  source_colorspace: bitonal | gray | rgb
  scanning_agent: "..." | null
  source_dpi_nominal: 0 | null

publication_year_evidence:
  catalog_record: evidence/<book_id>/catalog.json
  catalog_record_sha256: "..."
  titlepage_asset: evidence/<book_id>/titlepage.png
  titlepage_asset_sha256: "..."
  titlepage_page_index: 0
  agree: true                         # false => book excluded before extraction

terms: { ...as in Section 4.3... }

determination:
  us_status: public_domain | in_copyright | not_determined
  us_rule_id: us-pd-pre1931-v1
  us_determined_by: rule | human
  source_country: CN
  source_country_status: not_determined
  release_country_status: not_determined
  determined_by_model: false          # schema invariant; true is a validation error

prominence:                           # Section 13.6 contamination bound
  hosted_by_major_aggregators: 0
  public_fulltext_layer: true | false

release_mode: full_public | server_only | link_only | excluded
approved_by: "<human id>"
approved_utc: "..."
manifest_version: rights-1
manifest_sha256: "<hash of this record with this field removed>"
```

Three rules attach to the manifest:

- **R1 (model boundary).** A model may populate only mechanical fields: page count, hashes, the year as printed on the title page, verbatim transcription of a rights statement, and the arithmetic `publication_year < 1931`. Every field in `determination` and `release_mode` is set by a human or by the named deterministic rule. `determined_by_model: true` fails schema validation.
- **R2 (default is not determined).** `source_country_status` and `release_country_status` default to `not_determined` and cannot be changed by the pipeline.
- **R3 (extraction gate, value-checked).** Pages of a book are rendered only if
  `determination.us_status == public_domain` **and** `release_mode != excluded` **and** `publication_year_evidence.agree == true` **and** `terms.redistribution_permitted != no` **and** `approved_by` is set. A presence check is not a gate.

`publication_year_evidence.agree` requires the catalog year and the title-page year to match. The title-page crop is produced automatically; a human looks at it once per book. Disagreement excludes the book; nobody edits the year.

### 4.6 Evidence bundle per book

Stored at `sources/evidence/<book_id>/` with every hash listed in the manifest: (1) hash and byte count of the master download; (2) the repository item page as served at download time; (3) the repository terms page as served at download time; (4) the catalog record and the title-page image. Per-book cost is one scripted fetch pass and roughly one megabyte. A Wayback Machine submission of pages (2) and (3) is optional and recorded if made.

### 4.7 Benchmark-card sentence (mandatory)

> Rights status was determined once per source book. The determination covers United States copyright status only and rests on the first-publication year evidenced by the volume's own title page and catalog record together with the term rules summarised in U.S. Copyright Office Circulars 15A and 38B. Status in the source country and in any other jurisdiction was not determined. Repository terms of use are recorded separately from copyright status and reproduced verbatim in `sources/rights_manifest.yaml`; where a repository conditions reuse, that condition applies to the corresponding items independently of their public-domain status.

---

## 5. Dataset item schema and immutable artifacts

### 5.1 Content-derived identifiers

Every path-valued field has a sibling `*_sha256` over the bytes as stored. Identifiers are derived from content so that re-running an unchanged pipeline yields the same ids and any change to the evaluation target changes the id:

```text
book_id  = "bk_"   + hex(sha256(source_file_bytes))[0:12]
page_uid = "pg_"   + hex(sha256(book_id + ":" + page_index))[0:12]
item_id  = "hmb_"  + hex(sha256(book_id + ":" + page_index + ":" + crop_bbox_px))[0:16]
card_uid = "card_" + hex(sha256(canonical_json(eligible_fact_set) + asc_version))[0:16]
```

Canonical JSON is RFC 8785 (JSON Canonicalization Scheme) over UTF-8 NFC strings. A human-readable `display_id` (`hmb_000001`) may be kept but is never authoritative.

### 5.2 Item record

Book-level fields are **not** copied into the item; they are joined through `book_id` and `rights_manifest_version`. `rights_manifest_sha256` is the `manifest_sha256` of the book's rights record that the run's R3 check read (added 2026-09-23 so the approved record, not only its schema tag, is pinned). Undefined scalars from the original design (`crop_confidence`, `card_confidence`, model-emitted `severity`) are removed; their replacements are defined where they are produced.

```yaml
id: hmb_4a91c0de77b21f08
display_id: hmb_000001
dataset_version: v1.0

source:
  book_id: bk_9f2c1a77b0de
  work_id: wk_0031
  rights_manifest_version: rights-1
  rights_manifest_sha256: "<manifest_sha256 of the approved record>"
  page_index: 147
  printed_page_label: "141" | null

extraction:                          # produced by Section 6
  pipeline_version: extract-1.0.0
  render_dpi: 400
  color_mode: gray8
  renderer_version: "..."
  proposal_id: "..."
  crop_bbox: [x0, y0, x1, y1]         # page pixels
  context_bbox: [x0, y0, x1, y1]
  proposer_ids: [det_a@1.0, det_b@1.0, cc@1.0]
  ocr_engine_version: "..."
  ocr_status: ok | low_confidence | unavailable
  caption_stripped: true | false
  panel_of: null | "<parent proposal id>"
  context_char_count: 0
  context_block_count: 0
  context_window_rule: ctxwin-1

assets:
  page_image: pages/<book_id>/0147.png
  page_image_sha256: "..."
  crop_image: crops/<item_id>.png
  crop_image_sha256: "..."
  crop_canonical_image: crops_canonical/<item_id>.png   # Section 9.7 canonicalization, judge input
  crop_canonical_sha256: "..."
  context_image: context/<item_id>.png
  context_image_sha256: "..."
  context_text: context/<item_id>.txt
  context_text_sha256: "..."

automatic_metadata:                  # produced by Section 6
  crop_agreement:
    n_proposers_agreeing: 3
    iou_pairwise: {det_a__det_b: 0.81, det_a__cc: 0.66, det_b__cc: 0.64}
    max_edge_delta_frac: 0.06
    agreement_rule_version: crop-agree-1
    border_ink: [0.004, 0.001, 0.000, 0.003]
    growth_steps: 0
  diagram_category:
    value: circle_geometry | ... | other | unknown    # reporting variable only, never an eligibility criterion
    n_agreeing: 2
  image_stats:                       # deterministic; Section 6.5
    stats_version: imgstat-1
    eff_res_xheight_px: 17.4
    contrast_rms: 0.213
    ink_frac: 0.081
    stroke_w_mean: 2.6
    stroke_w_cv: 0.41
    frag_index: 12.7
    endpoint_density: 0.084
    midtone_frac: 0.19
    speckle_mad: 0.031
    blur_norm: 0.62
    skew_deg: 0.8
    label_contact_cc: 2
  # `degradation` severity bins and `out_of_control_support` (V11) were removed with the
  # control suite (checkpoint simplification 2026-09-26); degradation analyses use the raw statistics above.

semantic_card:                       # produced by Sections 7-8
  asc_version: asc-1.0.0+<hash>
  card_uid: card_77b1e2aa90c41d03
  generation_manifest: manifests/cards/<card_uid>.json
  generation_manifest_sha256: "..."
  facts: semantic_cards/<item_id>.json
  facts_sha256: "..."
  union_mgs_count: 34
  eligible_mgs_count: {entity: 6, relation: 8, label: 5, layout: 3}
  eligible_geometric_count: 17
  eligible_model_count: 5
  abstained_count: 3
  ambiguous_entity_count: 0
  contradiction_rate: 0.0
  union_coverage_ratio: 0.65
  structure_coverage_proxy: 0.81
  context_assisted_fact_count: 1
  governing_rule_per_category: {entity: llr, relation: llr, label: fallback, layout: llr}

quality:
  eligible: true
  exclusion_reasons: []              # closed enum, Section 6.6
  funnel_exit_stage: null | S2..S8
  phash_cluster_id: dup_0042 | singleton
  is_cluster_representative: true

split_group: sg_03                   # Section 13.1
split: development | test | reserved
```

### 5.3 Submission record (per item, model, track)

```yaml
item_id: hmb_4a91c0de77b21f08
dataset_version: v1.0
release_hash: "sha256:..."
model_id: "..."
model_version_string: "<provider-reported>"
model_family: "..."
access_date: "..."
provider_terms_version: "..."
output_release: public | withheld
track: A | B
prompt_id: reconstruction-A-v2
prompt_sha256: "..."
decoding_params: {temperature: 0.0, top_p: 1.0, max_tokens: 0, seed: null}
image_encoding: {format: png, width: 1024, height: 1024, crop_sha256: "..."}
stop_reason: "..."
attempts: [{index: 1, utc: "...", transport_error: null}]
harness_version: harness-1.0.0
sandbox_image_digest: "sha256:..."
raw_output: submissions/<model>/<track>/<item_id>.txt
raw_output_sha256: "..."
extraction_path: fenced | bare_doc | bare_pic | none
wrapper_class: D | P | B | null
n_blocks: 1
body_sha256: "..."                   # Section 9.4 invariant
normalized_tex: submissions/<model>/<track>/<item_id>.tex
normalized_tex_sha256: "..."
outcome_code: OK | ...               # Section 9.5
cpu_seconds: 0.0
page_count: 1
ink_fraction: 0.0
pdf_sha256: "..."
render_sha256: "..."
cgx_record: cgx/<model>/<track>/<item_id>.json      # Section 10.2
cgx_sha256: "..."
judge_config_id: judge-1
verdicts: verdicts/<model>/<track>/<item_id>.json
verdicts_sha256: "..."
scores: scores/<model>/<track>/<item_id>.json
```

### 5.4 Immutability and retained artifacts

Retained for every scored item: the source-page image, crop, canonical crop, context image and text, the funnel row (Section 6.6), every builder's raw generation and verification output, the union card, the frozen eligible card, per-fact votes with builder families, the extraction record, compile log, PDF, render, extractor record, every judge call's raw JSON, verdicts, and scores. All are hashed into the release manifest (Section 15.2). A benchmark release never rewrites any of them; a change produces a new release hash.

### 5.5 What this component does not do

It does not attempt source-country legal determinations, per-item rights records, or automated parsing of repository terms into permissions. It records verbatim text and three human-set booleans, and it keeps the claim narrow enough for the evidence to support it.
