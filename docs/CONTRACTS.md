# HistMathBench v2 — shared engineering contracts (Scaffold 0)

Owner: lead engineer. Workers change the files listed here only by asking the lead.
The plan and `HistMathBench_v2_components/*.md` govern scientific behaviour; this file governs
module boundaries. Section numbers (§) refer to the plan.

> **Checkpoint simplification 2026-09-26.** The Track C control suite (CP3: control generation and
> CAL/VAL partitions, degradation and severity calibration, mutation algebra, card arm, `s*`, `w_p`,
> `controls_hash`) and evaluator validation (CP5: V1–V11, gates G3–G5) were deleted. `hmb/controls/` was
> renamed `hmb/synthetic/` (known-geometry fixtures for offline tests, plus the production shared predicate
> semantics `hmb.synthetic.geo_predicates` and `hmb.synthetic.spec.Geo`); `hmb/validation/` is gone. Milestones
> MS1–MS5 replace CP1–CP11; the pre-freeze contexts are `PILOT` and `DEV`. `release_hash` has no controls term.

## Environment
* Python: `.venv/Scripts/python.exe` (CPython 3.12, deps pinned in `requirements.lock`). Never
  install into or upgrade the global Python. Add a dependency only with a stated reason, via
  `uv pip install --python .venv/Scripts/python.exe <pkg>==<ver>`, then refresh `requirements.lock`.
* Tests: `.venv/Scripts/python.exe -m pytest tests/<your_dir> -q`. Markers: `docker` (needs the
  sandbox image; skip cleanly when unavailable), `live` (paid calls; deselected by default and
  never run in this build).
* Windows: force UTF-8 stdout (lessons L10); in Git Bash prefix docker commands that take
  POSIX paths with `MSYS_NO_PATHCONV=1` (L13); PowerShell wraps native stderr as failure (L2).
* This directory is its own git repository (`main`, remote `origin`). `corpus/`, `hmb_data*/` and
  `configs/*.yaml` are gitignored, so an overwrite there is unrecoverable.

## Shared modules (lead-owned, import them, do not edit)
| module | provides |
|---|---|
| `hmb.vocab` | every closed vocabulary: `Outcome` (§9.5), `OUTCOME_GROUPS`, `Verdict`, `ExtraClass` (§10.4), `CellStatus`, `NotScoredReason` (§10.8), `Exclusion` (§6.6), `FunnelOutcome`, rights enums (§4), `Track`, `Split`, `FactClass`/`CLASS_LETTER`, `Strictness`, `Decidability`, `Anchor`, `Vote`, `GoverningRule`, `JudgeQuestion`, `Role`, `ParamStatus`, `Profile`, `GateResult`, `Decision` |
| `hmb.ontology` | asc-1.0.0 predicate table `PREDICATES` (§7.1), `ENTITY_TYPES`, `ENUMS`, `validate_fact`, `canonical_args`, `fact_key(pred,args)` (hashable canonical identity), `mgs_sort_key`, `entity_args`, `PREDICATE_RANK`, `ALLOWED_VARIATION` (§7.6) |
| `hmb.hashing` | RFC 8785 `canonical_json`, `sha256_bytes/json/file`, `H(*parts)` chain hash (§15.1), `book_id`, `item_id`, `card_uid` (§5.1) |
| `hmb.store` | `Layout` (all artifact paths, §15.7/§5.2/§5.3), `write_immutable` (identical re-write ok, different bytes → `ImmutableConflict`), `write_json_immutable`, `write_mutable` (explicit working files only), `append_jsonl`/`iter_jsonl` (append-only), `FreezeViolation`, first-test-artifact marker |
| `hmb.config` | parameter `REGISTRY` (every named plan parameter, its § and kind), `Thresholds.value(name)`; `fixture_thresholds()`; `production_template()`; `load_thresholds`; `level_freeze` (end of MS3); `resolve_thresholds_arg` (`fixture`, a path, or `prefreeze:PILOT\|DEV`); `PREFREEZE_CONTEXTS`, `prefreeze`, `execution_context`/`prefreeze_record`, `set_provisional`, `resolve_artifact_provenance` (2026-09-25). **Production profile raises `CalibrationMissing` for any non-frozen, non-form-constant parameter, except a provisional parameter on the list of the named pre-freeze context it is opened in. Never hard-code a registered number; read it from `Thresholds`.** If you need a parameter that is not registered, ask the lead to add it (with its §). |
| `hmb.schemas` | pydantic records: `RightsManifest`, `ItemRecord`, `FunnelRow`, `BuilderOutput`, `FactRecord` (alias `class`), `CanonEntity`, `Card`, `SubmissionRecord`, `CgxRecord`/`CgxPrimitive`/`CgxFact`, `JudgeVote`, `FactVerdict`, `ExtraVerdict`, `VerdictFile`, `ScoreRecord`, `LedgerEvent`, `ModelSpec`, `ModelRequest`, `ModelResponse` |
| `hmb.cli` | `python -m hmb ...`; component modules expose `register_cli(subparsers)`; module names in `CLI_MODULES` |

## Profiles and evidence
* Every card, verdict file and score record carries `profile` (`fixture` | `production`).
  Fixture results are engineering evidence only; gate code treats them as `NOT_RUN`.
* Pre-freeze contexts (2026-09-25; renamed 2026-09-26): after the MS1 approval and before the level
  freeze, the root's production thresholds can be opened as `prefreeze:PILOT` (MS2 pilot) or
  `prefreeze:DEV` (MS3 development set), both on development books only. The `CAL` context was deleted
  with the controls. Only the provisional parameters listed for that
  context are readable (plan starting values, or `thresholds-provisional` values for calibrated ones).
  Each run appends `manifests/prefreeze_runs.jsonl` (stage, partition, values read, thresholds,
  registry and form-freeze identity, outputs with sha256); artifacts embed `execution_context` or a
  `levels` block. A book processed in MS2/MS3 is logged in `sources/development_books.jsonl` and
  `items-finalize` refuses a split that would put it in the test split. `items-finalize`, releases,
  `model-run`, `t3-probe` and `score-report` never run in a pre-freeze context; the contexts close
  at the level freeze or the first test artifact. `thresholds-freeze` accepts `{artifact, field}`
  provenance records and verifies that the artifact is a pre-freeze development artifact.
* Fake provider responses exist only in tests (`ModelResponse.fake = True`); production paths
  refuse a fake response.
* Missing evidence → `GateResult.NOT_RUN` / `BLOCKED`, never `PASS`; missing statistics → `None`
  with a reason, never `0`.

## Cross-component function contracts
Signatures are binding; internals are the owner's choice. Pure functions preferred.

### Model calls — `hmb.providers` (owner: Opus 1) and `hmb.ledger` (owner: Opus 5)
```python
# hmb.providers
class LiveCallsDisabled(RuntimeError): ...
def call_model(req: ModelRequest, layout: Layout, ledger: "Ledger | None", *,
               allow_live: bool, provider_override=None) -> ModelResponse
```
* Cache key = `sha256_json` of the request with `image_paths` replaced by `image_sha256`,
  including `spec` (model id, snapshot, decoding), `max_tokens`, `cache_scope`. Stored immutably
  under `layout.model_call_cache`. A hit returns `cache_hit=True` and charges nothing. Any change
  to an evaluator input changes the key (invalidation by construction).
* Miss with `allow_live=False` → `LiveCallsDisabled`. Miss with live: `ledger.reserve` **before**
  dispatch, ≤ `transport_retry_max` (3) retries on transport error / HTTP 5xx / rate limit only,
  every attempt logged, then `ledger.settle` with provider-reported usage (or `release` on final
  failure). Parse/format failures are never retried here (callers apply the protocol's rules).
* `provider_override` is the injection point for explicitly marked fake providers in tests.
* Providers (D14 closed 2026-09-25): only `bedrock` (bedrock-runtime Converse, client pinned to
  us-east-1) and `bedrock_mantle` (Chat Completions on Bedrock's OpenAI-compatible endpoint through
  the `openai` SDK as a client library; base URL = `ModelSpec.endpoint`, which must be a
  `https://bedrock-mantle.<region>.api.aws/` URL; Bedrock API key `AWS_BEARER_TOKEN_BEDROCK` only).
  The direct `anthropic`/`openai`/`gemini` connectors are removed; `ModelSpec`, `load_roster` and
  `adapter_for` refuse those names. Standard (default) service tier; no fallback model or transport.
  `effective_inference(spec, max_tokens)` records the controls actually sent and the unsupported ones.
* Locked contestants (`hmb.roster.LOCKED_CONTESTANTS`, owner decision 2026-09-26, superseding the
  4-model lock of 2026-09-25; Kimi K2.5 removed 2026-09-26 when it became an answer-key writer): one image model
  per family plus a small anchor, 5 in all — Llama 4 Maverick (Converse `us.` profile), Nemotron Nano 12B v2 VL,
  Ministral 3 14B (Converse
  in-region), Gemma 4 26B A4B and Gemma 4 E2B (the small anchor; Mantle `…/openai/v1`) — plus DeepSeek V3.1 `deepseek.v3.1` (Mantle `…/v1`), text only, never the
  direct-figure track (A). Each `Route` carries its family. `roster-check` also refuses a contestant that
  is a grader model or whose family writes the answer key (Anthropic, OpenAI, xAI, Moonshot); same-company-as-grader
  models compete and are judged by the other two families (§10.9).
  `roster-check` refuses any addition, alias, substitution, route or family change; `model-run` refuses Track A
  for a text-only model and, under production, a contestant that is not its locked entry; the run
  manifest records `route` (id, transport, endpoint, region, tier, capability flags) and
  `effective_inference`.
* Per-model settings (2026-09-23), all part of the cache key: `ModelSpec.effort` (`medium`|`high`)
  is sent as Mantle `reasoning_effort`; a Converse model must encode it in
  `decoding.additional_model_request_fields`. A text-only model (`input_mode == "text_only"`, or a
  locked route without image input): `call_model` refuses any request with an image before the cache,
  the ledger or any transport.
* Per-model request shape (repair 2026-09-26, live-verified; `hmb.roster.MODEL_REQUEST_SHAPE`):
  GPT-6 Luna and Sol accept no sampling control (non-default `temperature` is rejected), so their roster
  entries carry only their effort field. **Global CRIS (owner decision and live parity, 2026-09-26):** they
  run as `global.openai.gpt-6-luna` / `global.openai.gpt-6-sol` on provider `bedrock` (bedrock-runtime
  Converse, `maxTokens`, effort `{"reasoning": {"effort": …}}`, Luna's schema as `outputConfig.textFormat`);
  `roster-check` refuses the retired `openai.gpt-6-*` (Mantle) and `us.openai.gpt-6-*` ids and a `global.`
  GPT-6 id on any other provider (`hmb.roster.RETIRED_ROUTES`). Global CRIS may process prompts and outputs
  in any supported commercial AWS Region. The per-model Mantle `mantle_token_field` mechanism remains for
  any listed Mantle model. Claude Sonnet 4.5 on Converse rejects `temperature` together with `top_p`, so only one
  may be configured. The adapter refuses a violating request before dispatch and `roster-check` reports
  it. A shape entry is part of that model's cache key; unlisted models keep the transport default.
  A Converse effort must be the family's own field in `decoding.additional_model_request_fields`
  (Claude Opus 5.5: `{"thinking": {"type": "adaptive"}, "output_config": {"effort": …}}`; Grok 4.6:
  `{"reasoning_effort": …}`; GPT-6 on Converse: `{"reasoning": {"effort": …}}`, not `reasoning_effort`).
* Structured output (repair 2026-09-26): `ModelSpec.structured_output == "native_json_schema"` plus a
  caller-supplied `ModelRequest.response_schema` (`{"name", "schema"}`) is sent as Mantle
  `response_format: {"type": "json_schema", "json_schema": {…, "strict": true}}` or Converse
  `outputConfig.textFormat`. Without the spec flag the schema is never sent (prompt-stated only); there
  is no silent fallback. `ModelResponse.output_constraint` records `native_json_schema` or `prompt`.
  One mode per role (`roster-check`); contestants never (raw TikZ). Both new fields enter the cache key
  only when set, so earlier cache entries keep their keys. Mantle `message.reasoning` (Gemma 4) is kept
  as `ModelResponse.thinking`. Transport read timeout `READ_TIMEOUT_S` = 600 s on both transports.
```python
# hmb.ledger  (spending tracked, not capped: owner decision D2, 2026-09-23)
class Ledger:
    def __init__(self, layout: Layout): ...
    def estimate_usd(self, req: ModelRequest, unit_price_id: str | None = None) -> float   # worst case
    def batch_price(self, model_id: str) -> Price             # newest explicit lane: batch entry, else PriceMissing
    def reserve(self, phase: str, stage: str, model_id: str, est_usd: float, *, lane: str | None = None) -> str
    def settle(self, reservation_id: str, *, input_tokens: int, output_tokens: int,
               image_count: int, wall_ms: int, unit_price_id: str, lane: str | None = None,
               cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float
    def release(self, reservation_id: str) -> None
    def spent(self, phase: str | None = None) -> float
```
A reservation is logged before dispatch and settled or released after it (thread-safe). There is
no cap, halt or phase authorization. The ledger is `layout.ledger` (append-only JSONL of
`LedgerEvent`); prices come from `layout.pricing` with dated entries, and an unpriced model raises
`PriceMissing`. An estimate is not a billing guarantee; say so in any report. Pricing lanes (2026-09-26): a
rate-card entry has `lane: on_demand` (default) or `lane: batch`; a batch rate is its own dated entry with the
same `model_id`, never derived from an on-demand rate. Ledger rows carry `lane: batch` for batch records
(`lane` absent = on-demand). Prompt-cache tokens (2026-10-01, after the September invoice showed $44.79 of
unmetered GPT-6 Sol / 6.1 Sol cache writes): Converse `inputTokens` excludes `cacheReadInputTokens` and
`cacheWriteInputTokens`, which AWS bills on their own SKUs; Mantle `prompt_tokens` include
`prompt_tokens_details.cached_tokens`, which the adapter moves to cache reads. Both settle at the entry's
`usd_per_mtok_cache_read` / `usd_per_mtok_cache_write`, or at its input rate with a `note` when the entry has
none, and settle rows carry `cache_read_input_tokens` / `cache_write_input_tokens`. `estimate_usd` prices the
prompt at max(input, cache-write) rate, so it stays a worst case.

### Batch inference — `hmb.batch` (2026-09-26)
```python
# hmb.providers
class BatchDeferred(LiveCallsDisabled):  keys: frozenset[str]     # raised only under an installed router
def set_router(router) -> None                                     # hmb.batch.scheduler.HybridRouter or None
def converse_request(req, images) -> dict; converse_result(resp, req) -> dict   # shared sync/batch body
def store_batch_result(req, out, layout, *, key, utc, transport) -> ModelResponse
# hmb.batch
capability.decide(req, matrix=None) -> RouteDecision("batch" | "synchronous", reason, route_id, min_records)
jobs.prepare_job / submit_job / refresh_job / collect_job / cancel_job      # manifests under manifests/batch/
scheduler.HybridRouter(store, ledger, Policy(name in sync|fixed|window, fill_target, window_seconds, ...))
scheduler.HybridScheduler(router, backend, workers=, poll_seconds=).run({task_id: replayable callable})
```
* **Transport only.** A batch record is the synchronous Converse body (`converse_request`) of a request
  `call_model` would have sent; its answer is written as that request's ordinary cache entry (same key, same
  `ModelResponse`; the entry adds `transport: {lane: batch, job_name, record_id, …}`), and the caller is replayed
  from the cache. Prompts, parsers, scoring, gates, roster, efforts and the form freeze are untouched.
* **Fail-closed routing.** Only `final_route: batch` entries of `hmb/batch/capability_matrix.json` (loader-
  enforced: a completed, retrieved, parsed live Converse-job probe; `batch_structured_output` also proven when
  the role needs native schema) with an explicit dated batch rate are batched. Unknown routes, Mantle routes (no
  batch surface) and structured-output-blocked roles stay synchronous; `batch-plan` shows the reason per route.
* **Deferral.** Under a router, a batched cache miss (or a key another call is answering) raises
  `BatchDeferred` before anything of the caller is written. Judge stage 1 / stage 2 / Q3 primary slots and the
  two writers' generation and verification calls collect every independent deferral before re-raising, so one
  batch round carries a whole stage; order-dependent steps (escalation, substitution) wait for answers. With every answer cached the result is byte-identical to a synchronous run (tests).
* **Failures.** Job-level failure, record service error (retryable or not), missing output: not cached, the
  reservation released, the outcome recorded once; a retryable record is resubmitted to batch within
  `transport_retry_max`; otherwise the replay raises `ProviderError` (the caller's existing failure path). No
  synchronous fallback after a submitted record failed. Truncated or invalid answers are answers (cached).
* **Idempotence.** Record id = cache key; out-of-order, duplicate and unknown output lines are handled once;
  `prepared` jobs resubmit with the same clientRequestToken; collecting twice settles nothing twice.
* **Scheduling.** Cells (contestant call -> extraction -> serialized compile (C24, `sandbox._COMPILE_LOCK`) ->
  Q3 -> Q1 -> score) run as independent tasks on a bounded worker pool with per-model synchronous
  concurrency limits; a parked cell resumes as soon as its keys settle; cross-model reports run after all cells.

### Closure / MGS — `hmb.cards.closure` (owner: Opus 2; deliver first, others import it)
```python
FactKey = tuple   # ontology.fact_key(pred, args)
def forward_chain(facts: Iterable[FactKey], entity_types: dict[str, str]) -> set[FactKey]
def contradictions(closed: set[FactKey], entity_types: dict[str, str]) -> list[tuple[FactKey, FactKey]]
def minimal_generating_set(facts: Iterable[FactKey], entity_types: dict[str, str]) -> list[FactKey]
```
Rule set R and exclusion table E exactly as §8.2. Derived-only facts (`radius_of`, …) never
appear in the MGS. The CGX extractor and the known-geometry oracle graph use the same functions.

### Oracle graph (known-geometry fixtures, offline tests) — produced by `hmb.synthetic` (owner: Opus 5)
```jsonc
{"spec_id": "...", "family": "triangles_polygons", "seed": 17, "partition": "fixture",
 "canvas": {"w": 10.0, "h": 10.0},                 // TikZ units (cm)
 "entities": [{"entity_id": "p_A", "type": "point", "label": "A", "geom": {"x": 1.0, "y": 2.0}}],
 "facts": [{"predicate": "on", "args": ["p_A", "s_1"]}],   // MGS under hmb.cards.closure
 "tikz": "\\begin{tikzpicture}...\\end{tikzpicture}"}
```
`hmb.cards.oracle_card(oracle_graph) -> Card` (Opus 2) turns an oracle graph into a `Card` with
every fact eligible (direct oracle-card input in the offline tests, used to isolate downstream
correctness; `Ceiling_oracle` was removed 2026-09-26). The extractor round trip `hmb cgx-roundtrip`
(`hmb.synthetic.roundtrip`, formerly "V1x DEV") compiles generated known-geometry TikZ in the sandbox and
compares the CGX record with the geometry that produced it; no model call. It is engineering evidence: it
is not a gate, sets no level and supports no benchmark claim.

### Geometric predicate semantics (one convention for Γ, CGX, builder/judge prompts)
The oracle evaluator (Opus 5), the CGX extractor (Opus 3) and the builder/judge prompt texts
(Opus 2, Opus 4) all use this convention; otherwise the extractor round trip would measure a convention
mismatch rather than extractor error. The shared production implementation is
`hmb.synthetic.geo_predicates` over `hmb.synthetic.spec.Geo`. Decided by the lead on 2026-09-22 (form choice; plan §7.1 is silent):
* `intersect_at(x,y,p)`: only when p is in the relative interior of both x and y (a transversal
  crossing). Tangency points, endpoint meetings and T-junctions are expressed by `tangent_to`,
  `endpoint_of` and `on`.
* `parallel` and `perpendicular` hold over linear entities (segment, line, ray, axis) only.
  Collinear overlapping pairs are not parallel.
* Layout predicates (`left_of`, `above`, `inside`, `outside`, `same_side`, `opposite_sides`,
  `order_along`) hold between **labelled point** entities, with an ε_on tie margin (no fact inside
  the margin). `order_along` needs ≥ 3 points on a non-closed 1-D entity, with direction
  canonicalized so that seq[0] < seq[-1]. `orientation` applies to segment, line and ray.
  **asc-2.0.0 (2026-09-26):** layout is not a card class. Γ and CGX still compute these predicates (the
  E-exclusions behind CONTRADICTED read them), but no card, writer prompt, oracle card or round-trip count
  contains them (`ontology.CARD_CLASSES` = E, R, L).
* Label binding: nearest point within `label_bind_frac·D`, else the nearest non-point entity (§10.3).
* **Point entities (added 2026-09-22 after the first V1x run, C28).** A point entity exists only where
  the drawing gives it identity: it carries a name label, or it is drawn as a dot or marker.
  Unlabelled, unmarked endpoints, vertices, crossings and contact points are internal geometry. They
  are used to compute relations among real entities, but they are never point entities themselves,
  and no fact takes them as an argument. Γ, CGX and the builder/judge prompts all follow this rule.
* **Straight strokes (C28).** Classification uses the stroke's extreme incident point entities:
  - point entities at both ends → `segment`;
  - a point entity at exactly one end and the stroke continuing past the last point at the other end → `ray`;
  - no point entity at either end, with the stroke extending past its extreme incident points by more than ε_on → `line`.
  An axis is a line or segment carrying an arrow tip or an axis label. Γ derives the type from this
  same geometric test on the drawn stroke, not from the generator's intent.
* **Polygons (C30/C33/C34).** Every polygon edge is also a segment entity. A polygon entity is a
  simple cycle of straight strokes, after merging collinear (180°) corners into one side, that has no
  chord between its corners and at most 8 corners. There is ONE rule, independent of how the cycle
  was drawn (one closed path or separate strokes) and of whether its corners are labelled; §7.6
  source_refactor invariance requires this (C34 replaced C33's closed-path special case). A drawing
  whose outer cycle has chords, such as a square with both diagonals, therefore yields only its
  chordless sub-cycles. This is documented (the card-coverage instruments that measured the effect
  were removed 2026-09-26).
  Γ and CGX share one function, `hmb.cgx.polygons_from_strokes`.
* The shared format's `marker` field on points is always explicit (true = drawn dot/marker);
  readers treat a missing key as false.
* Where practical, the oracle evaluator and the CGX fact generator call ONE shared pure function
  over a common entity-geometry format, with tolerances passed in (exact for Γ, the §10.2
  tolerances for CGX).

### Track values
`Track.A`/`Track.B` are historical tracks. `Track.C` is only the harness label of generated
known-geometry fixture renders in the offline tests (since 2026-09-26 there is no Track C control suite and
it is never a benchmark track). Track C records never enter A/B aggregation (`REPORTED_TRACKS`).

### G1 audit log
`layout.human_audit_log` rows are `schemas.HumanAction`. G1's numerator counts distinct items that
have any row with `scope == "item"`. Protocol, book and phase rows (approvals) are permitted human
work (§3.1) and are not counted. The collapse guard is a standing rule in code
(`sampling.COLLAPSE_RULE`, D7), so it produces no human row.

### Submission and compilation — `hmb.submission` (owner: Opus 3)
```python
def extract_candidate(raw: str, stop_reason: str | None) -> Extracted   # §9.2; Extracted.outcome set on failure
# truncation: stop_reason in TRUNCATION_STOP_REASONS (Converse `max_tokens`, Mantle `length`) -> E_TRUNCATED
# before E_NO_TIKZ (extract-9.2-v2); the raw stop_reason is kept in the submission record
def normalize(ex: Extracted, th) -> Normalized                          # §9.3 wrapper classes, P_POLICY_REJECT
def compile_in_sandbox(tex: bytes, th, workdir: Path) -> CompileResult  # §9.6; never runs TeX on the host
def canonicalize_image(img_or_png_bytes, th) -> tuple[bytes, dict]      # §9.7 steps 1-6; shared with crops
def process_submission(raw: str, stop_reason, *, item_id, model: ModelSpec, track, layout, th,
                       meta: dict) -> SubmissionRecord                  # writes all §5.3 artifacts
```
If the sandbox image is unavailable, `compile_in_sandbox` raises `SandboxUnavailable`; there is
no host fallback.

### CGX extraction — `hmb.cgx` (owner: Opus 3)
```python
def extract_cgx(pdf_path: Path, *, item_id, model_id, track, th) -> CgxRecord   # §10.2
```

### Adjudication and grading (owner: Opus 4)
```python
# hmb.adjudication
def adjudicate(card: Card, cgx: CgxRecord, render_png: Path, crop_png: Path | None, *,
               judges: list[ModelSpec], layout, th, ledger, allow_live, provider_override=None
               ) -> VerdictFile                  # §10; majority of judge votes, every fact gets a verdict
# hmb.grading
def score_cell(card: Card, sub: SubmissionRecord | None, outcome: Outcome,
               verdicts: VerdictFile | None, th) -> ScoreRecord   # §11.3: S = g·PRESENT/scorable strict facts
# hmb.grading.tables — headline (group-balanced mean + IM t-interval), two diagnostic families (validity;
#   per-class recall E/R/L with contradicted/indeterminate shares, Q5, evaluator-failure rate),
#   the all-pairs BH partial order and `separation_evidence` (G7, D21: each Track A pair's BH separation,
#   difference and MDD floor; no named pair), context effect (G8), ceiling flag; hmb.grading.cli writes the gate
#   evidence, the descriptive `by_diagram_type` table, the §7.3 `context_rescore`, and `dev-sizing` (MS3; was `cp7-sizing`)
# hmb.stats — the IM group t (the one inferential method), BH partial order, MDD, §13.7 sizing simulation
```
* Judge questions (verdict schema `hmb-judge-verdict-2`, 2026-09-26): Q1 (model-class facts, render only)
  and Q3 (ambiguous bindings). Q2 and the extras ladder are removed; nothing precision-side is judged.
  Answers carry no `evidence` string. A reply is read answer by answer (`judges.parse_response`): an entry
  with an unknown id or a wrongly typed answer is skipped, an id answered twice differently is dropped, a
  reply cut at max_tokens keeps its complete entries; a skipped or missing answer is not a vote (the third
  judge is asked). A slot is attempted once (no identical-prompt retry); if it yields no usable answer it
  goes once to a judge of another family.
* Judge failure isolation (repair 2026-09-26): statements two judges decided are final and never depend
  on a later call; the third family is asked only about statements without two concurring answers.
  A failed slot (the judge and its substitute failed: no usable answer, refused, or a
  transport/provider error after `call_model`'s retries) is recorded in `BResult.failures`; only the
  statements it left undecided become INDETERMINATE, each with `binding.failed_judge_calls`. An
  authentication/authorization failure (`hmb.providers.AuthError`: HTTP 401/403, AccessDenied, expired key) is
  never a judge failure: it propagates and stops the run (resume after fixing the credential);
  (judge, family, status, detail, cache keys of the failed attempts); the cell carries
  `evaluator_failure`. Q3 follows the same rule (`Q3PartialFailure` keeps every agreed answer).

### Judge serialization — `hmb.adjudication.serialize` (frozen form, repair 2026-09-26)
The only code that turns a card into judge-visible text: Q1 statements (`verbalize`, `TEMPLATES`),
the ENTITIES block, the Q3 entity list, questions and candidate-object lines, and the native verdict
JSON Schema. `serializer_identity()` = `SERIALIZER_VERSION` + sha256 of the module source; it is part of
the `judge_prompts_and_verdict_schema` form-freeze component and of `judge_config_hash`.
* **Identity.** `EntityNames(card)`: an entity whose (type, normalized printed label) is unique in the card
  is named by its label (`point “A”`); every other entity (unlabelled or sharing its label) gets a typed
  opaque identifier `<prefix><n>` (`P` point, `S` segment, `L` line, `R` ray, `C` circle, `AR` arc, `EL`
  ellipse, `PG` polygon, `RG` region, `AX` axis, `AW` arrow, `AM` angle marker, `RA` right-angle mark,
  `LT` length tick, `CV` curve, `SD` solid). Numbers follow a canonical sort per type: incidence to
  uniquely-labelled entities, own label, rounded card anchor, then the canonical entity id. Names never
  depend on object identity, dict order, file paths, model output order or the contestant's drawing,
  and are the same in Q1 and Q3, for every judge and on reruns.
* **ENTITIES block.** Every identifier used in a call is defined, together with every identifier its
  definition uses: type, printed label, and the eligible GEO incidence facts (endpoints, vertices,
  centre, edges, arcs, points on it) that are not statements of the same call. Without incidence, the
  left-to-right rank among same-type opaque entities by anchor. Entities the card cannot distinguish
  (same colour under 1-WL refinement of the incidence graph, e.g. symmetric asymptotes) are declared
  as such (`twins`), never ranked. The block never contains a statement under test, a model-decidable
  fact, an ineligible fact, a verdict, a hidden-positive fact or a weight.
* **Uniqueness.** `check_unique` refuses (`StatementCollision`) two facts with different canonical
  identity (predicate + args) that would render with identical wording.
* **Native schema.** `verdict_json_schema(question, ids, version)`: `{schema, answers[{id ∈ call ids,
  answer: bool | object: str}]}`, closed objects. It has no `refusal` field (a required
  refusal boolean beside answers changed the prompt's meaning); a constrained judge refuses through
  the API refusal signal. G8b's native schema is `acquisition.gates.gatekeeper_json_schema()`.

### Cards (owner: Opus 2)
Answer-key writers (owner decision 2026-09-26): **2** builders in 2 families (`hmb.roster.N_BUILDERS`):
GPT-6 Sol (`global.openai.gpt-6-sol`, Global CRIS Converse, effort medium as
`{"reasoning": {"effort": <effort>}}`; `configs/roster.yaml` pins it; it replaced Claude Sonnet 5 late on 2026-09-26) and Kimi
K2.5 (`moonshotai.kimi-k2.5`, Converse in-Region; no documented effort control: `MODEL_REQUEST_SHAPE`
`effort_control: False`). `cards-build --batch-config` answers Kimi's calls by batch jobs (hmb.batch).

**asc-2.0.0 answer key (2026-09-26 streamlining; `reports/answer_key_judge_streamlining_execution_20260926.md`).**
* Generation, two staged calls per writer (`build.GENERATION_PROTOCOL = "staged-entities-facts-1"`, 2026-09-26
  repair; `reports/answer_key_judge_repairs_execution_20260926.md`): the entity call (`builder_generation_entities`)
  returns only entity lines (`{"id", "type", "label"?, "param"?, "label_from_context"?, "anchor": [x, y, w, h]}`);
  the fact call (`builder_generation_facts`) receives that writer's own parsed entity lines in its user text and
  returns only fact lines (`{"fact", "args"}`) over the card relations and `measure_label`. Each parse reads only
  its own kind of line from its own reply, so neither call can exceed the other's cap or introduce an entity; a
  fact on an undeclared id is abstained. Existence and `label` facts are derived from the entity lines. Hard caps
  `card_max_entities` (40) and `card_max_facts` (60): later lines are discarded and counted. Every line parses on its
  own, so a reply cut at `max_tokens` keeps its complete lines. A stage whose reply ends at the limit with no
  complete line is asked exactly once more with a different request (`prompts.retry_note`: at most half the cap,
  first line immediately); a writer with no entity line gets no fact call and still verifies. Every generation call
  is logged in `votes.generation[<writer>].calls` (stage, attempt, stop reason, output tokens, usable lines); the
  manifest lists `truncated_calls` and `empty_truncated_generation_calls` (cut with no usable line).
* Verification, chunked: every writer is asked about every union fact, in chunks of at most
  `card_verify_chunk_size` (20) statements; a fact in the closure of its own generation counts as its support
  (`generated`) unless it votes false on it (self-veto, the preregistered fallback: asking only about the other
  writer's facts measured P_card 0.566 against the 0.568 floor),
  answering one `<id> true|false|abstain` line per statement (line-level salvage; missing = abstain).
* Rule (§8.4, fixed for the two-writer roster): eligible iff both writers support the fact (generated and not
  vetoed, or voted true), neither voted false, no E-violation, no ambiguous argument, anchor B uniquely bindable. There is no
  family-count parameter, no LOBFO, no card instability, SC or UCR.
* Cache identity: every writer call has `cache_scope = "asc-2.0.0|writer-call"`; the request completes the key, so
  gate levels, thresholds and one writer's settings never re-bill the other's calls. A fact-call request carries
  its own writer's entity lines, so it changes only when those do. `asc_version` hashes what determines card
  content (vocabulary, rules, normalization, alignment, writers, max_tokens, caps, prompts, generation protocol,
  consensus rule), not gate levels.
* Card vs oracle: removed 2026-09-26 with the control suite (`hmb/cards/coverage.py`, `card_vs_oracle`, `P_card`,
  `M_card`, `R_card`, `card_map_coverage_floor`). Card precision and coverage are not measured (disclosed limitation).
```python
def build_card(item_id: str, crop_canonical: Path, ocr_words: list[dict], context_text: str | None, *,
               builders: list[ModelSpec], layout, th, ledger, allow_live, provider_override=None) -> Card
def consensus(al, U, votes, family_map, *, asc_version, profile) -> Card   # pure, no I/O
def item_gates(card: Card, th) -> list[Exclusion]   # §8.7 (no item_stats: the s* ceiling was removed 2026-09-26)
```

### Known-geometry fixtures and gates (owner: Opus 5)
```python
# hmb.synthetic (renamed from hmb.controls 2026-09-26): known-geometry generators, predicate evaluator,
#   TikZ and render for offline tests; `roundtrip` = `hmb cgx-roundtrip` (no model call, not a gate);
#   production: geo_predicates + spec.Geo (the shared predicate semantics above). No partitions, no
#   degradation or mutation operators, no card arm; hmb.validation is deleted.
# hmb.gates: G1, G2, G6, G7, G8, G9 + §17.6 decision table (NOT_RUN/BLOCKED propagate to UNDECIDABLE);
#   blocking G1 and G7, modulating G2, G6, G8; G9 is INFORMATIONAL (no X_V1 ceiling, D2), so SCALE needs
#   G2, G6 and G8 all to pass. G3-G5 were deleted with CP5; STOP_PUBLISH_METHODS no longer exists.
#   Milestone exits ms1_exit..ms5_exit and the readiness check books_ready (R3-approved test books form
#   at least G* split groups; checked before MS4).
```

### Acquisition, rights, sampling, manifests (owner: Opus 1)
```python
# hmb.rights: us-pd-pre1931-v1, manifest draft (mechanical fields only), R3 gate (value-checked)
# hmb.acquisition: collection adapter (completed-artifact boundary; Paddle's saved `image` and `chart`
#   crops are the items, D3/D4/D13), render (page asset + G0), G0/G2-G8 (no G1; G7a records border
#   measures and never excludes, D10; a figure-only page is recorded by G0b and never excluded for
#   having no text, C21 2026-09-25; G8b also returns one descriptive `DiagramType`, never a gate),
#   image statistics on the crop,
#   pHash, context window from Paddle text blocks, funnel rows, item records with `source_crop`
#   (collector file + sha256, box in Paddle's preprocessed frame, disclosed modifications)
# hmb.sampling: split groups (hard/soft edges, collapse guard = standing rule D7 with the D12 floor), dedup representatives,
#   funnel report (+ crop-border statistics, D10); S_adj and envelope exist as optional functions
# hmb.manifests: §15.1 hash chain (release_hash = H(cardset, split, thresholds, judge_config, harness,
#   metric); no controls_hash since 2026-09-26), release.json (+ data_licence, D9), run.json, tombstones,
#   version trigger check; cp1_approval_problems (release-build and ms1_exit require the owner's
#   approval of the current forms, recorded before the level freeze)
# hmb.stages: the stage commands (setup = MS1, pilot = MS2, develop = MS3, freeze = MS3 close / level
#   freeze, build = MS4, run and report = MS5); each runs the single commands in order through
#   hmb.cli.main and stops at the first failure; pilot runs in prefreeze:PILOT and develop in
#   prefreeze:DEV (develop also runs `dev-measure`); the calibrate and controls stages are gone
```
