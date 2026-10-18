"""Shared record schemas (pydantic v2). Field names follow the plan verbatim.

Sources: rights manifest §4.3/§4.5; item §5.2; submission §5.3; funnel §6.6; builder output
§7.4; fact record §8.5; CGX §10.2; verdicts §10.4; scores §11; ledger §16.4; release §15.2;
run manifest §15.3. Records are validated at every write boundary with `validate_record`.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ontology import ENTITY_TYPES, PREDICATES, OntologyError, entity_args, validate_fact
from .vocab import (AiUseClause, Anchor, Decidability, DiagramType, Exclusion,
                    ExtractionPath, FactClass, FunnelOutcome, JudgeQuestion,
                    EvaluatorFailure, OcrStatus, Outcome, ReleaseMode, Role, Split, Strictness,
                    Track, UsStatus, Verdict, Vote, WrapperClass, YesNoUnknown)

SHA = r"^[0-9a-f]{64}$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


# --------------------------------------------------------------------------- §4 rights
class Terms(Strict):
    terms_url: str
    terms_snapshot: str
    terms_snapshot_sha256: str = Field(pattern=SHA)
    terms_snapshot_utc: str
    repository_rights_statement_verbatim: str
    redistribution_permitted: YesNoUnknown
    commercial_reuse_permitted: YesNoUnknown
    ai_use_clause: AiUseClause
    required_attribution_string: Optional[str] = None


class Acquisition(Strict):
    source_repository: str
    source_url: str
    download_utc: str
    source_file_sha256: str = Field(pattern=SHA)
    source_file_bytes: int = Field(ge=1)
    page_count: int = Field(ge=1)
    source_codec: Literal["jpeg", "jp2", "tiff_g4", "png", "pdf_mixed"]
    source_bitdepth: Literal[1, 8, 24]
    source_colorspace: Literal["bitonal", "gray", "rgb"]
    scanning_agent: Optional[str] = None
    source_dpi_nominal: Optional[int] = None


class YearEvidence(Strict):
    catalog_record: str
    catalog_record_sha256: str = Field(pattern=SHA)
    titlepage_asset: str
    titlepage_asset_sha256: str = Field(pattern=SHA)
    titlepage_page_index: int = Field(ge=0)
    agree: bool


class Determination(Strict):
    us_status: UsStatus
    us_rule_id: Literal["us-pd-pre1931-v1"]
    us_determined_by: Literal["rule", "human"]
    source_country: str
    source_country_status: Literal["not_determined"] = "not_determined"   # R2: pipeline cannot change it
    release_country_status: Literal["not_determined"] = "not_determined"  # R2
    determined_by_model: Literal[False]                   # R1: true is a validation error


class Prominence(Strict):
    hosted_by_major_aggregators: int = Field(ge=0)
    public_fulltext_layer: bool


class RightsManifest(Strict):
    book_id: str = Field(pattern=r"^bk_[0-9a-f]{12}$")
    work_id: str = Field(pattern=r"^wk_[A-Za-z0-9_]+$")
    display_title: str
    author: str
    publisher: str
    publisher_id: str = Field(pattern=r"^pub_[A-Za-z0-9_]+$")
    edition_statement: str
    is_reprint_of_earlier_work: YesNoUnknown
    publication_year: int
    published: YesNoUnknown            # human-set; us-pd-pre1931-v1 requires "published" (§4.1)
    publication_place: str
    first_publication_country: str
    language: str
    acquisition: Acquisition
    publication_year_evidence: YearEvidence
    terms: Terms
    determination: Determination
    prominence: Prominence
    release_mode: ReleaseMode
    approved_by: Optional[str] = None
    approved_utc: Optional[str] = None
    manifest_version: Literal["rights-1"]
    manifest_sha256: str = Field(pattern=SHA)

    @model_validator(mode="after")
    def _invariants(self):
        if self.book_id != "bk_" + self.acquisition.source_file_sha256[:12]:
            raise ValueError("book_id must be bk_ + sha256(master download)[0:12]")
        if self.release_mode == ReleaseMode.full_public.value and self.terms.redistribution_permitted == "no":
            raise ValueError("redistribution_permitted: no cannot be full_public (§4.3)")
        if self.determination.us_status == UsStatus.public_domain.value and self.determination.us_determined_by == "rule":
            if self.publication_year > 1930 or self.published != YesNoUnknown.yes.value:
                raise ValueError("us-pd-pre1931-v1 needs year <= 1930 and published == yes (§4.1)")
        return self


# --------------------------------------------------------------------------- §5.2 item
class SourceCrop(Strict):
    """Provenance and disclosure of an item image: a Paddle saved crop used as-is (D3, 2026-09-23), with
    `block_bbox` in Paddle's preprocessed frame (not regenerable from the master); or an image of a
    pre-cropped source (`frame: precropped_image`), whose listing is the master and whose box is the whole image."""
    source: str
    snapshot_id: str
    file: str
    file_sha256: str = Field(pattern=SHA)
    block_bbox: list[int] = Field(min_length=4, max_length=4)
    frame: Literal["paddle_preprocessed", "precropped_image"]
    frame_wh: list[int] = Field(min_length=2, max_length=2)
    rotation_deg: Optional[int] = None
    modifications: list[Literal["rotated", "dewarped", "resampled", "jpeg_recompressed", "paddle_bounds",
                                "cropped_by_collector", "painted_over"]]
    # a pre-cropped source (no page, no Paddle output): the image file is the whole source page
    regenerable_from_master: bool
    precropped_origin: Optional[str] = None
    source_page: Optional[int] = None


class Extraction(Strict):
    pipeline_version: str
    render_dpi: int
    color_mode: str
    renderer_version: str
    proposal_id: str
    crop_bbox: list[int] = Field(min_length=4, max_length=4)          # Paddle frame (source_crop.frame)
    context_bbox: list[int] = Field(min_length=4, max_length=4)       # Paddle frame
    source_crop: SourceCrop
    ocr_engine_version: str
    ocr_status: OcrStatus
    caption_stripped: bool
    panel_of: Optional[str] = None
    context_char_count: int = Field(ge=0)
    context_block_count: int = Field(ge=0)
    context_window_rule: Literal["ctxwin-1"]


class SemanticCardSummary(Strict):
    asc_version: str
    card_uid: str = Field(pattern=r"^card_[0-9a-f]{16}$")
    generation_manifest: str
    generation_manifest_sha256: str = Field(pattern=SHA)
    facts: str
    facts_sha256: str = Field(pattern=SHA)
    union_mgs_count: int
    eligible_mgs_count: dict[str, int]
    eligible_geometric_count: int
    eligible_model_count: int
    abstained_count: int
    ambiguous_entity_count: int
    contradiction_rate: float
    context_assisted_fact_count: int


class Quality(Strict):
    eligible: bool
    exclusion_reasons: list[Exclusion] = []
    funnel_exit_stage: Optional[Literal["S2", "S3", "S4", "S5", "S6", "S7", "S8"]] = None
    phash_cluster_id: str
    is_cluster_representative: bool


class ItemRecord(Strict):
    id: str = Field(pattern=r"^hmb_[0-9a-f]{16}$")
    display_id: Optional[str] = None
    dataset_version: str
    source: dict[str, Any]          # book_id, work_id, rights_manifest_version, page_index, printed_page_label
    extraction: Extraction
    assets: dict[str, str]          # path fields each with a *_sha256 sibling (§5.1)
    automatic_metadata: dict[str, Any]
    semantic_card: Optional[SemanticCardSummary] = None
    quality: Quality
    split_group: Optional[str] = None
    split: Optional[Split] = None

    @field_validator("automatic_metadata")
    @classmethod
    def _diagram_type(cls, v: dict[str, Any]):
        dt = v.get("diagram_type")
        if dt is not None:
            label = dt.get("label") if isinstance(dt, dict) else "<not a record>"
            if label is not None and label not in {t.value for t in DiagramType}:
                raise ValueError(f"automatic_metadata.diagram_type.label {label!r} is not one of the six types")
        return v

    @field_validator("assets")
    @classmethod
    def _sibling_hashes(cls, v: dict[str, str]):
        for k, val in v.items():
            if k.endswith("_sha256"):
                if not re.match(SHA, val):
                    raise ValueError(f"assets.{k} is not a sha256")
            else:
                stem = k[:-len("_image")] if k.endswith("_image") and k != "page_image" else k
                sib = {"page_image": "page_image_sha256", "crop_image": "crop_image_sha256",
                       "crop_canonical_image": "crop_canonical_sha256",
                       "context_text": "context_text_sha256"}.get(k, f"{stem}_sha256")
                if sib not in v:
                    raise ValueError(f"assets.{k} lacks sibling {sib} (§5.1)")
        return v


# --------------------------------------------------------------------------- §6.6 funnel
class GateRecord(Strict):
    gate_id: str
    version: str
    passed: bool
    measured_value: Any = None
    threshold: Any = None


class FunnelRow(Strict):
    proposal_id: str
    parent_proposal_id: Optional[str] = None
    book_id: str
    page_index: int
    box_source: Optional[str] = None
    box_xyxy_page_px: Optional[list[int]] = None          # in `box_frame` coordinates (Paddle's frame)
    box_frame: Optional[Literal["paddle_preprocessed", "precropped_image"]] = None
    source_crop: Optional[SourceCrop] = None
    img_stats: dict[str, Optional[float]] = {}
    border_ink: Optional[list[float]] = None
    growth_steps: Optional[int] = None
    touching_cc: Optional[bool] = None
    ocr_status: Optional[OcrStatus] = None
    ocr_mean_word_conf: Optional[float] = None
    ocr_word_cover: Optional[float] = None
    caption_stripped: bool = False
    caption_text: Optional[str] = None
    prose_chars_in_crop: Optional[int] = None
    phash64: Optional[str] = None
    phash_cluster_id: Optional[str] = None
    cluster_size: Optional[int] = None
    is_cluster_representative: Optional[bool] = None
    gates: list[GateRecord] = []
    terminal_gate: Optional[str] = None
    exclusion_reasons: list[Exclusion] = []
    model_calls: list[dict] = []
    verifier_pred: Optional[str] = None
    diagram_type: Optional[DiagramType] = None          # descriptive, G8b survivors only (never a gate)
    outcome: Optional[FunnelOutcome] = None
    item_id: Optional[str] = None
    pipeline_version: str
    config_hash: str


# --------------------------------------------------------------------------- §7.4 builder output
class BuilderAnchor(Strict):
    x: float
    y: float
    w: float
    h: float


class BuilderEntity(Strict):
    local_id: str
    type: str
    label: Optional[str] = None
    anchor: BuilderAnchor

    @field_validator("type")
    @classmethod
    def _t(cls, v):
        if v not in ENTITY_TYPES:
            raise ValueError(f"entity type {v!r} not in ontology")
        return v


class BuilderFact(Strict):
    predicate: str
    args: list[Any]
    context_assisted: bool = False


class BuilderAbstention(Strict):
    scope: str
    entity: Optional[str] = None
    reason: str


class BuilderOutput(Strict):
    """Parsed, schema-checked builder card. OOV facts are moved to oov_dropped by the parser."""
    asc_version: str
    builder_id: str
    builder_family: str
    entities: list[BuilderEntity]
    facts: list[BuilderFact]
    abstentions: list[BuilderAbstention] = []
    oov_dropped: list[str] = []

    @model_validator(mode="after")
    def _facts_valid(self):
        ids = {e.local_id for e in self.entities}
        for f in self.facts:
            try:
                validate_fact(f.predicate, f.args)
            except OntologyError as e:
                raise ValueError(str(e)) from e
            undeclared = [a for a in entity_args(f.predicate, f.args) if a not in ids]
            if undeclared:
                raise ValueError(f"{f.predicate}{f.args}: undeclared entities {undeclared}")
        return self


# --------------------------------------------------------------------------- §8.5 fact record
class FactRecord(Strict):
    fact_id: str
    fact_class: FactClass = Field(alias="class")
    predicate: str
    args: list[Any]
    strictness: Strictness
    decidability: Decidability
    anchor: Anchor
    context_assisted: bool = False
    supporting_families: list[str] = []
    votes: dict[str, Vote] = {}
    eligible: bool
    ineligible_reasons: list[str] = []

    model_config = ConfigDict(extra="forbid", use_enum_values=True, populate_by_name=True)

    @model_validator(mode="after")
    def _ok(self):
        validate_fact(self.predicate, self.args)
        p = PREDICATES[self.predicate]
        got = (self.fact_class, self.strictness, self.decidability, self.anchor)
        want = (p.fclass.value, p.strictness.value, p.decidability.value, p.anchor.value)
        if got != want:
            raise ValueError(f"{self.predicate}: attributes {got} contradict §7.1 row {want}")
        return self


class CanonEntity(Strict):
    entity_id: str
    type: str
    label: Optional[str] = None
    anchor: Optional[BuilderAnchor] = None
    ambiguous: bool = False
    members: dict[str, str] = {}     # builder_id -> local_id


class Card(Strict):
    """Frozen card file `semantic_cards/<item_id>.json` (union MGS with eligibility per fact)."""
    item_id: str
    asc_version: str
    card_uid: str
    entities: list[CanonEntity]
    facts: list[FactRecord]
    abstained: list[dict] = []
    contradictions: list[list[str]] = []
    builder_order: list[str]
    family_map: dict[str, str]
    profile: str                     # fixture | production (§17.3 separation)


# --------------------------------------------------------------------------- §5.3 submission
class Attempt(Strict):
    index: int
    utc: str
    transport_error: Optional[str] = None


class SubmissionRecord(Strict):
    item_id: str
    dataset_version: str
    release_hash: Optional[str] = None
    model_id: str
    model_version_string: str
    model_family: str
    access_date: str
    provider_terms_version: str
    output_release: Literal["public", "withheld"]
    track: Track
    prompt_id: str
    prompt_sha256: str = Field(pattern=SHA)
    decoding_params: dict[str, Any]
    image_encoding: dict[str, Any]
    stop_reason: Optional[str] = None
    attempts: list[Attempt]
    harness_version: str
    sandbox_image_digest: Optional[str] = None
    raw_output: str
    raw_output_sha256: str = Field(pattern=SHA)
    extraction_path: ExtractionPath
    wrapper_class: Optional[WrapperClass] = None
    n_blocks: int = 0
    body_sha256: Optional[str] = None
    normalized_tex: Optional[str] = None
    normalized_tex_sha256: Optional[str] = None
    outcome_code: Outcome
    cpu_seconds: Optional[float] = None
    page_count: Optional[int] = None
    ink_fraction: Optional[float] = None
    pdf_sha256: Optional[str] = None
    render_sha256: Optional[str] = None
    cgx_record: Optional[str] = None
    cgx_sha256: Optional[str] = None
    judge_config_id: Optional[str] = None
    verdicts: Optional[str] = None
    verdicts_sha256: Optional[str] = None
    scores: Optional[str] = None
    scores_sha256: Optional[str] = Field(default=None, pattern=SHA)

    @model_validator(mode="after")
    def _ok_requires_artifacts(self):
        if self.outcome_code == Outcome.OK.value:
            for f in ("body_sha256", "pdf_sha256", "render_sha256", "normalized_tex_sha256"):
                if not getattr(self, f):
                    raise ValueError(f"outcome OK requires {f}")
            if self.page_count != 1:
                raise ValueError("outcome OK requires page_count == 1")
        return self


# --------------------------------------------------------------------------- §10.2 CGX
class CgxPrimitive(Strict):
    prim_id: str
    type: str                        # entity type from ontology, or "curve_generic"
    geom: dict[str, Any]             # coordinates in PDF points after CTM
    label: Optional[str] = None


class CgxFact(Strict):
    cfact_id: str
    predicate: str
    args: list[Any]
    margin: Optional[float] = None   # tolerance margin in eps units (hidden positives need >= 3)


class CgxRecord(Strict):
    extractor_version: str
    item_id: str
    model_id: str
    track: Track
    pdf_sha256: str = Field(pattern=SHA)
    diagonal_D: float
    primitives: list[CgxPrimitive]
    glyphs: list[dict]               # {text, bbox}
    labels: list[dict]               # {text, bbox, bound_prim}
    facts: list[CgxFact]
    path_op_count: int
    primitives_over_cap: Optional[int] = None   # C43: primitives found when above cgx_max_primitives (then none kept)


# --------------------------------------------------------------------------- §10.4-10.7 verdicts
class JudgeVote(Strict):
    judge_id: str
    judge_family: str
    question: JudgeQuestion
    positive: Optional[bool]         # None = missing/unparseable -> contributes nothing
    raw_call_sha256: Optional[str] = None


class FactVerdict(Strict):
    fact_id: str
    verdict: Verdict
    source: Literal["extractor", "judges", "deterministic"]
    votes: list[JudgeVote] = []
    binding: Optional[dict[str, Any]] = None
    evidence: Optional[str] = None


class VerdictFile(Strict):
    item_id: str
    card_uid: Optional[str] = None      # binds stored verdicts to the exact card they judged (B6 #10)
    model_id: str
    track: Track
    judge_config_hash: str
    evaluator_failure: Optional[EvaluatorFailure] = None   # undecided facts are INDETERMINATE
    facts: list[FactVerdict]
    alignment_ambiguous: list[str] = []
    profile: str


# --------------------------------------------------------------------------- §11 scores
class ScoreRecord(Strict):
    item_id: str
    model_id: str
    track: Track
    outcome_code: Outcome
    evaluator_failure: Optional[EvaluatorFailure] = None
    S: float                          # PRESENT / scorable strict facts, 0 unless OK (§11.3)
    per_class: dict[str, dict[str, Any]]
    metric_hash: str
    profile: str


# --------------------------------------------------------------------------- §16.4 ledger
class LedgerEvent(Strict):
    event: Literal["reserve", "settle", "release"]
    reservation_id: Optional[str] = None
    utc: str
    phase: str
    stage: str
    model_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None    # prompt-cache tokens, billed apart from input_tokens
    cache_write_input_tokens: Optional[int] = None   # (absent on rows before 2026-10-01 and when zero)
    image_count: Optional[int] = None
    wall_ms: Optional[int] = None
    unit_price_id: Optional[str] = None
    usd: float = 0.0                  # reserved estimate (reserve) or computed cost (settle)
    cache_hit: bool = False
    note: Optional[str] = None
    lane: Optional[Literal["on_demand", "batch"]] = None   # pricing lane; None = on-demand (rows before 2026-09-26)


# --------------------------------------------------------------------------- §3.1 / G1 audit log
class HumanAction(Strict):
    """One row of `layout.human_audit_log` (append-only). G1 counts rows with scope == "item"."""
    utc: str
    actor: str
    scope: Literal["protocol", "book", "phase", "item"]
    action: str                       # e.g. approve_book, cp1_form_freeze_approval
    item_id: Optional[str] = None     # required when scope == "item"
    book_id: Optional[str] = None
    phase: Optional[str] = None
    description: str

    @model_validator(mode="after")
    def _item_scope(self):
        if self.scope == "item" and not self.item_id:
            raise ValueError("scope=item requires item_id")
        if self.scope != "item" and self.item_id:
            raise ValueError("an item_id makes this a per-item intervention; use scope=item")
        return self


# --------------------------------------------------------------------------- model calls
class ModelSpec(Strict):
    """A benchmark-roster model (builder, judge, contestant...). Never an engineering agent."""
    model_id: str
    # Bedrock transports only (D14, 2026-09-25): Runtime Converse, Mantle Chat Completions; "fake" = tests.
    provider: Literal["bedrock", "bedrock_mantle", "fake"]
    family: str
    role: Role
    snapshot: Optional[str] = None
    unit_price_id: Optional[str] = None
    decoding: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0}
    # Reasoning effort pinned per model (roster decision 2026-09-23); None = no effort control sent.
    # Part of the spec, so it is in every cache key and recorded with results.
    effort: Optional[Literal["low", "medium", "high"]] = None   # "low": owner decision 2026-09-26 (Sonnet 5 writer)
    # "text_only": the model is never sent an image (§2.7(c)/(d) floor, e.g. DeepSeek V3.1).
    input_mode: Literal["vision", "text_only"] = "vision"
    # Transport base URL where the provider needs one per model (bedrock_mantle: the model card's
    # In-Region endpoint URL, e.g. https://bedrock-mantle.us-east-1.api.aws/openai/v1 for Gemma 4).
    endpoint: Optional[str] = None
    # Native constrained decoding for this role (repair 2026-09-26): "native_json_schema" sends the
    # caller's JSON Schema through the route's documented field (Mantle `response_format`, Converse
    # `outputConfig.textFormat`); None = the schema is stated in the prompt only. One value per role
    # (roster-check); never set for contestants (raw TikZ). Omitted from the cache key when None.
    structured_output: Optional[Literal["native_json_schema"]] = None


class ModelRequest(Strict):
    spec: ModelSpec
    system: Optional[str] = None
    text: str
    image_paths: list[str] = []
    image_sha256: list[str] = []
    max_tokens: int
    purpose: str                      # e.g. "card_generation", "Q1", "contestant"
    phase: str                        # ledger phase (PILOT, DEV, TEST, FINAL)
    cache_scope: str                  # evaluator-relevant version string folded into the cache key
    # {"name": str, "schema": JSON Schema}; sent natively only when spec.structured_output is set.
    response_schema: Optional[dict[str, Any]] = None


class ModelResponse(Strict):
    text: str
    thinking: Optional[str] = None
    stop_reason: Optional[str] = None
    model_version_string: str
    input_tokens: int = 0
    output_tokens: int = 0
    image_count: int = 0
    wall_ms: int = 0
    attempts: list[Attempt] = []
    cache_key: str
    cache_hit: bool = False
    fake: bool = False                # True only for explicitly marked test providers
    output_constraint: Optional[str] = None   # "native_json_schema" | "prompt" (how the reply was constrained)


def validate_record(model: type[BaseModel], data: dict) -> BaseModel:
    """Validate at a write boundary; raises pydantic.ValidationError with a clear message."""
    return model.model_validate(data)
