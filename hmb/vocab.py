"""Closed vocabularies shared by every HistMathBench component.

Every string here is copied verbatim from the v2 plan; free text is never an
acceptable substitute. Sources are cited per block.
"""
from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# §9.5 outcome codes (mutually exclusive, exhaustive); order = table order.
class Outcome(StrEnum):
    E_NO_OUTPUT = "E_NO_OUTPUT"
    E_TRUNCATED = "E_TRUNCATED"
    E_NO_TIKZ = "E_NO_TIKZ"
    E_MULTI_CANDIDATE = "E_MULTI_CANDIDATE"
    P_POLICY_REJECT = "P_POLICY_REJECT"
    C_ERROR = "C_ERROR"
    C_TIMEOUT = "C_TIMEOUT"
    C_WALLCLOCK = "C_WALLCLOCK"
    C_RESOURCE = "C_RESOURCE"
    C_MISSING_GLYPH = "C_MISSING_GLYPH"
    C_SANDBOX_VIOLATION = "C_SANDBOX_VIOLATION"
    V_MULTIPAGE = "V_MULTIPAGE"
    V_RASTER = "V_RASTER"
    V_EMPTY = "V_EMPTY"
    V_OVERSIZE = "V_OVERSIZE"
    OK = "OK"


# §9.5 derived-rate membership.
OUTCOME_GROUPS: dict[str, frozenset[Outcome]] = {
    "Valid@1": frozenset({Outcome.OK}),
    "Compile@1": frozenset({Outcome.OK, Outcome.V_MULTIPAGE, Outcome.V_RASTER,
                            Outcome.V_EMPTY, Outcome.V_OVERSIZE}),
    "Unsafe/Invalid@1": frozenset({Outcome.P_POLICY_REJECT, Outcome.C_SANDBOX_VIOLATION,
                                   Outcome.V_RASTER}),
    "Timeout@1": frozenset({Outcome.C_TIMEOUT}),
    "Extraction-failure@1": frozenset({Outcome.E_NO_OUTPUT, Outcome.E_TRUNCATED,
                                       Outcome.E_NO_TIKZ}),
    "Protocol-violation@1": frozenset({Outcome.E_MULTI_CANDIDATE, Outcome.V_MULTIPAGE}),
}


# §10.4 recall-side verdicts.
class Verdict(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    CONTRADICTED = "CONTRADICTED"
    INDETERMINATE = "INDETERMINATE"
    EXEMPT = "EXEMPT"


# §10.8 evaluator failure recorded on a cell (simplification 2026-09-25: every cell is scored; facts
# the evaluator could not decide are INDETERMINATE and count as not recovered; the failure rate is
# reported per model).
class EvaluatorFailure(StrEnum):
    judge_failure = "judge_failure"            # both families malformed/refused
    judge_refusal = "judge_refusal"


# §6.6 closed exclusion vocabulary (free text forbidden).
class Exclusion(StrEnum):
    page_low_resolution = "page_low_resolution"
    page_render_fail = "page_render_fail"
    page_blank = "page_blank"
    box_too_small = "box_too_small"
    box_too_large = "box_too_large"
    aspect_extreme = "aspect_extreme"
    ink_too_low = "ink_too_low"
    ink_too_high = "ink_too_high"
    solid_block = "solid_block"
    halftone = "halftone"
    photo_grayscale = "photo_grayscale"
    no_long_strokes = "no_long_strokes"
    text_block = "text_block"
    prose_in_crop = "prose_in_crop"
    border_truncation = "border_truncation"
    multi_panel_unresolved = "multi_panel_unresolved"
    split_into_panels = "split_into_panels"
    detector_disagreement = "detector_disagreement"
    category_not_target = "category_not_target"
    verifier_reject = "verifier_reject"
    duplicate_representative_lost = "duplicate_representative_lost"
    card_insufficient_facts = "card_insufficient_facts"
    card_low_support = "card_low_support"
    card_contradiction_rate = "card_contradiction_rate"
    card_ambiguous_entities = "card_ambiguous_entities"
    card_structure_coverage = "card_structure_coverage"
    rights_unresolved = "rights_unresolved"
    # Extension (BUILD_STATUS C11): G2's physical-size test cannot run without a known DPI.
    physical_size_unknown = "physical_size_unknown"
    # Extension (BUILD_STATUS C14): §8.7 moves excess items of a book over the 25% cap to the
    # unscored pool by a deterministic rule but names no reason code.
    book_share_cap = "book_share_cap"
    # Extension (owner 2026-09-28, `card-select`): before card building, a candidate that is another scan of a
    # multi-scan work or a non-kept member of a pHash cluster (precard_duplicate), or that falls outside its
    # split group's seeded quota under the owner's candidate budget (selection_budget).
    precard_duplicate = "precard_duplicate"
    selection_budget = "selection_budget"


# §6.6 funnel outcome and stages.
class FunnelOutcome(StrEnum):
    excluded = "excluded"
    unscored_pool = "unscored_pool"
    eligible = "eligible"


# One descriptive primary diagram type per diagram that survives the G8b gatekeeper (owner decision
# 2026-09-25). Analysis metadata only: never a gate, inclusion rule, quality judgment, score input or
# quota. `mixed_other` absorbs genuinely mixed and uncertain cases; there is no `unknown` class.
class DiagramType(StrEnum):
    geometry = "geometry"
    graph_plot = "graph_plot"
    table_grid = "table_grid"
    number_line_scale = "number_line_scale"
    relational_schematic = "relational_schematic"
    mixed_other = "mixed_other"


class OcrStatus(StrEnum):
    ok = "ok"
    low_confidence = "low_confidence"
    unavailable = "unavailable"


# §4 rights vocabularies.
class UsStatus(StrEnum):
    public_domain = "public_domain"
    in_copyright = "in_copyright"
    not_determined = "not_determined"


class ReleaseMode(StrEnum):
    full_public = "full_public"
    server_only = "server_only"
    link_only = "link_only"
    excluded = "excluded"


class YesNoUnknown(StrEnum):
    yes = "yes"
    no = "no"
    unknown = "unknown"


class AiUseClause(StrEnum):
    none = "none"
    academic_only = "academic_only"
    licensed = "licensed"
    unknown = "unknown"


# §2.5 tracks; §3.3 partitions; §13.2 splits.
class Track(StrEnum):
    A = "A"
    B = "B"
    C = "C"   # generated known-geometry renders through the production harness (engineering fixtures
              # only since CP3 was deleted, 2026-09-26); never aggregated with, or placed on one axis
              # with, A/B (§2.5)


REPORTED_TRACKS = (Track.A, Track.B)   # the only tracks of historical-item scoring


class Split(StrEnum):
    development = "development"
    test = "test"
    reserved = "reserved"


# §9.2 / §9.3 extraction path and wrapper class.
class ExtractionPath(StrEnum):
    fenced = "fenced"
    bare_doc = "bare_doc"
    bare_pic = "bare_pic"
    none = "none"


class WrapperClass(StrEnum):
    D = "D"
    P = "P"
    B = "B"


# §7.1 fact classes (scoring categories) and attributes.
class FactClass(StrEnum):
    entity = "entity"      # E
    relation = "relation"  # R
    label = "label"        # L
    layout = "layout"      # Y


CLASS_LETTER = {FactClass.entity: "E", FactClass.relation: "R",
                FactClass.label: "L", FactClass.layout: "Y"}


class Strictness(StrEnum):
    strict = "strict"
    coarse = "coarse"


class Decidability(StrEnum):
    geo = "geo"
    model = "model"


class Anchor(StrEnum):
    A = "A"
    B = "B"


class Vote(StrEnum):
    generated = "generated"     # asc-2.0.0: the fact is in the closure of the writer's own generation
    true = "true"
    false = "false"
    abstain = "abstain"


# §10.3 judge questions.
class JudgeQuestion(StrEnum):
    Q1 = "Q1"  # recall
    Q3 = "Q3"  # alignment (Q2 source support removed 2026-09-26 with the extras channel)


# Model roles (benchmark roster; distinct from engineering agents).
class Role(StrEnum):
    builder = "builder"
    judge = "judge"
    contestant = "contestant"
    verifier = "verifier"       # G8b gatekeeper, the only AI screening step (role audit 2026-09-23)


# Parameter lifecycle for the two-act freeze (§17.3).
class ParamStatus(StrEnum):
    form_constant = "form_constant"   # fixed by the plan text itself at MS1
    provisional = "provisional"       # starting value from the plan, not yet level-frozen
    fixture = "fixture"               # engineering-only value; never valid in production
    frozen = "frozen"                 # level-frozen at the end of MS3 with provenance


class Profile(StrEnum):
    fixture = "fixture"
    production = "production"


# §17.5 gate result values. Missing evidence is NOT_RUN or BLOCKED, never PASS.
class GateResult(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"
    INFORMATIONAL = "INFORMATIONAL"     # reported, never PASS/FAIL (G9 since the 2026-09-23 D2 decision)


# §17.6 decisions.
class Decision(StrEnum):
    SCALE = "SCALE"
    SCALE_REDUCED = "SCALE_REDUCED"
    SCALE_REDUCED_WIDENED = "SCALE_REDUCED_WITH_WIDENED_ELIGIBILITY"
    STOP = "STOP"
    STOP_IMMEDIATELY = "STOP_IMMEDIATELY"
    UNDECIDABLE = "UNDECIDABLE"   # some required gate NOT_RUN/BLOCKED
