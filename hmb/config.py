"""Named-parameter registry and the two-act freeze (plan §17.3, §15.5).

Every tunable number in the plan is registered here with the section that names it.
Three kinds of entry:
  form_constant - the plan fixes the number itself; valid in every profile.
  provisional   - the plan gives a starting value that must be confirmed at the level freeze (end of MS3).
  calibrated    - the plan names the parameter but its value comes only from pilot / development
                  data / an owner decision; there is no plan value.

A thresholds file has a profile:
  fixture    - engineering tests only; every parameter has a value (fixture values are marked),
               and nothing computed under it may be reported or used to pass a gate.
  production - only form constants and level-frozen values are readable; reading any other
               parameter raises CalibrationMissing. Production scoring therefore refuses to run
               until the level freeze exists.

Pre-freeze execution context (MS1 form freeze -> MS2 pilot -> MS3 development set -> level freeze;
2026-09-25, milestones renamed 2026-09-26). The level-freeze inputs are measured on development books
before the freeze, so a production thresholds file can be opened in one of the named contexts of
PREFREEZE_CONTEXTS (`--thresholds prefreeze:PILOT` or `prefreeze:DEV`). It is the production profile and the real
pipeline, not the fixture profile. Inside it a `provisional` parameter is readable only if it is on
that context's list and has a value (a plan starting value, or a pre-freeze value set with
`thresholds-provisional` for a calibrated parameter). Every read is recorded, and the command logs
the context, stage, partition, parameter values, registry and form-freeze identity and its outputs
(`manifests/prefreeze_runs.jsonl`). The context requires the MS1 owner approval, and it is refused once
levels are frozen or a test-book artifact exists. Test-book acquisition, final cards, items-finalize,
releases and reported scores stay behind the level freeze.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_json
from .store import FreezeViolation, Layout, read_json, write_mutable
from .vocab import ParamStatus, Profile

FC, PV, CAL = "form_constant", "provisional", "calibrated"
GIB = 1024 ** 3
MIB = 1024 ** 2


@dataclass(frozen=True)
class ParamSpec:
    name: str
    section: str
    kind: str                 # FC | PV | CAL
    plan_value: Any           # None for CAL
    fixture_value: Any        # engineering value; equals plan_value unless CAL
    desc: str


def _p(name, section, kind, plan_value, desc, fixture=None):
    return ParamSpec(name, section, kind, plan_value,
                     plan_value if fixture is None and kind != CAL else fixture, desc)


REGISTRY: list[ParamSpec] = [
    # ---- §6 acquisition gate ladder (confirmed at the level freeze from the pilot and development set) ----
    _p("render_dpi", "§5.2/§6.1", PV, 400, "page render resolution"),
    # (no g8c_embed_cos_min: G8c is cut by the 2026-09-23 role audit)
    _p("g8_max_tokens", "§6.3 G8b (decoding for the gatekeeper)", CAL, None, "G8 call max_tokens",
       fixture=512),
    _p("g0b_min_xheight_px", "§6.1", PV, 12, "G0b eff_res_xheight_px floor"),
    _p("g0c_min_page_ink", "§6.1", PV, 0.002, "G0c ink_frac_page floor"),
    # (no §6.2 match/fusion levels: Paddle's saved crops are the only region source, D3/D4 2026-09-23)
    _p("text_height_tol", "§6.4", PV, 0.25, "modal text-height tolerance"),
    _p("g2_min_short_side_cm", "§6.3", PV, 0.7, "G2 physical short side floor"),
    _p("g2_max_area_frac", "§6.3", PV, 0.60, "G2 max crop/page area"),
    _p("g2_aspect_min", "§6.3", PV, 1 / 6, "G2 aspect lower bound"),
    _p("g2_aspect_max", "§6.3", PV, 6.0, "G2 aspect upper bound"),
    _p("g3_ink_min", "§6.3", PV, 0.005, "G3 ink lower bound"),
    _p("g3_ink_max", "§6.3", PV, 0.45, "G3 ink upper bound"),
    _p("g4_solid_frac", "§6.3", PV, 0.90, "G4 single component area share"),
    _p("g4_min_components", "§6.3", PV, 5, "G4 component floor when ink>0.45"),
    _p("g5_bimodality_min", "§6.3", PV, 0.75, "G5 bimodality ratio floor"),
    _p("g5_mode_band", "§6.3", PV, 0.10, "G5 +/- band around Otsu modes"),
    _p("g6_skel_frac", "§6.3", PV, 0.35, "G6 longest skeleton / diagonal"),
    _p("g6_multi_count", "§6.3", PV, 3, "G6 alternative: number of long components"),
    _p("g6_multi_frac", "§6.3", PV, 0.15, "G6 alternative: skeleton length / diagonal"),
    _p("sauvola_window_px", "§6.3.1", PV, 25, "Sauvola window at 300-dpi-equivalent"),
    _p("sauvola_k", "§6.3.1", PV, 0.2, "Sauvola k"),
    _p("g7a_strip_px", "§6.3.1", PV, 2, "border strip width"),
    _p("g7a_border_ink_max", "§6.3.1", PV, 0.02, "max border ink"),
    _p("g7a_touch_area_frac", "§6.3.1", PV, 0.01, "touching component area / total ink"),
    _p("g7b_gap_frac", "§6.4", PV, 0.04, "panel gap / max side"),
    _p("g7b_side_ink_min", "§6.4", PV, 0.20, "min ink share per panel side"),
    _p("g7b_max_depth", "§6.4", PV, 2, "panel split recursion depth"),
    _p("g7b_max_panels", "§6.4", PV, 4, "max panels before multi_panel_unresolved"),
    _p("caption_band_frac", "§6.4", PV, 0.25, "caption band: top/bottom fraction of crop"),
    _p("caption_min_chars", "§6.4", PV, 8, "caption min characters"),
    _p("prose_run_chars", "§6.4", PV, 12, "residual prose run length"),
    _p("g7c_word_cover", "§6.3", PV, 0.35, "G7c OCR word box ink cover"),
    _p("g7c_median_word_len", "§6.3", PV, 4, "G7c median word length"),
    _p("phash_near_hamming", "§6.3", PV, 6, "G7d near-duplicate Hamming distance"),
    _p("ctx_window_heights", "§6.4", FC, 1.5, "ctxwin-1: +/- crop heights"),
    _p("ctx_cap_chars", "§6.4", FC, 1200, "ctxwin-1: character cap"),
    _p("frag_area_frac", "§6.5", FC, 0.005, "frag_index small-component threshold"),
    _p("dedup_jaccard", "§6.7", FC, 0.90, "redundancy Jaccard on eligible fact sets"),
    _p("pool_safety_factor", "§6.8", FC, 1.25, "N_cand = N*/r * 1.25"),
    # ---- §8 cards ----
    _p("align_w_L", "§8.1 (form constant 2026-09-25)", FC, 1.0, "alignment label weight"),
    _p("align_w_G", "§8.1 (form constant 2026-09-25)", FC, 1.0, "alignment geometry weight"),
    _p("align_w_R", "§8.1 (form constant 2026-09-25)", FC, 0.5, "alignment relational weight"),
    _p("align_theta_match", "§8.1 (form constant 2026-09-25)", FC, 0.9, "THETA_MATCH"),
    _p("align_delta_margin", "§8.1 (form constant 2026-09-25)", FC, 0.15, "DELTA_MARGIN"),
    _p("align_sigma", "§8.1 (form constant 2026-09-25)", FC, 0.05, "sim_geom length scale (crop-relative)"),
    _p("card_builder_max_tokens", "§7.3/§8.8 (builder decoding settings)", CAL, None,
       "builder generation/verification max_tokens", fixture=4096),
    _p("card_max_entities", "§7.4 (asc-2.0.0 hard cap, 2026-09-26 streamlining)", FC, 40,
       "entity lines kept per writer generation; later lines are discarded"),
    _p("card_max_facts", "§7.4 (asc-2.0.0 hard cap, 2026-09-26 streamlining)", FC, 60,
       "fact lines kept per writer generation; later lines are discarded"),
    _p("card_verify_chunk_size", "§8.3 (asc-2.0.0, 2026-09-26 streamlining)", FC, 20,
       "statements per verification call (at Sonnet 5 medium effort a 39-statement call spent all 8096 "
       "tokens thinking, measured 2026-09-26; 20 keeps both efforts inside the cap)"),
    _p("wl_rounds", "§8.1", FC, 2, "1-WL colour refinement rounds"),
    _p("item_min_entity_facts", "§8.7", PV, 3, "eligible entity facts floor"),
    _p("item_min_strict_relation_facts", "§8.7", PV, 3, "eligible strict relation facts floor"),
    _p("item_max_contradiction", "§8.7", PV, 0.10, "contradiction_rate ceiling"),
    _p("item_max_ambiguous_frac", "§8.7", PV, 0.20, "ambiguous / canonical entity ceiling"),
    _p("book_max_share", "§8.7", FC, 0.25, "max share of eligible set from one book"),
    # ---- §9 submission / sandbox ----
    _p("eps_ink", "§9.5", PV, 0.001, "V_EMPTY ink fraction of canonical canvas"),
    _p("cpu_seconds", "§9.6", PV, 20, "C_TIMEOUT CPU limit"),
    _p("wall_seconds", "§9.6", PV, 60, "C_WALLCLOCK liveness kill"),
    _p("memory_bytes", "§9.6", PV, 1 * GIB, "C_RESOURCE memory"),
    _p("output_bytes", "§9.6", PV, 32 * MIB, "C_RESOURCE output file"),
    _p("log_bytes", "§9.6", PV, 4 * MIB, "C_RESOURCE log"),
    _p("page_cap", "§9.6", PV, 4, "pages above -> C_RESOURCE"),
    _p("page_dim_cap_cm", "§9.6", PV, 200, "V_OVERSIZE page dimension"),
    _p("render_pixel_cap", "§9.5 V_OVERSIZE (pixel cap named, no value)", PV, 100_000_000,
       "W*H pixel cap at 300 dpi before rasterization"),
    _p("sandbox_pids_limit", "§9.6 (process limit named, no value)", PV, 64, "container pids limit"),
    _p("raster_cpu_seconds", "§9.7 (in-container pdftoppm)", PV, 20, "rasterizer CPU limit -> C_RESOURCE"),
    _p("render_dpi_candidate", "§9.7", FC, 300, "pdftoppm DPI"),
    _p("canvas_px", "§9.7", FC, 1024, "canonical canvas"),
    _p("max_upscale", "§9.7", FC, 4.0, "canonicalization upscale cap"),
    _p("transport_retry_max", "§9.1", FC, 3, "transport retries per call"),
    _p("contestant_max_tokens", "§9.1", CAL, None, "identical max_tokens across models", fixture=4096),
    _p("determinism_sample_frac", "§9.6", FC, 0.05, "HarnessDeterminism@rerun sample"),
    # ---- §10 adjudication ----
    _p("eps_on_frac", "§10.2", PV, 0.010, "incidence tolerance / render diagonal"),
    _p("tangency_rel", "§10.2", PV, 0.015, "tangency |dist-r| / r"),
    _p("parallel_deg", "§10.2", PV, 1.5, "parallel tolerance (deg)"),
    _p("perpendicular_deg", "§10.2", PV, 1.5, "perpendicular tolerance (deg)"),
    # C43 (owner decision 2026-09-27): the MGS repeats the closure once per fact, so its cost grows ~n^5 in a class of
    # mutually parallel segments (a pgfplots mesh gave 1,185 primitives and never finished). Above the bound the CGX
    # carries no primitives or facts and every card fact goes to Q1. MS2 pilot: real answers <= 40 primitives
    # (= card_max_entities); worst case at 60 all-parallel segments: 456 s.
    _p("cgx_max_primitives", "§10.2 (C43, owner decision 2026-09-27)", FC, 60,
       "CGX primitives above which the extractor records no evidence and every card fact goes to Q1"),
    _p("circle_fit_resid_rel", "§10.2.1 (tolerance named, no value)", PV, 0.01,
       "max |d_i - r| / r over sampled Bezier points to accept a circle"),
    _p("axis_min_length_frac", "C28 form choice (CONTRACTS.md stroke typing)", FC, 0.5,
       "an unlabelled tipped horizontal/vertical stroke is an axis only if >= this x D"),
    _p("label_bind_frac", "§10.3", PV, 0.06, "label anchor radius / D"),
    _p("subgraph_max_unmatched", "§10.3", FC, 15, "exact subgraph matching limit"),
    _p("judge_max_tokens", "§10.6 (pinned judge decoding)", CAL, None, "judge max_tokens", fixture=2048),
    _p("judge_chunk_size", "§10.8", FC, 25, "facts per judge chunk"),
    _p("drift_probe_n_triples", "§10.9 (\"a frozen set of 100 triples\")", FC, 100,
       "judged development-set facts sampled into the drift-probe triple set"),
    _p("kappa_min", "§10.9 (form constant 2026-09-25)", FC, 0.8, "drift probe agreement with the recorded development verdicts"),
    _p("alpha_min", "§10.9 (form constant 2026-09-25)", FC, 0.8, "drift probe open-vs-close agreement"),
    # ---- §11 grading ----
    _p("m_E", "§11.2", FC, 3, "diagnostic min scorable facts E"),
    _p("m_R", "§11.2", FC, 2, "diagnostic min scorable facts R"),
    _p("m_L", "§11.2", FC, 1, "diagnostic min scorable facts L"),
    _p("q5_tau_E", "§11.4", FC, 0.8, "Q5 entity recall conditioning"),
    _p("slice_min_items", "§11.11", FC, 30, ""),
    _p("slice_min_groups", "§11.11", FC, 3, ""),
    _p("slice_min_facts", "§11.11", FC, 150, ""),
    _p("ceiling_flag_H", "§14.6", FC, 0.90, ""),
    _p("ceiling_flag_sd", "§14.6", FC, 0.10, ""),
    # ---- §13-14 sampling and statistics ----
    _p("alpha", "§14.2", FC, 0.05, ""),
    _p("bh_q", "§14.4", FC, 0.10, ""),
    _p("G_min", "§13.7", FC, 6, "hard floor on test groups"),
    _p("power_target", "§13.7", FC, 0.80, ""),
    _p("sizing_reps", "§13.7", FC, 2000, ""),
    _p("sizing_grid", "§13.7", FC, [150, 250, 350, 450], ""),
    _p("delta_star", "§14.4/§17.5", CAL, None, "preregistered minimum effect", fixture=0.05),
    _p("common_output_cap_tokens", "§9.1 (owner decision 2026-09-25)", FC, 8192,
       "upper bound on the common contestant output cap (contestant_max_tokens)"),
    _p("N_star", "§13.7", CAL, None, "pilot size", fixture=None),
    _p("G_star", "§13.7", CAL, None, "test groups", fixture=None),
    _p("soft_edge_series_years", "§13.1", FC, 10, "same publisher+series within N years = soft edge"),
    _p("collapse_frac", "§13.1", FC, 0.50, "collapse guard"),
    _p("dev_groups", "§13.2", FC, 3, "development split groups"),
    # (no §12 control-suite or evaluator-validation parameters: CP3/CP5 deleted, checkpoint simplification 2026-09-26)
    # ---- §16-17 cost and gates ----
    # (no cost caps: spending is tracked, not capped; owner decision 2026-09-23, audit D2)
    _p("F_bar_min", "§17.5 G2", PV, 8, "mean eligible facts per item"),
    _p("min_items_per_group", "§17.5 G2", FC, 10, ""),
    _p("tau_b_min", "§17.5 G6", FC, 0.80, ""),
]

SPECS: dict[str, ParamSpec] = {s.name: s for s in REGISTRY}
assert len(SPECS) == len(REGISTRY), "duplicate parameter name"


class CalibrationMissing(RuntimeError):
    """A production path read a parameter that has no level-frozen value."""


class ThresholdsError(ValueError):
    pass


_ACQ_LEVELS = ("render_dpi", "g0b_min_xheight_px", "g0c_min_page_ink", "text_height_tol", "g2_min_short_side_cm",
               "g2_max_area_frac", "g2_aspect_min", "g2_aspect_max", "g3_ink_min", "g3_ink_max", "g4_solid_frac",
               "g4_min_components", "g5_bimodality_min", "g5_mode_band", "g6_skel_frac", "g6_multi_count",
               "g6_multi_frac", "sauvola_window_px", "sauvola_k", "g7a_strip_px", "g7a_border_ink_max",
               "g7a_touch_area_frac", "g7b_gap_frac", "g7b_side_ink_min", "g7b_max_depth", "g7b_max_panels",
               "caption_band_frac", "caption_min_chars", "prose_run_chars", "g7c_word_cover", "g7c_median_word_len",
               "phash_near_hamming")
_SANDBOX_LEVELS = ("eps_ink", "cpu_seconds", "wall_seconds", "memory_bytes", "output_bytes", "log_bytes", "page_cap",
                   "page_dim_cap_cm", "render_pixel_cap", "sandbox_pids_limit", "raster_cpu_seconds")
_DEV_READABLE = _ACQ_LEVELS + ("g8_max_tokens", "card_builder_max_tokens", "item_min_entity_facts",
                               "item_min_strict_relation_facts", "item_max_contradiction", "item_max_ambiguous_frac")
# A contestant cell on development items (the MS2 pilot's c_gen metering and the MS3 variance probe, C42 fix
# 2026-09-26): the compile harness, extractor tolerances and the contestant and judge decoding caps.
_RUN_LEVELS = _SANDBOX_LEVELS + ("eps_on_frac", "tangency_rel", "parallel_deg", "perpendicular_deg",
                                 "circle_fit_resid_rel", "label_bind_frac", "contestant_max_tokens",
                                 "judge_max_tokens")
# Pre-freeze contexts, one per milestone before the level freeze, and the parameters each may read
# (form constants are always readable). Both work on development books only (checkpoint simplification
# 2026-09-26: MS2 pilot, MS3 development set; the CAL control context was deleted with CP3). DEV also reads
# the owner's pre-freeze δ*, which the sizing simulation needs.
PREFREEZE_CONTEXTS: dict[str, dict] = {
    # PILOT runs without an MS1 approval of the current forms (owner instruction 2026-09-27: the pilot iterates on
    # the forms); every later context, the level freeze and production still require it.
    "PILOT": {"partition": "development", "stage": "MS2 pilot", "readable": _DEV_READABLE + _RUN_LEVELS,
              "requires_ms1": False},
    "DEV": {"partition": "development", "stage": "MS3 development set",
            "readable": _DEV_READABLE + _RUN_LEVELS + ("delta_star",)},
}
PREFREEZE_PREFIX = "prefreeze:"


@dataclass
class Thresholds:
    profile: Profile
    params: dict[str, dict]
    frozen_at: str | None
    source: str = "<memory>"
    context: str | None = None                  # a PREFREEZE_CONTEXTS name, or None
    reads: dict[str, Any] | None = None         # provisional values read in the context

    @property
    def is_production(self) -> bool:
        return self.profile == Profile.production

    @property
    def is_prefreeze(self) -> bool:
        return self.context is not None

    def value(self, name: str) -> Any:
        if name not in SPECS:
            raise KeyError(f"unregistered parameter {name!r}")
        entry = self.params.get(name)
        if entry is None:
            raise CalibrationMissing(f"{name}: absent from {self.source}")
        status = ParamStatus(entry["status"])
        if self.is_production and status not in (ParamStatus.form_constant, ParamStatus.frozen):
            ctx = PREFREEZE_CONTEXTS.get(self.context or "")
            if ctx is None:
                raise CalibrationMissing(
                    f"{name} ({SPECS[name].section}) is {status.value}; production requires a level freeze")
            if status != ParamStatus.provisional or name not in ctx["readable"]:
                raise CalibrationMissing(f"{name} ({SPECS[name].section}) is not readable in the pre-freeze "
                                         f"{self.context} context")
            if entry["value"] is None:
                raise CalibrationMissing(f"{name} ({SPECS[name].section}) has no pre-freeze value "
                                         "(set one with `thresholds-provisional`)")
            if self.reads is not None:
                self.reads[name] = entry["value"]
        v = entry["value"]
        if v is None:
            raise CalibrationMissing(f"{name} ({SPECS[name].section}) has no value")
        return math.inf if v == "inf" else v

    def levels_block(self) -> dict | None:
        """Compact provenance for an artifact written in a pre-freeze context (None otherwise)."""
        if not self.is_prefreeze:
            return None
        return {"levels": "provisional", "context": self.context,
                "partition": PREFREEZE_CONTEXTS[self.context or ""]["partition"],
                "thresholds_hash": self.hash, "registry_hash": registry_hash()}

    def status(self, name: str) -> ParamStatus:
        return ParamStatus(self.params[name]["status"])

    def missing_for_production(self) -> list[str]:
        out = []
        for name in SPECS:
            e = self.params.get(name)
            if e is None or e["value"] is None or ParamStatus(e["status"]) not in (
                    ParamStatus.form_constant, ParamStatus.frozen):
                out.append(name)
        return out

    def to_json(self) -> dict:                   # the file content; a context is never written
        return {"profile": self.profile.value, "frozen_at": self.frozen_at,
                "registry_hash": registry_hash(), "params": self.params}

    @property
    def hash(self) -> str:
        return sha256_json(self.to_json())


def registry_hash() -> str:
    """Hash of the parameter *forms* (MS1 form freeze artifact)."""
    return sha256_json([[s.name, s.section, s.kind,
                         s.plan_value if s.plan_value != math.inf else "inf"] for s in REGISTRY])


def form_constant(name: str) -> Any:
    """The value of a form constant, which is the same in every profile (for code that renders frozen
    text without a thresholds file, e.g. the builder prompt caps)."""
    spec = next((s for s in REGISTRY if s.name == name), None)
    if spec is None or spec.kind != FC:
        raise KeyError(f"{name} is not a registered form constant")
    return spec.plan_value


def fixture_thresholds() -> Thresholds:
    params = {}
    for s in REGISTRY:
        status = ParamStatus.form_constant if s.kind == FC else ParamStatus.fixture
        params[s.name] = {"value": s.fixture_value, "status": status.value, "provenance": None}
    return Thresholds(Profile.fixture, params, None, "fixture_thresholds()")


def production_template() -> Thresholds:
    """MS1 form-freeze thresholds: form constants set, starting values provisional, calibrated empty."""
    params = {}
    for s in REGISTRY:
        if s.kind == FC:
            params[s.name] = {"value": s.plan_value, "status": ParamStatus.form_constant.value,
                              "provenance": None}
        elif s.kind == PV:
            params[s.name] = {"value": s.plan_value, "status": ParamStatus.provisional.value,
                              "provenance": None}
        else:
            params[s.name] = {"value": None, "status": ParamStatus.provisional.value,
                              "provenance": None}
    return Thresholds(Profile.production, params, None, "production_template()")


def load_thresholds(path: str | Path) -> Thresholds:
    d = read_json(path)
    try:
        prof = Profile(d["profile"])
        params = d["params"]
    except (KeyError, ValueError) as e:
        raise ThresholdsError(f"{path}: malformed thresholds file") from e
    for name, e in params.items():
        if name not in SPECS:
            raise ThresholdsError(f"{path}: unregistered parameter {name}")
        st = ParamStatus(e["status"])
        if prof == Profile.production and st == ParamStatus.fixture:
            raise ThresholdsError(f"{path}: fixture value for {name} in a production file")
        if st == ParamStatus.frozen:
            prov = e.get("provenance") or {}
            for k in ("calibration_artifact_id", "sha256", "date", "rule"):
                if not prov.get(k):
                    raise ThresholdsError(f"{path}: frozen {name} lacks provenance.{k}")
    return Thresholds(prof, params, d.get("frozen_at"), str(path))


# ---------------------------------------------------------------- pre-freeze context (2026-09-25)
def prefreeze(th: Thresholds, context: str, layout: Layout) -> Thresholds:
    """Open a production thresholds file in a named pre-freeze context (see the module docstring)."""
    from .manifests import cp1_approval_problems
    if context not in PREFREEZE_CONTEXTS:
        raise ThresholdsError(f"unknown pre-freeze context {context!r} (one of {sorted(PREFREEZE_CONTEXTS)})")
    if not th.is_production:
        raise ThresholdsError("a pre-freeze context opens the production thresholds file, never a fixture")
    if th.frozen_at:
        raise ThresholdsError("levels are frozen: pre-freeze contexts are closed; use the frozen thresholds")
    if layout.test_artifacts_exist():
        raise FreezeViolation("a test-book artifact exists: pre-freeze contexts are closed")
    cp1 = cp1_approval_problems(layout)
    if cp1 and PREFREEZE_CONTEXTS[context].get("requires_ms1", True):
        raise ThresholdsError("pre-freeze work follows the MS1 form freeze: " + "; ".join(cp1))
    return Thresholds(th.profile, th.params, None, th.source, context=context, reads={})


def resolve_thresholds_arg(arg: str | None, layout: Layout) -> Thresholds:
    """One `--thresholds` resolver: 'fixture' (engineering), 'prefreeze:<context>' (the root's production
    file in a pre-freeze context), a path, or None (the root's thresholds.json)."""
    if arg == "fixture":
        return fixture_thresholds()
    if arg and arg.startswith(PREFREEZE_PREFIX):
        return prefreeze(load_thresholds(layout.thresholds), arg[len(PREFREEZE_PREFIX):], layout)
    return load_thresholds(arg or layout.thresholds)


def execution_context(th: Thresholds, command: str, inputs: dict | None = None) -> dict | None:
    """The provenance a pre-freeze artifact embeds (None outside a pre-freeze context): levels
    provisional, context, stage, partition, the provisional values read so far, the thresholds, registry
    and form-freeze identity, and the upstream inputs."""
    if not th.is_prefreeze:
        return None
    from .manifests import _form_freeze_sha, form_freeze_components
    ctx = PREFREEZE_CONTEXTS[th.context or ""]
    return {"command": command, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "levels": "provisional", "context": th.context, "stage": ctx["stage"], "partition": ctx["partition"],
            "params_read": dict(sorted((th.reads or {}).items())), "thresholds_hash": th.hash,
            "registry_hash": registry_hash(), "form_freeze_sha256": _form_freeze_sha(form_freeze_components()),
            "inputs": inputs or {}}


def prefreeze_record(th: Thresholds, layout: Layout, command: str, outputs: list,
                     inputs: dict | None = None) -> dict | None:
    """Append one run record for a command run in a pre-freeze context (None outside one): the
    execution context plus every output with its sha256 (`manifests/prefreeze_runs.jsonl`)."""
    rec = execution_context(th, command, inputs)
    if rec is None:
        return None
    from .hashing import sha256_file
    from .store import append_jsonl
    outs = {}
    for o in outputs:
        op = Path(o)
        if op.exists():
            try:
                key = op.resolve().relative_to(layout.root.resolve()).as_posix()
            except ValueError:
                key = str(op)
            outs[key] = sha256_file(op)
    rec["outputs"] = outs
    append_jsonl(layout.p("manifests", "prefreeze_runs.jsonl"), rec)
    return rec


def set_provisional(layout: Layout, values: dict[str, dict], now: str | None = None) -> Thresholds:
    """Record pre-freeze values for calibrated parameters in the production thresholds file:
    `values[name] = {value, source, rule}`. They stay `provisional` (readable only inside a pre-freeze
    context that lists them) and are never frozen by default: the level freeze needs a provenance record
    for every calibrated parameter. Provisional parameters keep their plan starting values."""
    import json
    th = load_thresholds(layout.thresholds)
    if not th.is_production or th.frozen_at:
        raise ThresholdsError("pre-freeze values go into an unfrozen production thresholds file only")
    if layout.test_artifacts_exist():
        raise FreezeViolation("a test-book artifact exists")
    date = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, rec in values.items():
        spec = SPECS.get(name)
        if spec is None or spec.kind != CAL:
            raise ThresholdsError(f"{name}: only calibrated parameters take a pre-freeze value")
        if "artifact" in rec:                     # e.g. a value from the dev-measure artifact
            r = resolve_artifact_provenance(layout, {name: rec})[name]
            rec = {"value": r["value"], "source": f"{r['calibration_artifact_id']} sha256={r['sha256']}",
                   "rule": r["rule"]}
        if rec.get("value") is None or not rec.get("source") or not rec.get("rule"):
            raise ThresholdsError(f"{name}: a pre-freeze value needs value, source and rule")
        th.params[name] = {"value": rec["value"], "status": ParamStatus.provisional.value,
                           "provenance": {"prefreeze": True, "source": rec["source"], "rule": rec["rule"],
                                          "date": date}}
    write_mutable(layout.thresholds, (json.dumps(th.to_json(), indent=1, sort_keys=True) + "\n").encode())
    return th


def _field(doc: Any, dotted: str) -> Any:
    for k in dotted.split("."):
        if not isinstance(doc, dict) or k not in doc:
            raise ThresholdsError(f"field {dotted!r} not found")
        doc = doc[k]
    return doc


def resolve_artifact_provenance(layout: Layout, provenance: dict[str, dict]) -> dict[str, dict]:
    """The level freeze consumes the pre-freeze measurements: a record `{artifact: <root-relative path>, field:
    <dotted key>}` takes its value from that artifact, which must have been written in a pre-freeze
    context on the development partition; artifact id, sha256, date and rule are filled from it."""
    from .hashing import sha256_file
    out = {}
    for name, rec in provenance.items():
        if "artifact" not in rec:
            out[name] = rec
            continue
        path = layout.p(*Path(rec["artifact"]).parts)
        if not path.exists():
            raise ThresholdsError(f"{name}: artifact {rec['artifact']} does not exist under the root")
        doc = read_json(path)
        ctx = doc.get("execution_context") or {}
        if ctx.get("levels") != "provisional" or ctx.get("partition") != "development":
            raise ThresholdsError(f"{name}: {rec['artifact']} is not a pre-freeze development artifact")
        out[name] = {"value": _field(doc, rec["field"]), "calibration_artifact_id": rec["artifact"],
                     "sha256": sha256_file(path), "date": doc.get("created_utc") or ctx.get("utc"),
                     "rule": f"{doc.get('artifact', 'artifact')}:{rec['field']} ({ctx.get('context')} "
                             f"{ctx.get('stage')})"}
    return out


REQUIRED_PROVENANCE = ("calibration_artifact_id", "sha256", "date", "rule")
PLAN_VALUE_CONFIRMED = "plan starting value confirmed at the level freeze"


def level_freeze(template: Thresholds, provenance: dict[str, dict], layout: Layout,
                 out_path: str | Path | None = None, now: str | None = None) -> Thresholds:
    """Level freeze (end of MS3). `provenance[name] = {value, calibration_artifact_id, sha256, date, rule}`.

    Every calibrated parameter must receive a provenance record. A provisional parameter without one
    is confirmed at its plan starting value, with provenance naming the registry hash of the frozen
    forms (rule PLAN_VALUE_CONFIRMED); a record overrides it with a measured value. Refuses to change
    any level once the first test-book artifact exists (§17.3 invariant).
    """
    if template.profile != Profile.production:
        raise ThresholdsError("level freeze requires a production template")
    out_path = Path(out_path or layout.thresholds)
    params = {k: dict(v) for k, v in template.params.items()}
    missing = []
    date = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    for s in REGISTRY:
        if s.kind == FC:
            continue
        rec = provenance.get(s.name)
        if rec is None and s.kind == PV:
            rh = registry_hash()
            rec = {"value": s.plan_value, "calibration_artifact_id": f"registry:{rh}", "sha256": rh,
                   "date": date, "rule": PLAN_VALUE_CONFIRMED}
        if rec is None or any(not rec.get(k) for k in REQUIRED_PROVENANCE) or rec.get("value") is None:
            missing.append(s.name)
            continue
        params[s.name] = {"value": rec["value"], "status": ParamStatus.frozen.value,
                          "provenance": {"parameter": s.name, **rec}}
    if missing:
        raise ThresholdsError(f"level freeze incomplete; missing provenance for {missing}")
    frozen = Thresholds(Profile.production, params,
                        now or datetime.now(timezone.utc).isoformat(timespec="seconds"), str(out_path))
    if layout.test_artifacts_exist():
        prev = load_thresholds(out_path) if out_path.exists() else None
        if prev is None or not prev.frozen_at:
            raise FreezeViolation("test-book artifacts exist but no prior level freeze; "
                                  "the freeze must predate every test artifact (§15.5)")
        if {k: v["value"] for k, v in prev.params.items()} != {k: v["value"] for k, v in params.items()}:
            raise FreezeViolation("levels are frozen and test-book artifacts exist; bump MAJOR version")
        return prev
    import json
    write_mutable(out_path, (json.dumps(frozen.to_json(), indent=1, sort_keys=True) + "\n").encode())
    return frozen
