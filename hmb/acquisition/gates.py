"""Gate ladder G0a-G8b (§6.1, §6.3, §6.3.1, §6.4). Every gate returns a record
{gate_id, version, passed, measured_value, threshold} plus the exclusion reasons it implies.

G0-G7 are CPU-only. G8b, the strict gatekeeper, is the only paid gate and the only AI screening step
(role audit 2026-09-23: the G8a sorter, G8c duplicate spotter and G8d tagger are cut). It refuses to
run unless every CPU gate G2..G7d is present and passed on the row. Missing evidence never yields
`passed=True`.
There is no G1: Paddle's saved crops are the only region source (D3/D4, 2026-09-23), so no
proposer agreement exists to test. G2-G7 run on the crop in Paddle's preprocessed frame. G7a records
border measures and never excludes (D10): a crop cannot grow because the flattened page was never saved.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..hashing import sha256_bytes, sha256_file
from ..schemas import ModelRequest, ModelSpec
from ..vocab import DiagramType, Exclusion, Role
from .context import panel_split, prose_run_chars, strip_caption, text_block_measures, tighten
from .imgstats import CropAnalysis, otsu_modes, sauvola_ink, sauvola_window
from .phash import phash64

GATE_VERSION = {g: f"{g.lower()}-1" for g in ("G0a", "G0b", "G0c", "G0d", "G2", "G3", "G4",
                                              "G5", "G6", "G7a", "G7b", "G7c", "G7d")}
GATE_VERSION["G8b"] = "g8b-4"          # g8b-3 (+ descriptive diagram_type, 2026-09-25) + "four questions" wording fix (2026-09-26)
GATE_VERSION["G0b"] = "g0b-2"          # a figure-only page is not excluded for unmeasurable text height (2026-09-25)
CPU_PROPOSAL_GATES = ("G2", "G3", "G4", "G5", "G6", "G7a", "G7b", "G7c", "G7d")
PROMPT_DIR = Path(__file__).with_name("prompts")
GATEKEEPER_PROMPT = "g8b_gatekeeper.txt"
GATEKEEPER_FIELDS = ("target_type", "readable", "tikz_representable")


class LadderIncomplete(RuntimeError):
    """A paid gate was asked to run before G2-G7 passed."""


class GateBlocked(RuntimeError):
    """A gate cannot run: a required model, parameter or dependency is missing (not a failure)."""


def rec(gate_id: str, passed: bool, measured, threshold) -> dict:
    return {"gate_id": gate_id, "version": GATE_VERSION[gate_id], "passed": bool(passed),
            "measured_value": _jsonable(measured), "threshold": _jsonable(threshold)}


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


# --------------------------------------------------------------------------- page gates §6.1
def g0a(r3: dict, page_count: int | None, expected_page_count: int | None, master_sha256: str | None,
        manifest_sha256: str | None = None):
    """R3 passed, page count plausible, and the snapshotted master IS the manifest's file:
    master sha256 must equal manifest acquisition.source_file_sha256 (else rights_unresolved)."""
    ok_count = isinstance(page_count, int) and page_count >= 1 and (
        expected_page_count is None or page_count == expected_page_count)
    same_file = bool(master_sha256) and master_sha256 == manifest_sha256
    passed = bool(r3.get("passed")) and ok_count and same_file
    reasons = []
    if not r3.get("passed") or not same_file:
        reasons.append(Exclusion.rights_unresolved.value)
    elif not passed:
        reasons.append(Exclusion.page_render_fail.value)
    return rec("G0a", passed, {"r3_failures": r3.get("failures"), "page_count": page_count,
                                "checksum": master_sha256, "manifest_checksum_match": same_file},
               {"r3": "all five conditions", "expected_page_count": expected_page_count,
                "manifest_source_file_sha256": manifest_sha256}), reasons


def g0b(rendered: bool, xh: float | None, th, figure_only: bool = False):
    """Text x-height floor. A figure-only page (Paddle found item blocks and no text block: no prose,
    caption or body text) has no text to measure, so it is recorded as such and never excluded for that
    reason alone (owner decision 2026-09-25, C21); its crops meet the same gates as every other crop."""
    t = th.value("g0b_min_xheight_px")
    passed = rendered and (figure_only or (xh is not None and xh >= t))
    reasons = [] if passed else [Exclusion.page_render_fail.value if not rendered
                                 else Exclusion.page_low_resolution.value]
    return rec("G0b", passed, {"rendered": rendered, "eff_res_xheight_px": xh, "figure_only": figure_only},
               {"min_xheight_px": t, "figure_only": "no text to measure: recorded, not excluded"}), reasons


def g0c(ink_frac_page: float | None, th):
    t = th.value("g0c_min_page_ink")
    passed = ink_frac_page is not None and ink_frac_page > t
    return rec("G0c", passed, ink_frac_page, t), ([] if passed else [Exclusion.page_blank.value])


def g0d(ocr_status: str, script: str | None, skew_deg: float | None, paddle_frame: dict | None):
    """Records OCR status, script, skew and Paddle's preprocessed frame. Never excludes (§6.1)."""
    return rec("G0d", True, {"ocr_status": ocr_status, "script_detected": script, "skew_deg": skew_deg,
                             "paddle_frame": paddle_frame}, "records only"), []


# --------------------------------------------------------------------------- proposal gates §6.3
def g2(box, page_shape, physical_dpi: float | None, th):
    w, h = box[2] - box[0], box[3] - box[1]
    H, W = page_shape
    area_frac = (w * h) / float(W * H) if W * H else None
    aspect = w / h if h > 0 else None
    short_cm = min(w, h) / physical_dpi * 2.54 if physical_dpi else None
    t = {"short_side_cm_min": th.value("g2_min_short_side_cm"), "area_frac_max": th.value("g2_max_area_frac"),
         "aspect": [th.value("g2_aspect_min"), th.value("g2_aspect_max")],
         "physical_dpi": physical_dpi if physical_dpi else "dpi_unknown"}
    reasons = []
    if short_cm is None:
        reasons.append(Exclusion.physical_size_unknown.value)
    elif short_cm < t["short_side_cm_min"]:
        reasons.append(Exclusion.box_too_small.value)
    if area_frac is None or area_frac > t["area_frac_max"]:
        reasons.append(Exclusion.box_too_large.value)
    if aspect is None or not (t["aspect"][0] <= aspect <= t["aspect"][1]):
        reasons.append(Exclusion.aspect_extreme.value)
    return rec("G2", not reasons, {"short_side_cm": short_cm, "area_frac": area_frac, "aspect": aspect}, t), reasons


def g3(ink_frac: float | None, th):
    lo, hi = th.value("g3_ink_min"), th.value("g3_ink_max")
    if ink_frac is None or ink_frac < lo:
        reasons = [Exclusion.ink_too_low.value]
    elif ink_frac > hi:
        reasons = [Exclusion.ink_too_high.value]
    else:
        reasons = []
    return rec("G3", not reasons, ink_frac, [lo, hi]), reasons


def g4(an: CropAnalysis, th):
    """Solid or halftone block. Uses a global Otsu ink mask: a local (Sauvola) threshold
    treats the interior of a large dark fill as background, which is exactly what G4 must see."""
    import cv2
    n_pix = an.gray.size
    m = otsu_modes(an.gray)
    dark = (an.gray <= m[0]) if m else np.zeros(an.gray.shape, bool)
    n, _, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    largest = int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    share = largest / n_pix if n_pix else None
    ink_frac = float(dark.sum()) / n_pix if n_pix else None
    t = {"solid_frac": th.value("g4_solid_frac"), "ink_frac": th.value("g3_ink_max"),
         "min_components": th.value("g4_min_components"), "mask": "global_otsu"}
    reasons = []
    if share is None or share >= t["solid_frac"]:
        reasons.append(Exclusion.solid_block.value)
    if ink_frac is not None and ink_frac > t["ink_frac"] and n - 1 < t["min_components"]:
        reasons.append(Exclusion.halftone.value)
    return rec("G4", not reasons, {"largest_component_share": share, "ink_frac_otsu": ink_frac,
                                   "n_components": n - 1}, t), reasons


def bimodality_ratio(gray: np.ndarray, band_frac: float) -> float | None:
    m = otsu_modes(gray)
    if m is None:
        return None
    _, lo, hi = m
    band = band_frac * 255.0
    g = gray.astype(np.float64)
    near = (np.abs(g - lo) <= band) | (np.abs(g - hi) <= band)
    return float(near.mean())


def g5(gray: np.ndarray, th):
    t = th.value("g5_bimodality_min")
    r = bimodality_ratio(gray, th.value("g5_mode_band"))
    passed = r is not None and r >= t
    return rec("G5", passed, r, t), ([] if passed else [Exclusion.photo_grayscale.value])


def g6(an: CropAnalysis, th):
    d = an.diagonal
    sl = an.skel_len[1:] if an.n_components else np.zeros(0)
    longest = float(sl.max()) / d if len(sl) and d else 0.0
    n_long = int((sl >= th.value("g6_multi_frac") * d).sum())
    passed = longest >= th.value("g6_skel_frac") or n_long >= th.value("g6_multi_count")
    return rec("G6", passed, {"max_skel_frac": longest, "n_long_components": n_long},
               {"skel_frac": th.value("g6_skel_frac"), "multi_count": th.value("g6_multi_count"),
                "multi_frac": th.value("g6_multi_frac")}), ([] if passed else [Exclusion.no_long_strokes.value])


# --------------------------------------------------------------------------- G7a §6.3.1
def border_test(gray_crop: np.ndarray, th, dpi: float) -> dict:
    """border_ink per edge [left, top, right, bottom], touching_cc, offending edges."""
    import cv2
    ink = sauvola_ink(gray_crop, sauvola_window(th, dpi), th.value("sauvola_k"))
    H, W = ink.shape
    s = int(th.value("g7a_strip_px"))
    strips = [ink[:, :s], ink[:s, :], ink[:, max(0, W - s):], ink[max(0, H - s):, :]]
    border = [float(x.mean()) if x.size else 0.0 for x in strips]
    total = int(ink.sum())
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    big = np.zeros(n, bool)
    if total:
        big[1:] = st[1:, cv2.CC_STAT_AREA] >= th.value("g7a_touch_area_frac") * total
    edge_labels = [lab[:, 0], lab[0, :], lab[:, W - 1], lab[H - 1, :]] if H and W else [np.zeros(0, int)] * 4
    touch_edges = [bool(big[np.unique(e)].any()) if e.size else False for e in edge_labels]
    offending = [i for i in range(4) if border[i] > th.value("g7a_border_ink_max") or touch_edges[i]]
    return {"border_ink": [round(b, 6) for b in border], "touching_cc": any(touch_edges), "offending": offending}


def g7a(page_gray: np.ndarray, box, th, dpi: float):
    """Border measures only; never excludes (owner decision D10, 2026-09-23). A Paddle crop cannot grow
    (its flattened page was never saved), so the §6.3.1 test cannot tell a tight complete figure from
    a cut-off one. `border_ink`, `touching_cc` and the offending edges are recorded and reported."""
    t = border_test(page_gray[box[1]:box[3], box[0]:box[2]], th, dpi)
    measured = {"border_ink": t["border_ink"], "touching_cc": t["touching_cc"], "offending": t["offending"]}
    thr = {"border_ink_max": th.value("g7a_border_ink_max"), "touch_area_frac": th.value("g7a_touch_area_frac"),
           "action": "records only"}
    return rec("G7a", True, measured, thr), [], t


# --------------------------------------------------------------------------- G7b §6.4
def g7b(page_gray: np.ndarray, box, xh: float | None, th, dpi: float, ocr_units: list | None,
        depth: int = 0) -> dict:
    """Returns {record, reasons, action: pass|split|exclude, box, panels, caption}."""
    crop = page_gray[box[1]:box[3], box[0]:box[2]]
    ink = sauvola_ink(crop, sauvola_window(th, dpi), th.value("sauvola_k"))
    thr = {"gap_frac": th.value("g7b_gap_frac"), "side_ink_min": th.value("g7b_side_ink_min"),
           "max_depth": th.value("g7b_max_depth"), "max_panels": th.value("g7b_max_panels"),
           "caption_band_frac": th.value("caption_band_frac"), "caption_min_chars": th.value("caption_min_chars"),
           "prose_run_chars": th.value("prose_run_chars")}
    panels = panel_split(ink, th) if depth == 0 else [[0, 0, ink.shape[1], ink.shape[0]]]
    if len(panels) > th.value("g7b_max_panels"):
        return {"record": rec("G7b", False, {"n_panels": len(panels)}, thr),
                "reasons": [Exclusion.multi_panel_unresolved.value], "action": "exclude", "box": box,
                "panels": [], "caption": None}
    if len(panels) > 1:
        pboxes = []
        for p in panels:
            tp = tighten(ink, p)
            pboxes.append([box[0] + tp[0], box[1] + tp[1], box[0] + tp[2], box[1] + tp[3]])
        return {"record": rec("G7b", False, {"n_panels": len(panels)}, thr),
                "reasons": [Exclusion.split_into_panels.value], "action": "split", "box": box,
                "panels": pboxes, "caption": None}
    cap = strip_caption(list(box), ink, ocr_units, xh, th)
    nb = cap["crop"]
    if nb[3] - nb[1] <= 0:
        nb = list(box)
    ink2 = ink[nb[1] - box[1]:nb[3] - box[1], nb[0] - box[0]:nb[2] - box[0]]
    if cap["caption_stripped"] and ink2.any():
        tb = tighten(ink2, [0, 0, ink2.shape[1], ink2.shape[0]])
        nb = [nb[0] + tb[0], nb[1] + tb[1], nb[0] + tb[2], nb[1] + tb[3]]
        ink2 = ink[nb[1] - box[1]:nb[3] - box[1], nb[0] - box[0]:nb[2] - box[0]]
    prose = prose_run_chars(ocr_units, nb, ink2, xh, th)
    measured = {"n_panels": 1, "caption_stripped": cap["caption_stripped"], "caption_mode": cap["mode"],
                "prose_chars": prose["prose_chars"], "prose_mode": prose["mode"], "box": nb}
    if prose["prose_chars"] is not None and prose["prose_chars"] >= th.value("prose_run_chars"):
        return {"record": rec("G7b", False, measured, thr), "reasons": [Exclusion.prose_in_crop.value],
                "action": "exclude", "box": nb, "panels": [], "caption": cap}
    return {"record": rec("G7b", True, measured, thr), "reasons": [], "action": "pass", "box": nb,
            "panels": [], "caption": cap}


def g7c(page_gray, box, xh, th, dpi, ocr_units):
    crop = page_gray[box[1]:box[3], box[0]:box[2]]
    ink = sauvola_ink(crop, sauvola_window(th, dpi), th.value("sauvola_k"))
    m = text_block_measures(ocr_units, list(box), ink, xh, th)
    t = {"word_cover": th.value("g7c_word_cover"), "median_word_len": th.value("g7c_median_word_len")}
    is_text = (m["word_cover"] is not None and m["median_word_len"] is not None
               and m["word_cover"] >= t["word_cover"] and m["median_word_len"] >= t["median_word_len"])
    if m["word_cover"] is None:      # not measurable (no OCR, no geometric evidence): never exclude
        return rec("G7c", True, None, t), []                       # solely because OCR failed (§6.4)
    return rec("G7c", not is_text, m, t), ([Exclusion.text_block.value] if is_text else [])


def g7d(crop_gray: np.ndarray, th):
    ph = phash64(crop_gray)
    return rec("G7d", True, {"phash64": ph}, {"near_hamming": th.value("phash_near_hamming"),
                                              "action": "cluster only"}), [], ph


# --------------------------------------------------------------------------- paid gates G8
def require_cpu_ladder_passed(row: dict) -> None:
    got = {g["gate_id"]: g for g in row.get("gates", [])}
    missing = [g for g in CPU_PROPOSAL_GATES if g not in got]
    failed = [g for g in CPU_PROPOSAL_GATES if g in got and not got[g]["passed"]]
    if missing or failed or row.get("exclusion_reasons"):
        raise LadderIncomplete(f"{row.get('proposal_id')}: G8 refused (missing={missing}, failed={failed}, "
                               f"exclusions={row.get('exclusion_reasons')})")


def load_prompt(name: str) -> tuple[str, str]:
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return text, sha256_bytes(text.encode("utf-8"))


def parse_json_object(text: str) -> dict | None:
    """First JSON object in a model reply (fenced or bare). None when absent or malformed."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def gatekeeper_json_schema() -> dict:
    """The G8b reply as a closed JSON Schema (the prompt states the same four fields). Sent as a native
    decoding constraint only when the gatekeeper's roster entry sets structured_output (2026-09-26)."""
    props: dict[str, dict] = {k: {"type": "boolean"} for k in GATEKEEPER_FIELDS}
    props["diagram_type"] = {"type": "string", "enum": [t.value for t in DiagramType]}
    return {"name": "hmb_g8b_gatekeeper", "schema": {"type": "object", "properties": props,
                                                     "required": list(props), "additionalProperties": False}}


def _call(spec: ModelSpec, prompt_name: str, crop_png: Path, purpose: str, phase: str, max_tokens: int,
          layout, ledger, allow_live: bool, provider_override):
    from ..providers import call_model
    text, psha = load_prompt(prompt_name)
    req = ModelRequest(spec=spec, text=text, image_paths=[str(crop_png)], image_sha256=[sha256_file(crop_png)],
                       max_tokens=max_tokens, purpose=purpose, phase=phase,
                       cache_scope=f"{GATE_VERSION[purpose]}+prompt:{psha[:16]}",
                       response_schema=gatekeeper_json_schema() if spec.structured_output else None)
    resp = call_model(req, layout, ledger, allow_live=allow_live, provider_override=provider_override)
    call = {"stage": purpose, "model_id": spec.model_id, "n_calls": 0 if resp.cache_hit else 1,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "unit_price_id": spec.unit_price_id, "effort": spec.effort, "cache_key": resp.cache_key,
            "fake": resp.fake, "output_constraint": resp.output_constraint or "prompt"}
    return resp, call


def g8b(row: dict, crop_png: Path, gatekeeper: ModelSpec, *, layout, ledger, phase: str, max_tokens: int,
        allow_live: bool, provider_override=None):
    """Strict gatekeeper (§6.3 G8b as amended 2026-09-23): one call answers the target-type,
    readability and TikZ-representability questions. Reject when in doubt: the crop passes only when
    all three answers are literally `true`; a missing, non-boolean or unparseable answer rejects."""
    require_cpu_ladder_passed(row)
    if gatekeeper.role != Role.verifier.value:
        raise GateBlocked("G8b: model spec role must be 'verifier' (the gatekeeper)")
    resp, call = _call(gatekeeper, GATEKEEPER_PROMPT, crop_png, "G8b", phase, max_tokens, layout, ledger,
                       allow_live, provider_override)
    obj = parse_json_object(resp.text)
    answers = {k: (obj.get(k) if obj else None) for k in GATEKEEPER_FIELDS}
    passed = all(v is True for v in answers.values())
    pred = None if obj is None else ("accept" if passed else "reject")
    raw_type = obj.get("diagram_type") if obj else None
    label = raw_type if raw_type in DiagramType.__members__ else None     # never part of the decision
    out = {"verifier_pred": pred}
    if passed:                                   # labelled only when the diagram survives the gatekeeper
        out["diagram_type"] = label
        out["diagram_type_status"] = "predicted" if label else "invalid_reply"
    return rec("G8b", passed, {"verifier_pred": pred, **answers, "diagram_type": raw_type},
               "all of target_type, readable, tikz_representable true (reject when in doubt); "
               "diagram_type is descriptive only"),         ([] if passed else [Exclusion.verifier_reject.value]), out, [call]
