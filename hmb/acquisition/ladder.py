"""CPU gate ladder driver: book -> R3 -> pages (G0) -> Paddle crops -> G2..G7d -> funnel.

Items are Paddle's saved `image` and `chart` crops (D13), used as they are (owner decision D3/D4,
2026-09-23): each crop is one proposal, its box is Paddle's block_bbox in Paddle's preprocessed frame (`box_frame`), and G2-G7d
run on the crop pixels. There is no second region source, so no fusion and no G1. The master page is
still rendered for the page gates G0b-G0d and stored as the item's page asset; the page x-height and
physical DPI are carried into Paddle's frame by the ratio of the long sides, which ignores the dewarp's
local scale change.

One call processes one snapshotted source. Rows are appended to the funnel only after pHash clustering
of the batch (G7d clusters, never drops). Paid gates (G8) are separate (`gates.g8a/g8b/g8d`) and refuse
rows whose CPU ladder did not pass.
"""
from __future__ import annotations

from ..rights import r3_gate
from ..store import Layout
from ..vocab import Exclusion, Profile
from . import collection as col
from . import gates as G
from .funnel import PIPELINE_VERSION, append_page_row, append_row, config_hash, finalize, proposal_id
from .imgstats import STATS_VERSION, analyze, compute_from_analysis, numeric_stats, sauvola_ink, sauvola_window
from .phash import exact_and_near
from .render import RenderError, load_page_gray, page_measures


def process_source(layout: Layout, snap: dict, manifest: dict | None, th, *, profile: Profile | str,
                   page_indices: list[int] | None = None, store_pages: bool = True) -> dict:
    """Run R3 + G0 + G2..G7d on the Paddle crops of the chosen pages of one source.

    Returns a summary {book_id, r3, pages: [...], rows: [...]}.
    `manifest=None` means no rights record exists: R3 fails and nothing is rendered.
    """
    profile = Profile(profile)
    book = snap["book_id"]
    cfg = config_hash(th, {"profile": profile.value, "stats_version": STATS_VERSION})
    r3 = r3_gate(manifest, profile) if manifest is not None else {
        "passed": False, "failures": ["no_manifest_record"], "profile": profile.value}
    summary = {"book_id": book, "source_name": snap["source_name"], "profile": profile.value, "r3": r3,
               "pages": [], "rows": []}
    if not r3["passed"]:
        rec, reasons = G.g0a(r3, None, None, snap["master"]["sha256"],
                             ((manifest or {}).get("acquisition") or {}).get("source_file_sha256"))
        append_page_row(layout, {"book_id": book, "page_index": None, "gates": [rec], "exclusion_reasons": reasons,
                                 "pipeline_version": PIPELINE_VERSION, "config_hash": cfg, "profile": profile.value})
        return summary
    pre = col.is_precropped(snap)
    master = None if pre else col.verified_master_path(snap)
    try:
        ocr = col.parse_ocr(snap)
        ocr_problem = None
    except (col.OcrMalformed, ValueError) as e:
        ocr, ocr_problem = None, str(e)
    from .render import page_count as _pc
    n_pages = len(ocr["pages"]) if pre and ocr else (0 if pre else _pc(master))
    acq = (manifest or {}).get("acquisition") or {}
    expected = acq.get("page_count")
    indices = list(range(n_pages)) if page_indices is None else page_indices
    rows: list[dict] = []
    for pi in indices:
        page = _process_page(layout, snap, master, book, pi, n_pages, expected, r3, ocr, ocr_problem, th, cfg,
                             profile, acq.get("source_dpi_nominal"), store_pages, acq.get("source_file_sha256"))
        summary["pages"].append(page["page_row"])
        rows.extend(page["rows"])
    updates = _cluster(layout, rows, th)
    for r in rows:
        summary["rows"].append(append_row(layout, r))
    summary["cluster_updates"] = [append_row(layout, r) for r in updates]
    return summary


def _process_page(layout, snap, master, book, pi, n_pages, expected, r3, ocr, ocr_problem, th, cfg, profile,
                  source_dpi, store_pages, manifest_file_sha) -> dict:
    gates, reasons = [], []
    rec, rs = G.g0a(r3, n_pages, expected, snap["master"]["sha256"], manifest_file_sha)
    gates.append(rec)
    reasons += rs
    out: dict = {"rows": []}
    base = {"book_id": book, "page_index": pi, "pipeline_version": PIPELINE_VERSION, "config_hash": cfg,
            "profile": profile.value}
    pre = col.is_precropped(snap)
    try:
        if pre:
            gray, meta = _precropped_page(snap, ocr, pi, source_dpi)
        else:
            gray, meta = load_page_gray(master, pi, int(th.value("render_dpi")), source_dpi)
    except RenderError as e:
        rec, rs = G.g0b(False, None, th)
        out["page_row"] = {**base, "gates": gates + [rec], "exclusion_reasons": reasons + rs, "error": str(e)}
        append_page_row(layout, out["page_row"])
        return out
    page_png_sha = None
    if store_pages:
        from .render import encode_png
        from ..store import write_immutable
        page_png_sha = write_immutable(layout.page_image(book, pi), encode_png(gray))
    m = page_measures(gray, th, meta["physical_dpi"])
    m.pop("_ink")
    ocr_page = ocr["pages"][pi] if ocr and pi < len(ocr["pages"]) else None
    ocr_status = "ok" if ocr_page is not None and not pre else "unavailable"
    script = col.detect_script(" ".join(b["content"] for b in ocr_page["blocks"])) if ocr_page else None
    frame = None
    if ocr_page is not None:          # Paddle frame vs our render of the master: ratio of the long sides
        frame = {"frame_wh": [ocr_page["frame_w"], ocr_page["frame_h"]], "rotation_deg": ocr_page.get("angle"),
                 "scale_from_render": max(ocr_page["frame_w"], ocr_page["frame_h"]) / max(gray.shape)}
    # figure-only page: Paddle found item blocks and no text block (no prose, caption or body text)
    figure_only = bool(ocr_page is not None and ocr_page["image_crops"] and not col.frame_text_blocks(ocr_page))
    for fn in (lambda: G.g0b(True, m["eff_res_xheight_px"], th, figure_only), lambda: G.g0c(m["ink_frac_page"], th),
               lambda: G.g0d(ocr_status, script, m["skew_deg"], frame)):
        rec, rs = fn()
        gates.append(rec)
        reasons += rs
    crops, unreadable = [], []
    if ocr_page is not None and not reasons and all(g["passed"] for g in gates):
        for bbox, rel in sorted(ocr_page["image_crops"].items()):
            try:
                crop = col.load_crop_gray(snap, rel)
            except col.OcrMalformed as e:
                unreadable.append({"file": rel, "error": str(e)})
                continue
            if crop.shape != (bbox[3] - bbox[1], bbox[2] - bbox[0]):
                unreadable.append({"file": rel, "error": f"crop {crop.shape} differs from block_bbox {list(bbox)}"})
                continue
            crops.append((bbox, rel, crop))
    out["page_row"] = {**base, "gates": gates, "exclusion_reasons": reasons, "page_image_sha256": page_png_sha,
                       "render": meta, "eff_res_xheight_px": m["eff_res_xheight_px"],
                       "ink_frac_page": m["ink_frac_page"], "skew_deg": m["skew_deg"],
                       "ocr_status": ocr_status, "ocr_problem": ocr_problem, "paddle_frame": frame,
                       "script_detected": script, "crops_unreadable": unreadable, "figure_only": figure_only}
    append_page_row(layout, out["page_row"])
    if not crops or frame is None or ocr_page is None:
        return out
    s = frame["scale_from_render"]
    xh = m["eff_res_xheight_px"] * s if m["eff_res_xheight_px"] else None
    dpi = meta["physical_dpi"] * s if meta["physical_dpi"] else None
    units = col.frame_text_blocks(ocr_page)
    for bbox, rel, crop in crops:
        ox, oy = bbox[0], bbox[1]
        ctx = {"crop": crop, "origin": (ox, oy),
               "frame_shape": None if pre else (ocr_page["frame_h"], ocr_page["frame_w"]),
               "xh": xh, "skew": m["skew_deg"], "dpi": dpi, "ocr_status": ocr_status,
               "units": [dict(u, bbox=_shift(u["bbox"], -ox, -oy)) for u in units]}
        fz = {"box": list(bbox), "box_source": col.PRECROPPED_SOURCE if pre else col.REGION_SOURCE,
              "source_crop": col.source_crop_record(snap, ocr_page, bbox, rel)}
        out["rows"].extend(proposal_ladder(fz, ctx, base, th))
    return out


def _precropped_page(snap: dict, ocr: dict | None, pi: int, source_dpi) -> tuple:
    """The page of a pre-cropped source is its image, decoded as the crop is (no render)."""
    page = ocr["pages"][pi] if ocr and pi < len(ocr["pages"]) else None
    if not page or not page["image_crops"]:
        raise RenderError(f"{snap['source_name']} image {pi}: does not decode")
    try:
        gray = col.load_crop_gray(snap, next(iter(page["image_crops"].values())))
    except col.OcrMalformed as e:
        raise RenderError(str(e)) from e
    from .render import RENDER_VERSION
    return gray, {"renderer": "precropped", "renderer_version": col.PRECROPPED_SOURCE, "render_dpi": None,
                  "physical_dpi": float(source_dpi) if source_dpi else None, "width": int(gray.shape[1]),
                  "height": int(gray.shape[0]), "color_mode": "gray8", "render_version": RENDER_VERSION}


def _shift(b, dx: int, dy: int) -> list[int]:
    return [int(b[0]) + dx, int(b[1]) + dy, int(b[2]) + dx, int(b[3]) + dy]


def proposal_ladder(fz: dict, ctx: dict, base: dict, th, *, parent: str | None = None, depth: int = 0) -> list[dict]:
    """G2..G7d for one Paddle crop or one of its panels. Returns rows (children of a panel split included).

    `fz["box"]` and every recorded box are in Paddle's frame; the gates see the crop, in crop
    coordinates (`ctx["origin"]` is the crop's top-left corner in the frame), with OCR text blocks
    shifted to match. G7a records the border measures and never excludes (D10)."""
    gray, (ox, oy), xh, dpi = ctx["crop"], ctx["origin"], ctx["xh"], ctx["dpi"]
    box = [int(v) for v in fz["box"]]
    loc = _shift(box, -ox, -oy)
    units = ctx["units"]
    pid = proposal_id(base["book_id"], base["page_index"], box, fz["box_source"], parent)
    row = {k: v for k, v in base.items() if k != "profile"}
    row.update(proposal_id=pid, parent_proposal_id=parent, box_source=fz["box_source"], box_xyxy_page_px=box,
               box_frame=fz["source_crop"]["frame"], source_crop=fz["source_crop"], ocr_status=ctx["ocr_status"], gates=[],
               exclusion_reasons=[], model_calls=[])
    crop = gray[loc[1]:loc[3], loc[0]:loc[2]]
    stat_dpi = dpi or 300
    an = analyze(crop, sauvola_ink(crop, sauvola_window(th, stat_dpi), th.value("sauvola_k")))
    row["img_stats"] = numeric_stats(compute_from_analysis(an, th, eff_res_xheight_px=xh, skew_deg=ctx["skew"]))
    out: list[dict] = []

    def step(result) -> bool:
        rec, rs = result[0], result[1]
        row["gates"].append(rec)
        row["exclusion_reasons"] += [r for r in rs if r not in row["exclusion_reasons"]]
        return rec["passed"]
    for fn in (lambda: _g2(box, ctx["frame_shape"], gray.shape, dpi, th), lambda: G.g3(row["img_stats"]["ink_frac"], th),
               lambda: G.g4(an, th), lambda: G.g5(crop, th), lambda: G.g6(an, th)):
        if not step(fn()):
            return [finalize(row)]
    rec, rs, border = G.g7a(gray, loc, th, stat_dpi)
    row.update(border_ink=border["border_ink"], touching_cc=border["touching_cc"], growth_steps=0)
    step((rec, rs))                                  # records only (D10)
    b = G.g7b(gray, loc, xh, th, stat_dpi, units, depth)
    step((b["record"], b["reasons"]))
    if b["action"] == "split":
        out.append(finalize(row))
        for pbox in b["panels"]:
            child = dict(fz, box=_shift(pbox, ox, oy), box_source=f"panel_of:{fz['box_source']}")
            out.extend(proposal_ladder(child, ctx, base, th, parent=pid, depth=depth + 1))
        return out
    if b["action"] == "exclude":
        return [finalize(row)]
    final = [int(v) for v in b["box"]]
    row["box_xyxy_page_px"] = _shift(final, ox, oy)
    if b["caption"]:
        row["caption_stripped"] = b["caption"]["caption_stripped"]
        row["caption_text"] = b["caption"]["caption_text"]
    row["prose_chars_in_crop"] = b["record"]["measured_value"].get("prose_chars")
    if not step(G.g7c(gray, final, xh, th, stat_dpi, units)):
        return [finalize(row)]
    rec, rs, ph = G.g7d(gray[final[1]:final[3], final[0]:final[2]], th)
    row["phash64"] = ph
    step((rec, rs))
    return [finalize(row)]


def _g2(box, frame_shape, crop_shape, dpi, th):
    """G2. A pre-cropped source (`frame_shape` None) has no page: the crop/page area test does not apply, so
    G2 runs against the image itself and its area outcome is set aside (size and aspect still apply)."""
    if frame_shape is not None:
        return G.g2(box, frame_shape, dpi, th)
    rec, reasons = G.g2(box, crop_shape, dpi, th)
    reasons = [r for r in reasons if r != Exclusion.box_too_large.value]
    rec = dict(rec, passed=not reasons, measured_value=dict(rec["measured_value"], area_frac=None),
               threshold=dict(rec["threshold"], area_frac_max="not_applicable: pre-cropped source, no page"))
    return rec, reasons


def _cluster(layout: Layout, rows: list[dict], th) -> list[dict]:
    """G7d: union-find over this batch plus every pHash already in the funnel (cluster only).

    Sets cluster fields on the batch rows and returns updated copies of EARLIER funnel rows whose
    cluster id or size changed (to be appended; the funnel is never rewritten)."""
    from .funnel import latest_rows
    prior = {pid: r for pid, r in latest_rows(layout).items() if r.get("phash64")}
    hashes = {pid: r["phash64"] for pid, r in prior.items()}
    for r in rows:
        if r.get("phash64"):
            hashes[r["proposal_id"]] = r["phash64"]
    if not hashes:
        return []
    cl = exact_and_near(hashes, int(th.value("phash_near_hamming")))
    batch = {r["proposal_id"] for r in rows}
    for r in rows:
        if r.get("phash64"):
            c = cl[r["proposal_id"]]
            r["phash_cluster_id"], r["cluster_size"] = c["phash_cluster_id"], c["cluster_size"]
    updates = []
    for pid, r in sorted(prior.items()):
        if pid in batch:
            continue
        c = cl[pid]
        if (r.get("phash_cluster_id"), r.get("cluster_size")) != (c["phash_cluster_id"], c["cluster_size"]):
            updates.append({**r, "phash_cluster_id": c["phash_cluster_id"], "cluster_size": c["cluster_size"]})
    return updates


def summarize(summary: dict) -> dict:
    """Counts for a smoke report: pages by first failing page gate, rows by terminal gate / reason."""
    pages_term: dict[str, int] = {}
    for p in summary["pages"]:
        t = next((g["gate_id"] for g in p["gates"] if not g["passed"]), "passed")
        pages_term[t] = pages_term.get(t, 0) + 1
    term: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for r in summary["rows"]:
        term[r["terminal_gate"]] = term.get(r["terminal_gate"], 0) + 1
        for x in r["exclusion_reasons"]:
            reasons[x] = reasons.get(x, 0) + 1
    ocr: dict[str, int] = {}
    for p in summary["pages"]:
        k = str(p.get("ocr_status"))
        ocr[k] = ocr.get(k, 0) + 1
    return {"book_id": summary["book_id"], "source": summary["source_name"], "r3_passed": summary["r3"]["passed"],
            "r3_failures": summary["r3"]["failures"], "n_pages": len(summary["pages"]),
            "pages_by_terminal_gate": pages_term, "ocr_status": ocr,
            "crops_unreadable": sum(len(p.get("crops_unreadable") or []) for p in summary["pages"]),
            "n_proposals": len(summary["rows"]), "rows_by_terminal_gate": term, "exclusions": reasons}
