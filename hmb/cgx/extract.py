"""§10.2 deterministic candidate extraction (CGX) from the compiled PDF.

    extract_cgx(pdf_path, *, item_id, model_id, track, th) -> CgxRecord

Pipeline:
 1. Paths: PyMuPDF `page.get_drawings()` (content stream with the CTM applied, page frame, y down,
    PDF points). Invisible paths (no stroke and no fill, or a white fill without stroke) are dropped.
    Each path is split into subpaths (item i starts where item i-1 ends); `re`/`qu` items are closed
    quadrilaterals.
 2. Primitives (entity types of the ontology, geometry in the shared evaluator format agreed with
    Opus 5, `hmb.synthetic.geo_predicates`):
      * closed Bézier runs: circle fit (Kasa) accepted when max radial residual / r <=
        `circle_fit_resid_rel`; radius <= eps_on -> point marker (a drawn dot); otherwise an ellipse
        fit with the same relative tolerance; otherwise `curve_generic` (routes to `model`).
      * open Bézier runs: arc when the circle fit passes, tested BEFORE straightness (a short marker
        arc is nearly straight; an arc of radius <= ANGLE_MARK_REL*D centred on a vertex where >= 2
        linear primitives meet is an `angle_marker`); else a nearly straight Bézier (all samples within
        STRAIGHT_REL*eps_on of the chord) is a segment; else `curve_generic`.
      * line runs: one `segment` per piece (closed runs included); open runs of >= CURVE_MIN_SEGS
        pieces whose every turn is <= CURVE_MAX_TURN_DEG -> `curve_generic` (plots).
      * polygons (C30/C34, CONTRACTS "Polygons"; cgx-1.5.0), by the shared pure
        `polygons_from_strokes` (Γ imports the same function). ONE rule, independent of drawing style
        and of corner labelling: a polygon is a simple cycle of the straight-stroke graph (nodes =
        stroke ends merged within eps_on) which, after merging collinear corners (turn <=
        parallel_deg) into one side, has no chord (a drawn stroke between two non-adjacent corners)
        and at most POLY_MAX_CORNERS (8) corners; reversals (turn >= 180 - parallel_deg),
        self-intersection and near-zero area are rejected. A closed path and the same cycle drawn as
        separate strokes are therefore identical, labelled or not (§7.6 source_refactor).
        Documented case: a square WITH both diagonals has chords, so it yields only the 4
        corner-sharing triangles, in BOTH drawing styles (no square polygon).
        Every polygon edge is also a `segment` entity (a drawn stroke with those ends, else a
        synthesized one); polygon-edge segments stay `segment` (not retyped by the C28 stroke rule).
      * arrow tips: a small subpath (bbox diagonal <= TIP_REL*D) of >= 2 pieces or closed (a single
        straight piece is a stroke, e.g. a tick, never a tip) within eps_on of the end of a
        segment makes that segment an `arrow` (tail -> head); a long (>= axis_min_length_frac*D) horizontal or
        vertical arrow is an `axis`. Tips themselves are not primitives.
      * right-angle marks: an open two-piece polyline of short (<= MARK_REL*D) perpendicular pieces
        whose free ends lie on two longer linear primitives -> `right_angle_marked` at the corner
        those primitives share. Length ticks: a short segment crossing a longer segment's interior
        at >= TICK_MIN_ANGLE_DEG -> `length_tick`; parallel ticks on one host within TICK_REL*D of
        each other form ONE multi-tick `length_tick` with g["n"] = count (as Γ).
      * point entities (C28, CONTRACTS "Geometric predicate semantics"): a point entity exists only
        where the drawing gives it identity -- a drawn dot/marker, or an internal candidate location
        (segment/arrow/arc endpoints, polygon vertices, transversal crossings, tangency contacts,
        circle-circle intersections, circle/arc centres) that a name label promotes. Binding order
        (lead decision, cgx-1.3.0): a name label with a drawn dot within label_bind binds to it and
        promotes nothing; only otherwise does it promote its nearest internal candidate within
        label_bind (one label promotes at most one candidate). Drawn dots carry g["marker"] = true. Candidates
        are merged within eps_on (priority marker > endpoint/vertex > centre > crossing/contact).
        Unpromoted candidates are internal geometry only: never entities, never fact arguments.
      * straight strokes (C28), typed by their extreme incident point entities (within eps_on):
        both ends -> segment; exactly one end -> ray (p = that point entity, q = the far drawn end);
        neither end but >= 1 incident point -> line (the stroke then extends past its extreme
        incident points by > eps_on; p, q = drawn ends); no incident point at all -> segment
        (form choice, agreed with Opus 5 for Γ). A tipped stroke (arrow tip at an end) is `axis` iff it
        is horizontal/vertical within parallel_deg and long (>= axis_min_length_frac*D) or carries an
        axis-name label; otherwise `arrow`. An untipped stroke with an axis-name label is `axis`.
        "Carries an axis-name label" is decided exactly as Γ does (cgx-1.5.0): the AXIS_NAMES labels
        are bound with the shared `bind_labels` over all current entities, and a stroke carries one
        iff such a label binds to it. An axis-name label within label_bind of ANY straight stroke
        (distance to the stroke segment) never promotes a point. Typing is done by the SHARED
        `geo_predicates.classify_straight` (the same function Γ uses).
      * tangency envelope (documented asymmetry): a tangency contact point is a candidate point when
        |dist(centre, line) - r| <= tangency_rel*r (a line just missing OR just cutting the circle);
        the shared evaluator then decides tangent_to, where the chord check applies only to secants
        (a line cutting the circle with half-chord > eps_on is not tangent) while an external
        near-contact is judged by the radial tolerance alone.
      * canonical output (cgx-1.5.0): undirected primitives (segment, line, length_tick) store their
        end points in lexicographic order (rounded to 1e-4 pt) before sorting and id assignment, so
        drawing `B -- A` instead of `A -- B` gives byte-identical CGX (except pdf_sha256). Every
        point primitive carries an explicit g["marker"] (true only for a drawn dot).
 3. Labels: `get_text("rawdict")` per-glyph boxes and Unicode (NFC). Consecutive glyphs are one label
    when their boxes overlap vertically and the horizontal gap is <= GLYPH_GAP_REL * glyph size.
    Label kind rule (cgx-1.1.0): a string is a "measure" iff it contains a digit AND (a degree/prime
    mark, a unit suffix, a decimal separator between digits, or an operator + - x . / : = ^), or it is a
    digit string longer than MAX_NAME_DIGITS; everything else (letters, A1, A', short numerals such
    as "1" or "12") is a candidate "name" label and binds. Measures are recorded but not passed to
    the evaluator (measure_label is model-class).
    Binding is per label INSTANCE (duplicate texts each bind to their own primitive) with the shared
    §10.3 rule (`geo_predicates.bind_labels`); CgxPrimitive.label = the nearest bound instance.
 4. Facts: `evaluate_geo(geom, tolerances_10_2(th, D))` (the ONE shared predicate function), then
    canonicalized with `hmb.cards.closure.minimal_generating_set` (§8.2) so the universe does not
    inflate with primitive count. `margin` is the evaluator's margin in ε units.
 D = diagonal of the ink bounding box (union of visible path rects and glyph boxes), in points.
Every number below that is not a registered parameter is an extractor-internal constant versioned by
EXTRACTOR_VERSION. Coordinates are rounded to 1e-4 pt before evaluation for bit-stable output.
"""
from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

import numpy as np

from hmb.cards.closure import minimal_generating_set
from hmb.synthetic.geo_predicates import (AXIS_NAMES, GEO_FORMAT_VERSION, bind_labels,
                                        classify_straight, evaluate_geo, tolerances_10_2)
from hmb.hashing import sha256_bytes, sha256_json
from hmb.ontology import PREDICATES, fact_key
from hmb.schemas import CgxFact, CgxPrimitive, CgxRecord, validate_record
from hmb.vocab import Track

EXTRACTOR_VERSION = "cgx-1.6.0"   # 1.6.0: C43 primitive bound; 1.5.0: red-team round 2 (C34 polygons, axis binding, canonical ends)

# extractor-internal constants (versioned by EXTRACTOR_VERSION; not registered parameters)
BEZ_SAMPLES = 16            # samples per cubic for fitting
JOIN_TOL_PT = 1e-3          # subpath continuity
STRAIGHT_REL = 0.1          # straight-Bezier test, fraction of eps_on
TIP_REL = 0.04              # arrow-tip bbox diagonal / D
MARK_REL = 0.08             # right-angle mark piece length / D
TICK_REL = 0.05             # length-tick length / D
TICK_MIN_ANGLE_DEG = 60.0
ANGLE_MARK_REL = 0.12       # angle-marker arc radius / D
CURVE_MIN_SEGS = 6
CURVE_MAX_TURN_DEG = 30.0
POLY_MAX_CORNERS = 8        # C34 (contract constant): at most 8 corners
POLY_MAX_LEN = 16           # enumeration bound: strokes per cycle (sides may be split into pieces)
POLY_MAX_CYCLES = 2000      # enumeration bound: cycles examined (deterministic order)
GLYPH_GAP_REL = 0.35
ROUND = 4
MAX_NAME_DIGITS = 3
_MARKS_RX = re.compile("[\u00b0\u2032\u2033'\"]")
_UNIT_RX = re.compile(r"\d\s*(cm|mm|dm|km|m|in|ft|yd)\b")
_DECIMAL_RX = re.compile(r"\d[.,]\d")
_OPERATOR_RX = re.compile("[+\u2212\\-\u00d7\u00b7/:=^]")


def label_kind(text: str) -> str:
    """'measure' or 'name' (rule in the module docstring, versioned by EXTRACTOR_VERSION)."""
    t = text.strip()
    if not any(ch.isdigit() for ch in t):
        return "name"
    if _MARKS_RX.search(t) or _UNIT_RX.search(t) or _DECIMAL_RX.search(t) or _OPERATOR_RX.search(t):
        return "measure"
    if t.isdigit() and len(t) > MAX_NAME_DIGITS:
        return "measure"
    return "name"

_TYPE_ORDER = ("point", "segment", "ray", "line", "arrow", "axis", "length_tick", "circle", "arc", "ellipse",
               "polygon", "angle_marker", "right_angle_marked", "curve_generic")
_PREFIX = {"point": "pt", "segment": "sg", "ray": "ry", "line": "ln", "arrow": "aw", "axis": "ax", "length_tick": "tk",
           "circle": "ci", "arc": "ac", "ellipse": "el", "polygon": "pg", "angle_marker": "am",
           "right_angle_marked": "ra", "curve_generic": "cv"}


# ----------------------------------------------------------------------------- geometry helpers
def _bez(p0, p1, p2, p3, n=BEZ_SAMPLES) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def _fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Kasa fit -> (centre, r, max |d - r| / r)."""
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    b = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    (d, e, f), *_ = np.linalg.lstsq(A, b, rcond=None)
    c = np.array([-d / 2, -e / 2])
    r2 = c @ c - f
    if r2 <= 0:
        return c, 0.0, math.inf
    r = math.sqrt(r2)
    res = float(np.max(np.abs(np.hypot(*(pts - c).T) - r)) / r)
    return c, r, res


def _fit_ellipse(pts: np.ndarray):
    """Algebraic conic fit ax^2+bxy+cy^2+dx+ey=1 -> (centre, rx, ry, rot_deg, rel residual) or None."""
    m = pts.mean(axis=0)
    q = pts - m
    x, y = q[:, 0], q[:, 1]
    A = np.c_[x * x, x * y, y * y, x, y]
    sol, *_ = np.linalg.lstsq(A, np.ones(len(q)), rcond=None)
    a, b, c, d, e = sol
    M = np.array([[a, b / 2], [b / 2, c]])
    if np.linalg.det(M) <= 0:
        return None
    ctr = np.linalg.solve(2 * M, -np.array([d, e]))
    k = 1 + ctr @ M @ ctr
    if k <= 0:
        return None
    w, V = np.linalg.eigh(M / k)
    if np.any(w <= 0):
        return None
    rx, ry = 1 / math.sqrt(w[0]), 1 / math.sqrt(w[1])
    rot = math.degrees(math.atan2(V[1, 0], V[0, 0]))
    loc = (q - ctr) @ V
    rel = np.sqrt((loc[:, 0] / rx) ** 2 + (loc[:, 1] / ry) ** 2)
    return ctr + m, rx, ry, rot, float(np.max(np.abs(rel - 1)))


def _seg_point_dist(p, a, b) -> tuple[float, float]:
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 == 0 else float(np.clip((p - a) @ ab / L2, 0, 1))
    return float(np.hypot(*(p - (a + t * ab)))), t


def _seg_intersection(a, b, c, d):
    r, s = b - a, d - c
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / den
    u = ((c - a)[0] * r[1] - (c - a)[1] * r[0]) / den
    return a + t * r, t, u


def _angle_deg(u, v) -> float:
    cu = float(np.hypot(*u)) * float(np.hypot(*v))
    if cu == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v) / cu))))


def _self_intersecting(pts: list) -> bool:
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            hit = _seg_intersection(edges[i][0], edges[i][1], edges[j][0], edges[j][1])
            if hit is not None and 1e-9 < hit[1] < 1 - 1e-9 and 1e-9 < hit[2] < 1 - 1e-9:
                return True
    return False


def _corner_cycle(pts: list, straight_deg: float):
    """(indices of corners, reason) for a closed vertex cycle; None when rejected."""
    n = len(pts)
    if n < 3:
        return None
    turns = [_angle_deg(pts[i] - pts[i - 1], pts[(i + 1) % n] - pts[i]) for i in range(n)]
    if any(t >= 180.0 - straight_deg for t in turns):
        return None                         # a reversal: the cycle retraces a collinear edge (spike)
    if _self_intersecting(pts):
        return None
    corners = [i for i in range(n) if turns[i] > straight_deg]
    return corners if len(corners) >= 3 else None


def _canonical(vs: list, eps: float):
    area = 0.5 * sum(float(vs[i][0] * vs[(i + 1) % len(vs)][1] - vs[(i + 1) % len(vs)][0] * vs[i][1])
                     for i in range(len(vs)))
    if abs(area) <= eps * eps:
        return None, None
    if area < 0:
        vs = vs[::-1]
    r = [(round(float(v[0]), 2), round(float(v[1]), 2)) for v in vs]
    s0 = min(range(len(r)), key=lambda i: r[i])
    return vs[s0:] + vs[:s0], tuple(r[s0:] + r[:s0])


def polygons_from_strokes(strokes: list, eps: float, straight_deg: float, *, points: list | None = None,
                          closed_paths: list = (), max_corners: int = POLY_MAX_CORNERS,
                          max_len: int = POLY_MAX_LEN, max_cycles: int = POLY_MAX_CYCLES) -> list[list[np.ndarray]]:
    """C34 polygons (CONTRACTS "Polygons"; rule in the module docstring). Pure.

    strokes: [(p, q)] straight strokes. `points` and `closed_paths` are accepted for callers'
    convenience but NEVER change polygon membership (C34: one rule, independent of drawing style and
    corner labelling). Returns corner lists in canonical order (start at the smallest rounded corner,
    positive signed area in the given frame), sorted."""
    import networkx as nx
    del points, closed_paths
    out, seen = [], set()
    ordered = sorted(((np.asarray(a, float), np.asarray(b, float)) for a, b in strokes),
                     key=lambda ab: (round(float(min(ab[0][0], ab[1][0])), 3), round(float(min(ab[0][1], ab[1][1])), 3),
                                     round(float(max(ab[0][0], ab[1][0])), 3), round(float(max(ab[0][1], ab[1][1])), 3)))
    nodes: list[np.ndarray] = []

    def node(pt):
        for i, q in enumerate(nodes):
            if float(np.hypot(*(pt - q))) <= eps:
                return i
        nodes.append(pt)
        return len(nodes) - 1
    edges = set()
    for a, b in ordered:
        i, j = node(a), node(b)
        if i != j:
            edges.add((min(i, j), max(i, j)))
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(sorted(edges))
    for k, cyc in enumerate(nx.simple_cycles(G, length_bound=max_len)):
        if k >= max_cycles:
            break
        cc = _corner_cycle([nodes[i] for i in cyc], straight_deg)
        if cc is None or len(cc) > max_corners:
            continue
        cn = [cyc[i] for i in cc]
        m = len(cn)
        if any((min(cn[x], cn[y]), max(cn[x], cn[y])) in edges
               for x in range(m) for y in range(x + 2, m) if not (x == 0 and y == m - 1)):
            continue                        # a chord between non-adjacent corners
        vs, key = _canonical([nodes[i] for i in cn], eps)
        if key is not None and key not in seen:
            seen.add(key)
            out.append(vs)
    return sorted(out, key=lambda vs: [(round(float(v[0]), 2), round(float(v[1]), 2)) for v in vs])


def _is_white(col) -> bool:
    return col is not None and all(abs(ch - 1.0) < 1e-6 for ch in col[:3])


# ----------------------------------------------------------------------------- PDF parsing
def _subpaths(d: dict) -> list[dict]:
    """Split one drawing into subpaths: {'items': [(kind, [pts])], 'closed': bool}."""
    out: list[dict] = []
    cur: list = []
    last = None
    for it in d["items"]:
        kind = it[0]
        if kind == "re":
            r = it[1]
            pts = [np.array([r.x0, r.y0]), np.array([r.x1, r.y0]), np.array([r.x1, r.y1]), np.array([r.x0, r.y1])]
            out.append({"items": [("l", [pts[i], pts[(i + 1) % 4]]) for i in range(4)], "closed": True})
            continue
        if kind == "qu":
            qd = it[1]
            pts = [np.array([p.x, p.y]) for p in (qd.ul, qd.ur, qd.lr, qd.ll)]
            out.append({"items": [("l", [pts[i], pts[(i + 1) % 4]]) for i in range(4)], "closed": True})
            continue
        pts = [np.array([p.x, p.y], dtype=float) for p in it[1:]]
        if last is None or float(np.hypot(*(pts[0] - last))) > JOIN_TOL_PT:
            if cur:
                out.append({"items": cur, "closed": False})
            cur = []
        cur.append((kind, pts))
        last = pts[-1]
    if cur:
        out.append({"items": cur, "closed": False})
    for i, sp in enumerate(out):
        if not sp["closed"] and sp["items"]:
            a, z = sp["items"][0][1][0], sp["items"][-1][1][-1]
            if float(np.hypot(*(a - z))) <= JOIN_TOL_PT or (d.get("closePath") and i == len(out) - 1):
                sp["closed"] = True
    return out


def _runs(sp: dict) -> list[tuple[str, list, bool]]:
    """Homogeneous runs of 'l' or 'c' items -> (kind, items, closed-if-whole-subpath)."""
    runs: list[tuple[str, list]] = []
    for kind, pts in sp["items"]:
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(pts)
        else:
            runs.append((kind, [pts]))
    whole = len(runs) == 1
    return [(k, its, sp["closed"] and whole) for k, its in runs]


def _glyphs(page) -> list[dict]:
    import fitz
    raw = page.get_text("rawdict", flags=fitz.TEXT_INHIBIT_SPACES | fitz.TEXT_MEDIABOX_CLIP)
    out = []
    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                for ch in sp.get("chars", []):
                    c = ch.get("c", "")
                    if not c or c.isspace():
                        continue
                    out.append({"text": unicodedata.normalize("NFC", c),
                                "bbox": [round(float(v), ROUND) for v in ch["bbox"]],
                                "font": sp.get("font", ""), "size": round(float(sp.get("size", 0.0)), ROUND)})
    return out


def _group_labels(glyphs: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for g in glyphs:
        x0, y0, x1, y1 = g["bbox"]
        if groups:
            G = groups[-1]
            gx0, gy0, gx1, gy1 = G["bbox"]
            size = max(g["size"], G["size"], 1e-6)
            gap = x0 - gx1
            vover = min(y1, gy1) - max(y0, gy0)
            if -0.5 * size <= gap <= GLYPH_GAP_REL * size and vover > 0:
                G["text"] += g["text"]
                G["bbox"] = [min(gx0, x0), min(gy0, y0), max(gx1, x1), max(gy1, y1)]
                G["size"] = max(G["size"], g["size"])
                continue
        groups.append({"text": g["text"], "bbox": list(g["bbox"]), "size": g["size"]})
    for G in groups:
        G["text"] = unicodedata.normalize("NFC", G["text"])
        G["kind"] = label_kind(G["text"])
    return groups


# ----------------------------------------------------------------------------- primitives
def _r(v) -> list[float]:
    return [round(float(v[0]), ROUND), round(float(v[1]), ROUND)]


def _build(page, th) -> tuple[dict, list[dict], list[dict], float, int]:
    drawings = page.get_drawings()
    path_ops = sum(len(d["items"]) for d in drawings)
    visible = [d for d in drawings
               if not (d.get("color") is None and (d.get("fill") is None or _is_white(d.get("fill"))))
               and not ((d.get("stroke_opacity") or 1) == 0 and (d.get("fill_opacity") or 1) == 0)]
    glyphs = _glyphs(page)
    labels = _group_labels(glyphs)
    rects = [d["rect"] for d in visible] + [g["bbox"] for g in glyphs]
    if not rects:
        return {"y_up": False, "entities": {}, "texts": []}, glyphs, labels, 0.0, path_ops
    xs0 = [float(r[0]) for r in rects]; ys0 = [float(r[1]) for r in rects]
    xs1 = [float(r[2]) for r in rects]; ys1 = [float(r[3]) for r in rects]
    D = math.hypot(max(xs1) - min(xs0), max(ys1) - min(ys0))
    eps = float(th.value("eps_on_frac")) * D
    resid_tol = float(th.value("circle_fit_resid_rel"))
    perp_deg = float(th.value("perpendicular_deg"))
    par_deg = float(th.value("parallel_deg"))
    bind = float(th.value("label_bind_frac")) * D

    prims: list[dict] = []           # {"type", "g", ...}
    markers: list[np.ndarray] = []
    closed_paths: list[list[np.ndarray]] = []   # C33 rule (a): paths drawn as one closed cycle
    deferred: list[tuple] = []       # small subpath runs: arrow-tip candidates, classified later
    polylines: list[list[np.ndarray]] = []   # open two-piece line runs, examined for right-angle marks

    def classify(kind, items, closed, diag, allow_defer):
        if kind == "c":
            samp = np.vstack([_bez(*its) for its in items])
            c, r, res = _fit_circle(samp)
            if closed and len(items) >= 2 and res <= resid_tol:
                if r <= eps:
                    markers.append(c)
                else:
                    prims.append({"type": "circle", "g": {"c": c, "r": r}})
                return
            if allow_defer and diag <= TIP_REL * D and not closed:
                deferred.append((kind, items, closed, diag, samp))
                return
            chord_a, chord_b = samp[0], samp[-1]
            if not closed and res <= resid_tol and r < 10 * D:   # arc first: a short marker arc is nearly straight
                yup = samp * np.array([1.0, -1.0])
                cu = c * np.array([1.0, -1.0])
                ang = np.unwrap(np.arctan2(yup[:, 1] - cu[1], yup[:, 0] - cu[0]))
                p0, q0 = (samp[0], samp[-1]) if ang[-1] - ang[0] > 0 else (samp[-1], samp[0])
                prims.append({"type": "arc", "g": {"c": c, "r": r, "p": p0, "q": q0}})
                return
            if not closed and max(_seg_point_dist(q, chord_a, chord_b)[0] for q in samp) <= STRAIGHT_REL * eps:
                prims.append({"type": "segment", "g": {"p": chord_a, "q": chord_b}})
                return
            if closed:
                el = _fit_ellipse(samp)
                if el is not None and el[4] <= resid_tol:
                    ctr, rx, ry, rot, _ = el
                    rot_c = round(rot % 180.0, ROUND) % 180.0      # ellipse symmetry: canonical [0, 180)
                    # page frame (y down); to_internal flips it exactly once
                    prims.append({"type": "ellipse", "g": {"c": ctr, "rx": rx, "ry": ry, "rot_deg": rot_c}})
                    return
            prims.append({"type": "curve_generic", "g": {"points": samp}})
            return
        verts = [items[0][0]] + [its[1] for its in items]
        if closed and float(np.hypot(*(verts[0] - verts[-1]))) <= JOIN_TOL_PT:
            verts = verts[:-1]
        if allow_defer and diag <= TIP_REL * D and not (closed and len(verts) >= 3 and diag > 2 * eps) \
                and (len(items) >= 2 or closed):         # one straight piece is a stroke (e.g. a tick), not a tip
            deferred.append((kind, items, closed, diag, np.array(verts)))
            return
        if closed and len(verts) >= 3:          # edges as strokes; the polygon itself is decided later (C33)
            closed_paths.append(list(verts))
            for i in range(len(verts)):
                prims.append({"type": "segment", "g": {"p": verts[i], "q": verts[(i + 1) % len(verts)]}})
            return
        pieces = list(zip(verts[:-1], verts[1:]))
        if len(pieces) >= CURVE_MIN_SEGS and all(
                _angle_deg(b1 - a1, c2 - b1) <= CURVE_MAX_TURN_DEG
                for (a1, b1), (_b2, c2) in zip(pieces[:-1], pieces[1:])):
            prims.append({"type": "curve_generic", "g": {"points": np.array(verts)}})
            return
        run = None
        if len(pieces) == 2:
            polylines.append(verts)
            run = len(polylines)
        for a1, b1 in pieces:
            if float(np.hypot(*(b1 - a1))) > JOIN_TOL_PT:
                prims.append({"type": "segment", "g": {"p": a1, "q": b1}, "run": run})

    for d in visible:
        for sp in _subpaths(d):
            allpts = np.array([q for _k, pts in sp["items"] for q in pts])
            diag = float(np.hypot(*(allpts.max(0) - allpts.min(0)))) if len(allpts) else 0.0
            for kind, items, closed in _runs(sp):
                classify(kind, items, closed, diag, True)
    # --- arrow tips: a small subpath within eps_on of a segment end; others are classified normally
    for kind, items, closed, diag, tip in deferred:
        best, bd, which = None, math.inf, None
        for s in prims:
            if s["type"] not in ("segment", "arrow"):
                continue
            for end in ("p", "q"):
                dd = float(np.min(np.hypot(*(tip - s["g"][end]).T)))
                if dd < bd:
                    best, bd, which = s, dd, end
        if best is not None and bd <= eps:
            g = best["g"]
            if which == "p":
                g["p"], g["q"] = g["q"], g["p"]
            best["type"] = "arrow"
            best["tipped"] = True
        else:
            classify(kind, items, closed, diag, False)
    # --- right-angle marks (two short perpendicular pieces whose free ends lie on longer lines)
    long_lin = lambda: [q for q in prims if q["type"] in ("segment", "arrow", "axis") and
                        float(np.hypot(*(q["g"]["q"] - q["g"]["p"]))) > MARK_REL * D]
    for k, verts in enumerate(polylines, start=1):
        a, corner, b = verts
        la, lb = float(np.hypot(*(corner - a))), float(np.hypot(*(b - corner)))
        if not (0 < la <= MARK_REL * D and 0 < lb <= MARK_REL * D):
            continue
        if abs(_angle_deg(corner - a, b - corner) - 90.0) > perp_deg * 2:
            continue
        L = long_lin()
        on_a = [s for s in L if _seg_point_dist(a, s["g"]["p"], s["g"]["q"])[0] <= eps]
        on_b = [s for s in L if _seg_point_dist(b, s["g"]["p"], s["g"]["q"])[0] <= eps]
        vertex = None
        for sa in on_a:
            for sb in on_b:
                if sa is sb:
                    continue
                hit = _seg_intersection(sa["g"]["p"], sa["g"]["q"], sb["g"]["p"], sb["g"]["q"])
                if hit is not None:
                    vertex = hit[0]
                    break
            if vertex is not None:
                break
        if vertex is None:
            continue
        prims[:] = [q for q in prims if q.get("run") != k]
        prims.append({"type": "right_angle_marked", "g": {"vertex": vertex, "size": max(la, lb)}})
    # --- length ticks
    segs = [p for p in prims if p["type"] == "segment"]
    for s in segs:
        a, b = s["g"]["p"], s["g"]["q"]
        L = float(np.hypot(*(b - a)))
        if L > TICK_REL * D:
            continue
        mid = (a + b) / 2
        for o in prims:
            if o is s or o["type"] not in ("segment", "arrow", "axis"):
                continue
            oa, ob = o["g"]["p"], o["g"]["q"]
            if float(np.hypot(*(ob - oa))) <= TICK_REL * D:
                continue
            dd, t = _seg_point_dist(mid, oa, ob)
            ang = _angle_deg(b - a, ob - oa)
            if dd <= eps and eps / max(float(np.hypot(*(ob - oa))), 1e-9) < t < 1 - eps / max(float(np.hypot(*(ob - oa))), 1e-9) \
                    and min(ang, 180 - ang) >= TICK_MIN_ANGLE_DEG:
                s["type"] = "length_tick"
                s["host"] = o
                break
    # --- a multi-tick mark (double/triple tick) is ONE length_tick: ticks crossing the same host stroke,
    # parallel within parallel_deg, with midpoints within TICK_REL*D of each other along the host, merge;
    # geometry = the mean of the group's p and q (the group centre), "n" = tick count (cgx-1.3.1, as Γ)
    tks = [p for p in prims if p["type"] == "length_tick"]
    groups: list[list[dict]] = []
    for t in sorted(tks, key=lambda p: (round(float((p["g"]["p"][0] + p["g"]["q"][0]) / 2), 3),
                                        round(float((p["g"]["p"][1] + p["g"]["q"][1]) / 2), 3))):
        mid = (t["g"]["p"] + t["g"]["q"]) / 2
        dirv = t["g"]["q"] - t["g"]["p"]
        home = None
        for g_ in groups:
            h = g_[0]
            hm = (h["g"]["p"] + h["g"]["q"]) / 2
            if (h.get("host") is t.get("host") and float(np.hypot(*(mid - hm))) <= TICK_REL * D * len(g_)
                    and min(_angle_deg(dirv, h["g"]["q"] - h["g"]["p"]), 180 - _angle_deg(dirv, h["g"]["q"] - h["g"]["p"])) <= par_deg):
                home = g_
                break
        if home is None:
            groups.append([t])
        else:
            home.append(t)
    merged_ticks = []
    for g_ in groups:
        ref = g_[0]["g"]["q"] - g_[0]["g"]["p"]
        ps = [t["g"]["p"] if float(ref @ (t["g"]["q"] - t["g"]["p"])) >= 0 else t["g"]["q"] for t in g_]
        qs = [t["g"]["q"] if float(ref @ (t["g"]["q"] - t["g"]["p"])) >= 0 else t["g"]["p"] for t in g_]
        merged_ticks.append({"type": "length_tick",
                             "g": {"p": np.mean(ps, axis=0), "q": np.mean(qs, axis=0), "n": len(g_)}})
    prims = [p for p in prims if p["type"] != "length_tick"] + merged_ticks
    # --- de-duplicate identical primitives (overdraw)
    uniq: list[dict] = []
    seen: set = set()
    for p in prims:
        key = (p["type"], sha256_json(_jsonable(p["g"], nd=2)))
        if p["type"] in ("segment", "length_tick"):
            a, b = sorted([_r(p["g"]["p"]), _r(p["g"]["q"])])
            key = (p["type"], round(a[0], 2), round(a[1], 2), round(b[0], 2), round(b[1], 2))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    prims = uniq
    # --- points
    cand: list[tuple[int, np.ndarray]] = []
    lin = [p for p in prims if p["type"] in ("segment", "arrow", "axis")]
    for p in prims:
        t = p["type"]
        if t in ("segment", "arrow", "axis"):
            cand += [(1, p["g"]["p"]), (1, p["g"]["q"])]
        elif t == "arc":
            cand += [(1, p["g"]["p"]), (1, p["g"]["q"])]
    for p in prims:
        if p["type"] in ("circle", "arc"):
            cand.append((2, p["g"]["c"]))
    for i in range(len(lin)):
        for j in range(i + 1, len(lin)):
            a, b, c, d = lin[i]["g"]["p"], lin[i]["g"]["q"], lin[j]["g"]["p"], lin[j]["g"]["q"]
            hit = _seg_intersection(a, b, c, d)
            if hit is None:
                continue
            x, t, u = hit
            La, Lc = float(np.hypot(*(b - a))), float(np.hypot(*(d - c)))
            if eps / La < t < 1 - eps / La and eps / Lc < u < 1 - eps / Lc:
                cand.append((3, x))
    circ = [p for p in prims if p["type"] == "circle"]
    tan_rel = float(th.value("tangency_rel"))
    for ci in circ:
        c, r = ci["g"]["c"], ci["g"]["r"]
        for s in lin:
            a, b = s["g"]["p"], s["g"]["q"]
            dv = b - a
            A = float(dv @ dv)
            if A == 0:
                continue
            # tangency contact point, a candidate point exactly like a crossing: the foot of the
            # perpendicular from the centre when |dist - r| <= tangency_rel*r and the foot lies on the
            # segment's extent within eps_on
            L = math.sqrt(A)
            tf = float((c - a) @ dv) / A
            foot = a + tf * dv
            if abs(float(np.hypot(*(c - foot))) - r) <= tan_rel * r and -eps / L <= tf <= 1 + eps / L:
                cand.append((3, foot))
            B = 2 * float(dv @ (a - c))
            C = float((a - c) @ (a - c)) - r * r
            disc = B * B - 4 * A * C
            if disc <= 0:
                continue
            for sgn in (-1, 1):
                t = (-B + sgn * math.sqrt(disc)) / (2 * A)
                if 0 <= t <= 1:
                    cand.append((3, a + t * dv))
    for i in range(len(circ)):
        for j in range(i + 1, len(circ)):
            c1, r1, c2, r2 = circ[i]["g"]["c"], circ[i]["g"]["r"], circ[j]["g"]["c"], circ[j]["g"]["r"]
            dd = float(np.hypot(*(c2 - c1)))
            if dd > 0 and abs(dd - (r1 + r2)) <= tan_rel * min(r1, r2):          # external tangency
                cand.append((3, c1 + r1 * (c2 - c1) / dd))
            elif dd > 0 and abs(dd - abs(r1 - r2)) <= tan_rel * min(r1, r2):     # internal tangency
                cand.append((3, c1 + r1 * (c2 - c1) / dd if r1 >= r2 else c2 + r2 * (c1 - c2) / dd))
            if dd == 0 or dd > r1 + r2 or dd < abs(r1 - r2):
                continue
            a_ = (r1 * r1 - r2 * r2 + dd * dd) / (2 * dd)
            h = math.sqrt(max(r1 * r1 - a_ * a_, 0.0))
            mid = c1 + a_ * (c2 - c1) / dd
            perp = np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / dd
            cand += [(3, mid + h * perp), (3, mid - h * perp)]
    # merge: drawn markers first (always entities), then internal candidates by priority
    merged: list[tuple[np.ndarray, bool]] = []          # (position, is_marker)
    ordered = [(0, m) for m in markers] + cand
    for pri, pt in sorted(ordered, key=lambda t: (t[0], round(float(t[1][0]), 3), round(float(t[1][1]), 3))):
        if all(float(np.hypot(*(pt - q))) > eps for q, _m in merged):
            merged.append((pt, pri == 0))
    # C28: an internal candidate becomes a point entity only when a name label promotes it
    promoted = [m for _pt, m in merged]
    straight_now = [p for p in prims if p["type"] in ("segment", "arrow", "axis")]
    for lab in labels:
        if lab["kind"] != "name":
            continue
        c = np.array([(lab["bbox"][0] + lab["bbox"][2]) / 2, (lab["bbox"][1] + lab["bbox"][3]) / 2])
        if lab["text"] in AXIS_NAMES and any(_seg_point_dist(c, t["g"]["p"], t["g"]["q"])[0] <= bind
                                              for t in straight_now):
            continue                                  # an axis label names a stroke, never a point
        if any(m and float(np.hypot(*(pt - c))) <= bind for pt, m in merged):
            continue                                  # a drawn dot within label_bind wins (cgx-1.3.0)
        best, bd = None, math.inf
        for k, (pt, m) in enumerate(merged):
            if m:
                continue
            dd = float(np.hypot(*(pt - c)))
            if dd < bd:
                best, bd = k, dd
        if best is not None and bd <= bind:
            promoted[best] = True
    points = [pt for (pt, _m), keep in zip(merged, promoted) if keep]
    for (pt, m), keep in zip(merged, promoted):
        if keep:
            prims.append({"type": "point", "g": {"x": pt[0], "y": pt[1], "marker": bool(m)}})
    # --- C30/C33 polygons (after point entities: rule (b) needs point-entity corners)
    strokes = [p for p in prims if p["type"] == "segment"]
    for vs in polygons_from_strokes([(s_["g"]["p"], s_["g"]["q"]) for s_ in strokes], eps, par_deg,
                                    points=points, closed_paths=closed_paths):
        prims.append({"type": "polygon", "g": {"vertices": vs}})
        for i in range(len(vs)):
            a, b = vs[i], vs[(i + 1) % len(vs)]
            same = [s_ for s_ in strokes
                    if (float(np.hypot(*(s_["g"]["p"] - a))) <= eps and float(np.hypot(*(s_["g"]["q"] - b))) <= eps)
                    or (float(np.hypot(*(s_["g"]["p"] - b))) <= eps and float(np.hypot(*(s_["g"]["q"] - a))) <= eps)]
            if same:
                same[0]["poly_edge"] = True
            else:
                seg = {"type": "segment", "g": {"p": a, "q": b}, "poly_edge": True}
                prims.append(seg)
                strokes.append(seg)
    # C28: type straight strokes with the ONE shared function (hmb.synthetic.geo_predicates)
    axis_texts = [{"text": lab["text"], "x": round((lab["bbox"][0] + lab["bbox"][2]) / 2, ROUND),
                   "y": round((lab["bbox"][1] + lab["bbox"][3]) / 2, ROUND)}
                  for lab in labels if lab["kind"] == "name" and lab["text"] in AXIS_NAMES]
    axis_bound: set[int] = set()
    if axis_texts:                                   # exactly Γ's test: bind_labels over all entities
        tmp = {"y_up": False, "texts": axis_texts,
               "entities": {f"e{k}": {"type": q["type"] if q["type"] != "curve_generic" else "curve",
                                      "label": None, "g": _jsonable(q["g"])}
                            for k, q in enumerate(prims)}}
        axis_bound = {int(eid[1:]) for _t, eid, _d in bind_labels(tmp, tolerances_10_2(th, D)) if eid}
    for st in [p for p in prims if p["type"] in ("segment", "arrow") and not p.get("poly_edge")]:
        a, b = st["g"]["p"], st["g"]["q"]
        axis_lab = any(prims[k] is st for k in axis_bound)
        typ, ga, gb = classify_straight(a, b, points, eps, tipped=bool(st.get("tipped")), D=D,
                                        parallel_deg=par_deg, axis_label=axis_lab,
                                        axis_min_frac=float(th.value("axis_min_length_frac")))
        st["type"] = typ
        st["g"] = {"p": np.array(ga), "q": np.array(gb)}
    # --- angle markers: small arcs centred on a vertex where >= 2 linear primitives meet
    lin = [p for p in prims if p["type"] in ("segment", "arrow", "axis", "ray", "line")]   # refreshed
    for p in prims:
        if p["type"] != "arc" or p["g"]["r"] > ANGLE_MARK_REL * D:
            continue
        c = p["g"]["c"]
        n = sum(1 for s in lin if _seg_point_dist(c, s["g"]["p"], s["g"]["q"])[0] <= eps)
        if n >= 2:
            p["type"] = "angle_marker"
            p["g"] = {"vertex": c, "r": p["g"]["r"], "p": p["g"]["p"], "q": p["g"]["q"]}
    # --- a multi-arc angle mark (double/triple arc) is ONE angle_marker: concentric marker arcs at the
    # same vertex (within eps_on) spanning the same angle (end directions within parallel_deg) merge,
    # keeping the innermost arc; "n" records the arc count, as in Γ (cgx-1.2.0)
    am = sorted([p for p in prims if p["type"] == "angle_marker"], key=lambda p: p["g"]["r"])
    kept: list[dict] = []
    for p in am:
        g = p["g"]
        du, dv = g["p"] - g["vertex"], g["q"] - g["vertex"]
        twin = next((k for k in kept if float(np.hypot(*(k["g"]["vertex"] - g["vertex"]))) <= eps
                     and _angle_deg(k["g"]["p"] - k["g"]["vertex"], du) <= par_deg
                     and _angle_deg(k["g"]["q"] - k["g"]["vertex"], dv) <= par_deg), None)
        if twin is None:
            g["n"] = 1
            kept.append(p)
        else:
            twin["g"]["n"] += 1
    prims = [p for p in prims if p["type"] != "angle_marker"] + kept
    # --- canonical end order of undirected primitives (segment, line, length_tick)
    for p in prims:
        if p["type"] in ("segment", "line", "length_tick"):
            a, b = p["g"]["p"], p["g"]["q"]
            if (round(float(b[0]), ROUND), round(float(b[1]), ROUND)) < (round(float(a[0]), ROUND), round(float(a[1]), ROUND)):
                p["g"]["p"], p["g"]["q"] = b, a
    # --- ids and shared-format geometry
    def sort_key(p):
        g = p["g"]
        anchor = (g.get("x"), g.get("y")) if p["type"] == "point" else tuple(
            _r(g.get("p") if "p" in g else g.get("c") if "c" in g else g.get("vertex") if "vertex" in g
               else (g.get("vertices") or g.get("points"))[0]))
        second = tuple(_r(g["q"])) if "q" in g else (0.0, 0.0)     # tie-break shared anchors by the other end
        return (_TYPE_ORDER.index(p["type"]), round(float(anchor[0]), 1), round(float(anchor[1]), 1),
                round(float(second[0]), 1), round(float(second[1]), 1), sha256_json(_jsonable(g)))
    prims.sort(key=sort_key)
    ents: dict[str, dict] = {}
    counters: dict[str, int] = {}
    for p in prims:
        counters[p["type"]] = counters.get(p["type"], 0) + 1
        eid = f"{_PREFIX[p['type']]}{counters[p['type']]}"
        ents[eid] = {"type": p["type"], "label": None, "g": _jsonable(p["g"])}
    texts = [{"text": lab["text"], "x": round((lab["bbox"][0] + lab["bbox"][2]) / 2, ROUND),
              "y": round((lab["bbox"][1] + lab["bbox"][3]) / 2, ROUND)} for lab in labels if lab["kind"] == "name"]
    return {"y_up": False, "entities": ents, "texts": texts}, glyphs, labels, D, path_ops


def _jsonable(g, nd: int = ROUND):
    if isinstance(g, dict):
        return {k: _jsonable(v, nd) for k, v in sorted(g.items())}
    if isinstance(g, np.ndarray):
        return _jsonable(g.tolist(), nd)
    if isinstance(g, (list, tuple)):
        return [_jsonable(v, nd) for v in g]
    if isinstance(g, (float, np.floating)):
        return round(float(g), nd)
    if isinstance(g, (np.integer,)):
        return int(g)
    return g


def _args_json(args) -> list:
    return [list(a) if isinstance(a, tuple) else a for a in args]


# ----------------------------------------------------------------------------- entry point
def extract_cgx(pdf_path: Path, *, item_id: str, model_id: str, track, th) -> CgxRecord:
    import fitz
    pdf_path = Path(pdf_path)
    data = pdf_path.read_bytes()
    with fitz.open(stream=data, filetype="pdf") as doc:
        page = doc[0]
        geom, glyphs, labels, D, path_ops = _build(page, th)
    over_cap = None
    if len(geom["entities"]) > int(th.value("cgx_max_primitives")):      # C43: no extractor evidence, all facts to Q1
        over_cap = len(geom["entities"])
        geom = {**geom, "entities": {}}
    facts_raw = evaluate_geo(geom, tolerances_10_2(th, D)) if geom["entities"] else []
    margins = {(p, a): m for p, a, m in facts_raw}
    types = {eid: e["type"] for eid, e in geom["entities"].items()}
    keys = [(p, a) for p, a, _m in facts_raw if p in PREDICATES]
    mgs = minimal_generating_set(keys, types) if keys else []
    facts = []
    for k in mgs:
        pred, args = k
        margin = margins.get(k)
        cid = "cf_" + sha256_json([pred, _args_json(args)])[:12]
        facts.append(CgxFact(cfact_id=cid, predicate=pred, args=_args_json(args),
                             margin=None if margin is None else round(float(margin), ROUND)))
    # the shared §10.3 rule, one entry per name-label instance in input order (geom["texts"] holds
    # exactly the name labels, in label order)
    binds = iter(bind_labels(geom, tolerances_10_2(th, D)) if geom["entities"] and geom["texts"] else [])
    lab_out, best_label = [], {}
    for lab in labels:
        bound_prim = None
        if lab["kind"] == "name" and geom["entities"]:
            _t, bound_prim, dist = next(binds)
            if bound_prim is not None:
                prev = best_label.get(bound_prim)
                if prev is None or (dist, lab["text"]) < prev:
                    best_label[bound_prim] = (dist, lab["text"])
        lab_out.append({"text": lab["text"], "bbox": [round(v, ROUND) for v in lab["bbox"]],
                        "kind": lab["kind"], "bound_prim": bound_prim})
    prims = [CgxPrimitive(prim_id=eid, type=e["type"], geom=e["g"],
                          label=best_label[eid][1] if eid in best_label else None)
             for eid, e in geom["entities"].items()]
    rec = {"extractor_version": f"{EXTRACTOR_VERSION}+{GEO_FORMAT_VERSION}", "item_id": item_id,
           "model_id": model_id, "track": Track(track).value, "pdf_sha256": sha256_bytes(data),
           "diagonal_D": round(D, ROUND), "primitives": [p.model_dump() for p in prims],
           "glyphs": glyphs, "labels": lab_out, "facts": [f.model_dump() for f in facts],
           "path_op_count": path_ops, "primitives_over_cap": over_cap}
    return validate_record(CgxRecord, rec)


def cgx_geometry(rec: CgxRecord) -> dict:
    """Rebuild the shared evaluator geometry from a stored record (for holds_geo recall queries)."""
    return {"y_up": False,
            "entities": {p.prim_id: {"type": p.type, "label": p.label, "g": p.geom} for p in rec.primitives},
            "texts": [{"text": lab["text"], "x": (lab["bbox"][0] + lab["bbox"][2]) / 2,
                       "y": (lab["bbox"][1] + lab["bbox"][3]) / 2} for lab in rec.labels if lab.get("kind") == "name"]}


def write_cgx(rec: CgxRecord, layout, model_id: str, track, item_id: str) -> tuple[Path, str]:
    """Write the CGX record immutably to layout.cgx(...) as canonical JSON; returns (path, sha256)."""
    from hmb.store import write_json_immutable
    from hmb.submission.process import model_dir_name
    path = layout.cgx(model_dir_name(model_id), Track(track).value, item_id)
    return path, write_json_immutable(path, rec.model_dump(mode="json"), canonical=True)
