"""§10.2 CGX on PDFs compiled in the pinned sandbox from TikZ whose facts are known by construction.
Expected facts are derived by hand from the coordinates below (not by the extractor). §9.8 smoke:
glyph strings and per-glyph boxes for Latin, Greek, Cyrillic and CJK labels."""
import math
from pathlib import Path

import numpy as np
import pytest

from hmb.cards.closure import entails, forward_chain
from hmb.cgx import extract_cgx
from hmb.hashing import canonical_json
from hmb.submission import compile_in_sandbox, extract_candidate, normalize
from w3_helpers import tikz

pytestmark = pytest.mark.docker
CM = 72.0 / 2.54


def build(th, tmp_path, body, name="c"):
    nz = normalize(extract_candidate(tikz(body), "end_turn"), th)
    r = compile_in_sandbox(nz.tex_bytes, th, tmp_path / name)
    assert r.outcome.value == "OK", r.to_json()
    rec = extract_cgx(Path(r.pdf_path), item_id="hmb_00000000000000c0", model_id="m", track="C", th=th)
    return rec, Path(r.pdf_path)


class View:
    def __init__(self, rec):
        self.rec = rec
        self.P = {p.prim_id: p for p in rec.primitives}
        self.keys = {(f.predicate, tuple(tuple(a) if isinstance(a, list) else a for a in f.args)) for f in rec.facts}
        types = {p.prim_id: p.type for p in rec.primitives}
        self.closed = forward_chain(self.keys, types)
        self.by_label = {lab["text"]: lab["bound_prim"] for lab in rec.labels if lab["bound_prim"]}

    def pt(self, label):
        return self.by_label[label]

    def xy(self, pid):
        g = self.P[pid].geom
        return np.array([g["x"], g["y"]])

    def linear(self, a, b, types=("segment", "arrow", "axis")):
        A, B = self.xy(a), self.xy(b)
        for p in self.rec.primitives:
            if p.type in types:
                u, v = np.array(p.geom["p"]), np.array(p.geom["q"])
                if (np.allclose(u, A, atol=0.5) and np.allclose(v, B, atol=0.5)) or \
                        (np.allclose(u, B, atol=0.5) and np.allclose(v, A, atol=0.5)):
                    return p.prim_id
        raise AssertionError(f"no linear primitive {a}-{b}")

    def of_type(self, t):
        return [p.prim_id for p in self.rec.primitives if p.type == t]

    def holds(self, pred, *args):
        from hmb.ontology import fact_key
        return entails(self.closed, fact_key(pred, list(args)))


TRI = r"""
\coordinate (O) at (0,0); \coordinate (A) at (0,2);
\coordinate (B) at (-1.7320508,-1); \coordinate (C) at (1.7320508,-1);
\draw (O) circle (2);
\draw (A) -- (B) -- (C) -- cycle;
\draw (-3,2) -- (3,2);
\fill (O) circle (1.5pt);
\node[above left] at (A) {A}; \node[below left] at (B) {B};
\node[below right] at (C) {C}; \node[below] at (O) {O};
"""


def test_triangle_circumcircle_tangent(th, tmp_path):
    rec, _ = build(th, tmp_path, TRI)
    v = View(rec)
    A, B, C, O = (v.pt(x) for x in "ABCO")
    (circ,) = v.of_type("circle")
    (poly,) = v.of_type("polygon")
    # C28: the tangent stroke has point entities at neither end (A is interior) -> line
    tang = next(p.prim_id for p in rec.primitives if p.type == "line" and
                abs(p.geom["p"][1] - p.geom["q"][1]) < 1e-6 and
                abs(abs(p.geom["p"][0] - p.geom["q"][0]) - 6 * CM) < 0.5)
    BC = v.linear(B, C)
    # hand-derived: circumcircle through A, B, C centred at O, radius 2 cm
    assert abs(v.P[circ].geom["r"] - 2 * CM) < 0.05
    assert v.holds("center_of", O, circ)
    for X in (A, B, C):
        assert v.holds("on", X, circ)
    # tangent at A (y = 2 touches the circle x^2+y^2=4 only at A) and parallel to BC (both horizontal)
    assert v.holds("tangent_to", tang, circ, A)
    assert v.holds("parallel", tang, BC)
    assert v.holds("polygon", poly, 3)
    for X, Y in ((A, B), (B, C), (C, A)):
        s = v.linear(X, Y)
        assert v.holds("edge_of", s, poly) and v.holds("endpoint_of", X, s) and v.holds("endpoint_of", Y, s)
    for lab, pid in (("A", A), ("B", B), ("C", C), ("O", O)):
        assert v.holds("label", lab, pid)
    assert v.holds("inside", O, poly)                       # centroid of an equilateral triangle
    assert not v.holds("perpendicular", tang, BC)
    for f in rec.facts:                                     # margins in eps units, >= 0
        assert f.margin is None or f.margin >= 0


CROSS = r"""
\draw (0,0) -- (4,4); \draw (0,4) -- (4,0);
\draw (2.2121,2.2121) -- (2.4243,2.0) -- (2.2121,1.7879);
\node[left] at (0,0) {P}; \node[right] at (4,4) {Q};
\node[left] at (0,4) {R}; \node[right] at (4,0) {S};
\node[above=2pt] at (2,2) {E};
\draw (6,0) -- (6,4); \draw (7,0) -- (7,4);
\draw (6,1) -- (6.8,1.5);
\draw (6.9,2) -- (7.1,2);
"""


def test_crossing_perpendicular_right_angle_parallel_tick(th, tmp_path):
    rec, _ = build(th, tmp_path, CROSS)
    v = View(rec)
    Pp, Q, R, S, E = (v.pt(x) for x in "PQRSE")
    PQ, RS = v.linear(Pp, Q), v.linear(R, S)
    # hand-derived: y=x and y=4-x cross at E=(2,2), transversally, at 90 degrees
    assert np.allclose(v.xy(E), [v.xy(Pp)[0] + 2 * CM, v.xy(Pp)[1] - 2 * CM], atol=0.5)
    assert v.holds("intersect_at", PQ, RS, E)
    assert v.holds("perpendicular", PQ, RS)
    (ra,) = v.of_type("right_angle_marked")
    assert np.allclose(v.P[ra].geom["vertex"], v.xy(E), atol=0.5)
    # a 0.2 cm horizontal stroke crossing x=7 at (7,2) is a length tick, not a segment
    (tk,) = v.of_type("length_tick")
    assert np.allclose(np.mean([v.P[tk].geom["p"], v.P[tk].geom["q"]], axis=0),
                       v.xy(Pp) + np.array([7 * CM, -2 * CM]), atol=0.5)
    # x=6 and x=7 are parallel verticals
    verts = [p for p in rec.primitives if p.type == "segment" and abs(p.geom["p"][0] - p.geom["q"][0]) < 1e-6
             and abs(abs(p.geom["p"][1] - p.geom["q"][1]) - 4 * CM) < 0.5]
    assert len(verts) == 2 and v.holds("parallel", verts[0].prim_id, verts[1].prim_id)
    assert v.holds("orientation", verts[0].prim_id, "vertical")
    assert not v.holds("parallel", PQ, RS)


AXES = r"""
\draw[->] (0,0) -- (6,0); \draw[->] (0,0) -- (0,4);
\draw[-stealth] (3,2) -- (4,3);
\draw (1,0) -- (1,3);
\draw (0,0) -- (2,2);
\draw (0.6,0) arc (0:45:0.6);
\node[below left] at (0,0) {O};
"""


def test_axes_arrow_angle_marker(th, tmp_path):
    rec, _ = build(th, tmp_path, AXES)
    v = View(rec)
    axes = v.of_type("axis")
    assert len(axes) == 2
    os_ = sorted(next(f.args[1] for f in rec.facts if f.predicate == "axis" and f.args[0] == a) for a in axes)
    assert os_ == ["horizontal", "vertical"]
    (arrow,) = v.of_type("arrow")
    g = v.P[arrow].geom
    O = v.xy(v.pt("O"))
    assert np.allclose(np.array(g["q"]) - O, [4 * CM, -3 * CM], atol=1.5)      # head at (4,3), y down
    assert np.allclose(np.array(g["p"]) - O, [3 * CM, -2 * CM], atol=1.5)
    (am,) = v.of_type("angle_marker")
    assert np.allclose(v.P[am].geom["vertex"], O, atol=0.5)
    assert abs(v.P[am].geom["r"] - 0.6 * CM) < 0.2
    # x=1 is perpendicular to the horizontal axis
    vert = next(p.prim_id for p in rec.primitives if p.type == "segment" and abs(p.geom["p"][0] - p.geom["q"][0]) < 1e-6)
    hx = next(a for a in axes if abs(v.P[a].geom["p"][1] - v.P[a].geom["q"][1]) < 1e-6)
    assert v.holds("perpendicular", vert, hx)


def test_scale_invariance_and_bit_for_bit_determinism(th, tmp_path):
    r1, pdf1 = build(th, tmp_path, TRI, "s1")
    r1b = extract_cgx(pdf1, item_id="hmb_00000000000000c0", model_id="m", track="C", th=th)
    assert canonical_json(r1.model_dump(mode="json")) == canonical_json(r1b.model_dump(mode="json"))
    r2, _ = build(th, tmp_path, "\\begin{scope}[scale=2.5]" + TRI + "\\end{scope}", "s2")
    sig = lambda r: sorted((f.predicate, repr(f.args)) for f in r.facts)
    # tolerances are D-relative: uniform scaling (and the translation standalone applies) keeps the
    # same fact set; entity ids are assigned in geometric order, so they coincide too.
    assert sig(r1) == sig(r2)
    assert math.isclose(r2.diagonal_D / r1.diagonal_D, 2.5, rel_tol=0.08)


def test_rotated_ellipse_frame_and_containment(th, tmp_path):
    """Red-team #3: rot_deg is transformed exactly once. A 2x1 ellipse rotated +30 deg (TikZ, y up)
    contains a r=0.3 circle centred 1.4 along its major axis; with the rotation sign flipped the
    circle centre would sit 60 deg off-axis at normalized radius 1.26 > 1 (outside)."""
    body = (r"\draw[rotate=30] (0,0) ellipse (2 and 1);" "\n"
            r"\draw (1.2124356,0.7) circle (0.3);" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    (el,) = v.of_type("ellipse")
    (ci,) = v.of_type("circle")
    g = v.P[el].geom
    assert abs(g["rx"] - 2 * CM) < 0.3 and abs(g["ry"] - 1 * CM) < 0.3
    assert abs((g["rot_deg"] % 180) - 150.0) < 1.0        # +30 deg y-up == -30 deg (150) in the page frame
    assert v.holds("contains", el, ci)


def test_tangency_contact_point_is_a_candidate_point(th, tmp_path):
    """Red-team #4: the line y=2.02 is tangent to x^2+y^2=4 within tolerance (0.02 <= 0.015*2) but does
    not cross it, so only the contact-point rule creates the point the label T binds to."""
    body = (r"\draw (0,0) circle (2);" "\n" r"\draw (-3,2.02) -- (3,2.02);" "\n"
            r"\node[above] at (0,2.02) {T};" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    T = v.pt("T")
    assert v.P[T].type == "point"
    (circ,) = v.of_type("circle")
    (line,) = v.of_type("line")                            # C28: T is interior, ends unlabelled
    assert v.holds("tangent_to", line, circ, T)


def test_duplicate_and_numeral_labels_bind_per_instance(th, tmp_path):
    """Red-team #2 + #11: two "1"s and two "2"s each bind to their own endpoint; short numerals are
    names, "5 cm" is a measure and is not bound."""
    body = (r"\draw (0,0) -- (2,0); \draw (4,0) -- (6,0);" "\n"
            r"\node[below] at (0,0) {1}; \node[below] at (2,0) {2};" "\n"
            r"\node[below] at (4,0) {1}; \node[below] at (6,0) {2};" "\n"
            r"\node[above] at (1,0.3) {5 cm};" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    by_text: dict = {}
    for lab in rec.labels:
        by_text.setdefault(lab["text"], []).append(lab)
    assert [lab["kind"] for lab in by_text["5cm"]] == ["measure"] and by_text["5cm"][0]["bound_prim"] is None   # glyphs carry no space
    for t in ("1", "2"):
        labs = by_text[t]
        assert len(labs) == 2 and all(lab["kind"] == "name" for lab in labs)
        bound = [lab["bound_prim"] for lab in labs]
        assert len(set(bound)) == 2 and all(v.P[b].type == "point" for b in bound)
        for b in bound:
            assert v.P[b].label == t and v.holds("label", t, b)
    xs = sorted((v.xy(b)[0], lab["text"]) for t in ("1", "2") for lab in by_text[t] for b in [lab["bound_prim"]])
    assert [t for _x, t in xs] == ["1", "2", "1", "2"]      # each instance bound to its own endpoint


def test_c28_unlabelled_geometry_is_internal(th, tmp_path):
    """C28 (1): an unlabelled, unmarked triangle has NO point entities; its edges (no incident point
    entities at all) are segments; a drawn dot is a point even without a label; crossings stay
    internal (no intersect_at without a point entity)."""
    body = (r"\draw (0,0) -- (4,0) -- (2,3) -- cycle;" "\n"
            r"\draw (1,-1) -- (3,4);" "\n"                  # crosses two edges, no label
            r"\fill (6,1) circle (2pt);" "\n")              # a drawn dot, unlabelled
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    pts = v.of_type("point")
    assert len(pts) == 1                                     # only the drawn dot
    assert np.allclose(v.xy(pts[0]) - np.array([v.P[v.of_type("polygon")[0]].geom["vertices"][0]]),
                       [[6 * CM, -1 * CM]], atol=1.0)
    assert sorted(p.type for p in rec.primitives if p.type != "point") == \
        ["polygon", "segment", "segment", "segment", "segment"]
    for f in rec.facts:
        from hmb.ontology import entity_args
        assert all(v.P[a].type != "point" or a == pts[0] for a in entity_args(f.predicate, f.args))
    assert not any(f.predicate in ("intersect_at", "between", "collinear", "concyclic") for f in rec.facts)


def test_c28_segment_ray_line_typing(th, tmp_path):
    """C28 (2): both ends labelled -> segment; one end labelled, stroke continuing -> ray (origin at
    the labelled end); labelled points only inside the stroke -> line. Interior points are drawn as
    dots: a bare mid-stroke label has no candidate location and binds to the stroke itself (§10.3)."""
    body = (r"\draw (0,0) -- (3,0); \node[below] at (0,0) {A}; \node[below] at (3,0) {B};" "\n"
            r"\draw (0,2) -- (4,2); \node[below] at (0,2) {C}; \fill (2,2) circle (1.5pt) node[below] {M};" "\n"
            r"\draw (0,4) -- (5,4); \fill (1.5,4) circle (1.5pt) node[below] {P};"
            r" \fill (3.5,4) circle (1.5pt) node[below] {Q};" "\n"
            r"\draw (6,0) -- (8,0); \node[above] at (7,0) {k};" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    A, B, C, M, Pp, Q = (v.pt(x) for x in ("A", "B", "C", "M", "P", "Q"))
    seg = v.linear(A, B)
    assert v.P[seg].type == "segment"
    k = next(lab for lab in rec.labels if lab["text"] == "k")          # bare mid-stroke label
    assert v.P[k["bound_prim"]].type == "segment" and v.holds("label", "k", k["bound_prim"])
    (ray,) = v.of_type("ray")
    (line,) = v.of_type("line")
    assert v.holds("endpoint_of", A, seg) and v.holds("endpoint_of", B, seg)
    assert np.allclose(v.P[ray].geom["p"], v.xy(C), atol=0.5)            # origin at C
    assert v.holds("on", M, ray) and v.holds("on", C, ray)
    assert v.holds("on", Pp, line) and v.holds("on", Q, line)
    assert v.holds("orientation", line, "horizontal") and v.holds("orientation", ray, "horizontal")


def test_c28_axes_from_labelled_origin(th, tmp_path):
    """Tipped horizontal/vertical strokes from a labelled origin O are axes (the base type is ray;
    see the extractor docstring); a short oblique tipped stroke is an arrow."""
    body = (r"\draw[->] (0,0) -- (5,0) node[right] {$x$};" "\n" r"\draw[->] (0,0) -- (0,3) node[above] {$y$};" "\n"
            r"\node[below left] at (0,0) {O};" "\n" r"\draw[->] (1,1) -- (2,2);" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    assert len(v.of_type("axis")) == 2 and len(v.of_type("arrow")) == 1
    assert v.of_type("point") == [v.pt("O")]


def test_double_arc_is_one_angle_marker_and_ellipse_rot_canonical(th, tmp_path):
    """A double-arc angle mark (radii 0.35 and 0.43 at vertex B, same span) is ONE angle_marker with
    n=2; an axis-aligned ellipse stores rot_deg in [0, 180) (0, not 180)."""
    body = (r"\draw (0,0) -- (3,0); \draw (0,0) -- (1.5,2.598);" "\n"
            r"\draw (0.35,0) arc[start angle=0, end angle=60, radius=0.35];" "\n"
            r"\draw (0.43,0) arc[start angle=0, end angle=60, radius=0.43];" "\n"
            r"\node[below left] at (0,0) {B};" "\n"
            r"\draw (6,1) ellipse[x radius=1.5, y radius=0.8];" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    (am,) = v.of_type("angle_marker")
    assert v.P[am].geom["n"] == 2 and abs(v.P[am].geom["r"] - 0.35 * CM) < 0.2
    assert np.allclose(v.P[am].geom["vertex"], v.xy(v.pt("B")), atol=0.5)
    (el,) = v.of_type("ellipse")
    rot = v.P[el].geom["rot_deg"]
    assert 0.0 <= rot < 180.0 and min(rot, 180.0 - rot) < 0.5


def test_label_binds_to_drawn_dot_before_promoting(th, tmp_path):
    """cgx-1.3.0 rule (bug from circles DEV seed 9000003): "O" sits 5.0 pt from the drawn dot at the
    big circle's centre and 4.9 pt from the small circle's (undotted) centre. The dot wins: O is the
    big circle's centre and the endpoint of OT; the small centre stays internal."""
    body = (r"\draw (0,0) circle (2);" "\n" r"\fill (0,0) circle (1.5pt);" "\n"
            r"\draw (0.35,0) circle (0.5);" "\n"
            r"\node at (0.1764,0) {O};" "\n"
            r"\draw (0,0) -- (0,2); \node[above] at (0,2) {T};" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    O, T = v.pt("O"), v.pt("T")
    assert v.P[O].geom.get("marker") is True
    big = next(c for c in v.of_type("circle") if abs(v.P[c].geom["r"] - 2 * CM) < 0.1)
    small = next(c for c in v.of_type("circle") if abs(v.P[c].geom["r"] - 0.5 * CM) < 0.1)
    assert v.holds("center_of", O, big) and not v.holds("center_of", O, small)
    OT = v.linear(O, T)
    assert v.holds("endpoint_of", O, OT) and v.holds("endpoint_of", T, OT)
    assert len(v.of_type("point")) == 2                     # the small centre is not promoted


def _sig(rec):
    return sorted((f.predicate, repr(f.args)) for f in rec.facts), \
        sorted((p.prim_id, p.type) for p in rec.primitives)


def test_polygon_style_invariance_closed_path_vs_separate_strokes(th, tmp_path):
    """C30 / §7.6 source_refactor: `\\draw (A)--(B)--(C)--cycle` and three separate `\\draw`
    segments give IDENTICAL CGX facts and primitives."""
    lab = (r"\node[above] at (4,6) {A}; \node[below left] at (0,0) {B}; \node[below right] at (8,0) {C};"
           r"\fill (4,6) circle (1.2pt); \fill (0,0) circle (1.2pt); \fill (8,0) circle (1.2pt);")
    r1, _ = build(th, tmp_path, r"\draw (4,6) -- (0,0) -- (8,0) -- cycle;" + "\n" + lab, "closed")
    r2, _ = build(th, tmp_path, r"\draw (4,6) -- (0,0); \draw (0,0) -- (8,0); \draw (8,0) -- (4,6);" + "\n" + lab,
                  "separate")
    assert _sig(r1) == _sig(r2)
    v = View(r1)
    (poly,) = v.of_type("polygon")
    A, B, C = v.pt("A"), v.pt("B"), v.pt("C")
    for X, Y in ((A, B), (B, C), (C, A)):
        s = v.linear(X, Y)
        assert v.P[s].type == "segment" and v.holds("edge_of", s, poly)


def test_polygon_edges_split_by_a_midpoint(th, tmp_path):
    """Generator style: triangle ABC plus B-M, M-C, A-M drawn separately (M the midpoint of BC).
    Polygons ABC, ABM, AMC; every polygon edge is a segment entity with edge_of, including a
    synthesized B-C when BC is only drawn as B-M + M-C."""
    body = (r"\draw (4,6) -- (0,0); \draw (0,0) -- (4,0); \draw (4,0) -- (8,0); \draw (8,0) -- (4,6);"
            r" \draw (4,6) -- (4,0);" "\n"
            r"\node[above] at (4,6) {A}; \node[below left] at (0,0) {B}; \node[below right] at (8,0) {C};"
            r" \node[below] at (4,0) {M};" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    A, B, C, M = (v.pt(x) for x in "ABCM")
    polys = v.of_type("polygon")
    assert len(polys) == 3
    vsets = sorted(sorted(map(tuple, np.round(v.P[g].geom["vertices"], 1).tolist())) for g in polys)
    want = sorted(sorted(map(tuple, np.round([v.xy(X) for X in t], 1).tolist())) for t in ((A, B, C), (A, B, M), (A, M, C)))
    assert vsets == want
    BC = v.linear(B, C)                                     # synthesized edge
    tri = next(g for g in polys if len(v.P[g].geom["vertices"]) == 3 and
               sorted(map(tuple, np.round(v.P[g].geom["vertices"], 1).tolist())) == want[[i for i, w in enumerate(want)
               if sorted(map(tuple, np.round([v.xy(X) for X in (A, B, C)], 1).tolist())) == w][0]])
    assert v.holds("edge_of", BC, tri) and v.P[BC].type == "segment"


def test_square_with_diagonals_documented_case(th, tmp_path):
    """C34 documented case, through real compiles: a labelled square with both diagonals has ONLY the
    4 corner-sharing triangles as polygons (the diagonals are chords), in BOTH drawing styles, and the
    two styles give identical CGX facts."""
    lab = (r"\node[below left] at (0,0) {A}; \node[below right] at (5,0) {B};"
           r" \node[above right] at (5,5) {C}; \node[above left] at (0,5) {D};")
    diag = r"\draw (0,0) -- (5,5); \draw (5,0) -- (0,5);"
    sep, _ = build(th, tmp_path, r"\draw (0,0) -- (5,0); \draw (5,0) -- (5,5); \draw (5,5) -- (0,5);"
                   r" \draw (0,5) -- (0,0);" + diag + "\n" + lab, "sep")
    cyc, _ = build(th, tmp_path, r"\draw (0,0) -- (5,0) -- (5,5) -- (0,5) -- cycle;" + diag + "\n" + lab, "cyc")
    ns = sorted(len(View(sep).P[g].geom["vertices"]) for g in View(sep).of_type("polygon"))
    nc = sorted(len(View(cyc).P[g].geom["vertices"]) for g in View(cyc).of_type("polygon"))
    assert ns == [3, 3, 3, 3] and nc == [3, 3, 3, 3]
    assert _sig(sep) == _sig(cyc)


def test_unlabelled_triangle_style_invariance(th, tmp_path):
    """C34: an UNLABELLED triangle drawn as one closed path and as 3 separate strokes gives identical
    CGX facts and primitives (one polygon, no point entities)."""
    r1, _ = build(th, tmp_path, r"\draw (4,6) -- (0,0) -- (8,0) -- cycle;", "closed")
    r2, _ = build(th, tmp_path, r"\draw (4,6) -- (0,0); \draw (0,0) -- (8,0); \draw (8,0) -- (4,6);", "separate")
    assert _sig(r1) == _sig(r2)
    v = View(r1)
    assert len(v.of_type("polygon")) == 1 and v.of_type("point") == []


def test_reversed_strokes_give_byte_identical_cgx(th, tmp_path):
    """Red-team #6: drawing `B -- A` instead of `A -- B` (and reversed ticks) gives byte-identical
    CGX apart from pdf_sha256 (the content streams differ)."""
    lab = r"\node[below left] at (0,0) {A}; \node[below right] at (5,1) {B}; \node[above] at (2,4) {C};"
    fwd = (r"\draw (0,0) -- (5,1); \draw (0,0) -- (2,4); \draw (2.4,0.2) -- (2.6,0.8);"
           r" \draw (-1,5) -- (6,5);" "\n" + lab)
    rev = (r"\draw (5,1) -- (0,0); \draw (2,4) -- (0,0); \draw (2.6,0.8) -- (2.4,0.2);"
           r" \draw (6,5) -- (-1,5);" "\n" + lab)
    r1, _ = build(th, tmp_path, fwd, "fwd")
    r2, _ = build(th, tmp_path, rev, "rev")
    d1, d2 = r1.model_dump(mode="json"), r2.model_dump(mode="json")
    assert d1.pop("pdf_sha256") != d2.pop("pdf_sha256")
    assert canonical_json(d1) == canonical_json(d2)


def test_axis_label_binding_matches_gamma_and_markers_explicit(th, tmp_path):
    """Red-team #1/#2/#4: an axis-name label placed MID-stroke binds to the untipped stroke via
    bind_labels (as in Γ) and makes it an axis; it never promotes a point; every point primitive
    carries an explicit boolean marker (drawn dot true, promoted candidate false)."""
    body = (r"\draw (0,0) -- (6,0); \node[above] at (3,0) {$x$};" "\n"
            r"\draw (0,2) -- (6,2); \node[below left] at (0,2) {P}; \fill (6,2) circle (1.5pt) node[above] {Q};" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    (ax,) = v.of_type("axis")
    assert abs(v.P[ax].geom["p"][1] - v.P[ax].geom["q"][1]) < 1e-6
    assert next(lab for lab in rec.labels if lab["text"] == "x")["bound_prim"] == ax
    P_, Q = v.pt("P"), v.pt("Q")
    assert v.P[P_].geom["marker"] is False and v.P[Q].geom["marker"] is True
    assert all(isinstance(v.P[pt].geom.get("marker"), bool) for pt in v.of_type("point"))
    assert len(v.of_type("point")) == 2                     # the x label promoted nothing


def test_angle_marker_uses_refreshed_linear_list(th, tmp_path):
    """Red-team #5: a marker at C where CA and a split side (drawn B-M, M-C) meet, with the polygon
    edge BC synthesized, is an angle_marker at C (the linear list is refreshed after synthesis)."""
    body = (r"\draw (0,0) -- (4,0); \draw (4,0) -- (8,0); \draw (8,0) -- (4,6); \draw (4,6) -- (0,0);" "\n"
            r"\draw (7.6,0) arc[start angle=180, end angle=123.69, radius=0.4];" "\n"
            r"\node[below left] at (0,0) {B}; \node[below right] at (8,0) {C}; \node[above] at (4,6) {A};"
            r" \node[below] at (4,0) {M};" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    (am,) = v.of_type("angle_marker")
    assert np.allclose(v.P[am].geom["vertex"], v.xy(v.pt("C")), atol=0.5)
    assert v.P[v.linear(v.pt("B"), v.pt("C"))].type == "segment"   # synthesized polygon edge


def test_multi_ticks_and_short_marker_arcs(th, tmp_path):
    """cgx-1.3.1 (V1x similarity cells 9000010/9000002): a triple tick on one side is ONE length_tick
    (n=3) and never an arrow; a double tick is n=2; a short 21-degree marker arc (r=0.35) centred at a
    vertex is an angle_marker, not a tick."""
    body = (r"\draw (0,0) -- (6,0); \draw (0,0) -- (0,4);" "\n"
            r"\draw (2.92,-0.12) -- (2.92,0.12); \draw (3.0,-0.12) -- (3.0,0.12); \draw (3.08,-0.12) -- (3.08,0.12);" "\n"
            r"\draw (-0.12,1.96) -- (0.12,1.96); \draw (-0.12,2.04) -- (0.12,2.04);" "\n"
            r"\draw (6,0) -- (0,4);" "\n"
            r"\draw (5.65,0) arc[start angle=180, end angle=146.31, radius=0.35];" "\n"
            r"\node[below left] at (0,0) {A}; \node[below] at (6,0) {B}; \node[left] at (0,4) {C};" "\n")
    rec, _ = build(th, tmp_path, body)
    v = View(rec)
    assert v.of_type("arrow") == []
    ticks = sorted((v.P[t].geom["n"], round(v.P[t].geom["p"][0], 0)) for t in v.of_type("length_tick"))
    assert [n for n, _x in ticks] == [2, 3]
    tri = next(t for t in v.of_type("length_tick") if v.P[t].geom["n"] == 3)
    mid = np.mean([v.P[tri].geom["p"], v.P[tri].geom["q"]], axis=0)
    assert np.allclose(mid, v.xy(v.pt("A")) + np.array([3 * CM, 0.0]), atol=0.5)   # group centre at (3,0)
    (am,) = v.of_type("angle_marker")
    assert np.allclose(v.P[am].geom["vertex"], v.xy(v.pt("B")), atol=0.5)


def test_glyph_strings_and_boxes_multiscript(th, tmp_path):
    """§9.8 smoke: Latin, Greek, Cyrillic, CJK (+ kana, Vietnamese, math) extract with per-glyph boxes."""
    labels = ["A", "α", "Δ", "Ж", "Б", "ѣ", "甲", "中國", "圓", "イ", "ゐ", "Đ", "ệ", "$A_1$", "$\\alpha$", "$\\angle ABC$"]
    want = ["A", "α", "Δ", "Ж", "Б", "ѣ", "甲", "中國", "圓", "イ", "ゐ", "Đ", "ệ", "A1", "α", "∠ABC"]
    body = "\n".join(f"\\node at ({(i % 4) * 2.5},{-(i // 4) * 1.5}) {{{t}}};" for i, t in enumerate(labels))
    rec, pdf = build(th, tmp_path, body)
    got = [lab["text"] for lab in rec.labels]
    assert got == want
    assert len(rec.glyphs) == sum(len(w) for w in want)
    import fitz
    with fitz.open(pdf) as doc:
        W, H = doc[0].rect.width, doc[0].rect.height
    for g in rec.glyphs:
        x0, y0, x1, y1 = g["bbox"]
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
    # boxes sit where the nodes were placed: node i centre = origin + (2.5 cm * col, -1.5 cm * row)
    c0 = np.array([(rec.labels[0]["bbox"][0] + rec.labels[0]["bbox"][2]) / 2,
                   (rec.labels[0]["bbox"][1] + rec.labels[0]["bbox"][3]) / 2])
    for i, lab in enumerate(rec.labels):
        c = np.array([(lab["bbox"][0] + lab["bbox"][2]) / 2, (lab["bbox"][1] + lab["bbox"][3]) / 2])
        exp = c0 + np.array([(i % 4) * 2.5 * CM, (i // 4) * 1.5 * CM])
        assert np.allclose(c, exp, atol=4.0), (lab, exp)


def test_c43_primitive_bound_keeps_no_evidence(th, tmp_path):
    """C43: above cgx_max_primitives the record keeps no primitives or facts and states the count found."""
    import copy
    rec, _ = build(th, tmp_path, TRI, "under")
    n = len(rec.primitives)
    assert rec.primitives_over_cap is None and n > 3 and rec.facts
    th2 = copy.deepcopy(th)
    th2.params["cgx_max_primitives"]["value"] = 3
    over, _ = build(th2, tmp_path, TRI, "over")
    assert over.primitives_over_cap == n and over.primitives == [] and over.facts == []
