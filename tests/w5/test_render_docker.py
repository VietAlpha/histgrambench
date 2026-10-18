"""Track C candidates rendered through the PRODUCTION harness (process_submission, track C) in the
pinned sandbox image. Marked docker; skipped when the image is unavailable (never a host TeX fallback)."""
import pytest

from hmb.synthetic.families import generate
from hmb.synthetic.render import render_candidate
from hmb.synthetic.tikz import emit_tikz, picture_sha256, wrap
from hmb.store import Layout

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module", autouse=True)
def _need_sandbox():
    from hmb.submission.sandbox import sandbox_available
    if not sandbox_available():
        pytest.skip("pinned sandbox image unavailable")


@pytest.mark.parametrize("family,seed,kind", [("triangles_polygons", 0, "prose_fenced"),
                                              ("simple_solids", 1, "bare_body")])
def test_control_renders_ok_with_body_invariant(family, seed, kind, tmp_path, th):
    s = generate(family, seed)
    tikz = emit_tikz(s)
    raw, stop = wrap(tikz, kind)
    r = render_candidate(raw, stop, candidate_id=f"{s.spec_id}__c00", layout=Layout(tmp_path), th=th)
    assert r["status"] == "OK", r
    rec = r["record"]
    assert rec["track"] == "C" and rec["body_sha256"] == picture_sha256(tikz)
    assert rec["render_sha256"] and rec["page_count"] == 1


def test_dpi_transform_reports_blocked(tmp_path, th):
    r = render_candidate("x", None, candidate_id="c", layout=Layout(tmp_path), th=th, dpi_delta=25)
    assert r["status"] == "BLOCKED"
