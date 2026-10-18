"""§9.7 canonicalization (host half): determinism, geometry of s/letterbox, clean PNG chunks."""
import io
import math

import numpy as np
from PIL import Image, PngImagePlugin

from hmb.hashing import sha256_bytes
from hmb.submission import canonicalize_image
from hmb.submission.render import ink_fraction, png_chunk_types


def _png(img, **kw) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG", **kw)
    return b.getvalue()


def test_deterministic_and_hash_recorded(th):
    rng = np.random.default_rng(3)
    img = Image.fromarray(rng.integers(0, 255, (300, 500, 3), dtype=np.uint8))
    a, ma = canonicalize_image(_png(img), th)
    b, mb = canonicalize_image(img, th)
    assert a == b and ma == mb and ma["render_sha256"] == sha256_bytes(a)


def test_scale_rule_and_letterbox(th):
    img = Image.new("RGB", (2048, 1024), (0, 0, 0))
    png, m = canonicalize_image(img, th)
    assert m["s"] == 0.5 and (m["w"], m["h"]) == (1024, 512) and m["offset"] == [0, 256]
    out = np.asarray(Image.open(io.BytesIO(png)))
    assert out.shape == (1024, 1024, 3)
    assert (out[:256] == 255).all() and (out[768:] == 255).all() and (out[300:700] == 0).all()
    # upscale capped at 4x
    png, m = canonicalize_image(Image.new("RGB", (100, 50), (0, 0, 0)), th)
    assert m["s"] == 4.0 and (m["w"], m["h"]) == (400, 200) and m["offset"] == [312, 412]
    png, m = canonicalize_image(Image.new("RGB", (512, 300), (0, 0, 0)), th)
    assert m["s"] == 2.0 and (m["W"], m["H"]) == (512, 300)


def test_alpha_composited_on_white(th):
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    png, _ = canonicalize_image(img, th)
    assert (np.asarray(Image.open(io.BytesIO(png))) == 255).all()
    img = Image.new("LA", (64, 64), (0, 128))
    arr = np.asarray(Image.open(io.BytesIO(canonicalize_image(img, th)[0])))
    assert abs(int(arr[512, 512, 0]) - 127) <= 1


def test_no_time_or_metadata_chunks(th):
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "x")
    img = Image.new("RGB", (40, 30), (10, 20, 30))
    src = _png(img, pnginfo=info, dpi=(300, 300))
    assert "tEXt" in png_chunk_types(src) and "pHYs" in png_chunk_types(src)
    png, _ = canonicalize_image(src, th)
    assert png_chunk_types(png) == ["IHDR", "IDAT", "IEND"]


def test_grayscale_and_16bit_inputs(th):
    g = Image.new("L", (100, 100), 0)
    png, m = canonicalize_image(g, th)
    assert Image.open(io.BytesIO(png)).mode == "RGB"
    g16 = Image.fromarray(np.full((50, 80), 65535, dtype=np.uint16))
    png, _ = canonicalize_image(g16, th)
    assert (np.asarray(Image.open(io.BytesIO(png))) == 255).all()


def test_ink_fraction(th):
    blank, _ = canonicalize_image(Image.new("RGB", (100, 100), (255, 255, 255)), th)
    assert ink_fraction(blank) == 0.0
    img = Image.new("RGB", (1024, 1024), (255, 255, 255))
    img.paste((0, 0, 0), (0, 0, 32, 32))
    png, _ = canonicalize_image(img, th)
    assert math.isclose(ink_fraction(png), 32 * 32 / 1024 ** 2)


def test_same_code_path_for_crops(th, tmp_path):
    p = tmp_path / "crop.png"
    Image.new("L", (700, 350), 200).save(p)
    a, _ = canonicalize_image(p, th)
    b, _ = canonicalize_image(p.read_bytes(), th)
    assert a == b
