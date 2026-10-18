"""§9.7 render canonicalization, shared by candidate renders and historical crops (§6.4).

Rasterization of a candidate PDF happens INSIDE the sandbox container (`pdftoppm -png -r 300 -aa yes
-aaVector yes -cropbox -singlefile`, see sandbox/run-job.sh). This module is the host half: steps
1-6 on the rasterized image, with the pinned Pillow in requirements.lock.

    canonicalize_image(img_or_png_bytes, th) -> (png_bytes, meta)

Public for Opus 1 (crop canonicalization): import `hmb.submission.render.canonicalize_image`
(also re-exported as `hmb.submission.canonicalize_image`).

Form choices (recorded):
  * Output mode is 8-bit RGB for every input (colour crops keep their colour).
  * Resampled size = (max(1, floor(W*s + 0.5)), max(1, floor(H*s + 0.5))); letterbox offset =
    floor((canvas - w) / 2), floor((canvas - h) / 2).
  * PNG: Pillow writer with compress_level=9, optimize=False, no pnginfo, no dpi -> only IHDR, IDAT,
    IEND chunks (asserted by `png_chunk_types`).
  * Ink pixel (for V_EMPTY) = any canonical-canvas pixel that is not pure white (255,255,255); no
    threshold parameter is introduced.
"""
from __future__ import annotations

import io
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from hmb.hashing import sha256_bytes

RENDER_CANON_VERSION = "canon-9.7-v1"
PNG_COMPRESS_LEVEL = 9
RESAMPLE = "LANCZOS"

# Decompression-bomb guard for the host half: the container already refuses to rasterize above
# render_pixel_cap, so any larger image here is a harness error, not an item outcome.
Image.MAX_IMAGE_PIXELS = None


def pillow_version() -> str:
    import PIL
    return PIL.__version__


def _open(src) -> Image.Image:
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(src)))
    elif isinstance(src, (str, Path)):
        img = Image.open(src)
    else:
        raise TypeError(f"unsupported image source {type(src).__name__}")
    img.load()
    return img


def _composite_white(img: Image.Image) -> Image.Image:
    """Step 1: composite onto opaque white; returns RGB."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if img.mode == "I;16" or img.mode == "I":
        arr = np.asarray(img, dtype=np.float64)
        arr = np.clip(arr / (65535.0 if arr.max() > 255 else 255.0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "L").convert("RGB")
    return img.convert("RGB")


def encode_png(img: Image.Image) -> bytes:
    """Step 5: fixed compression, no time/metadata chunks."""
    clean = Image.frombytes(img.mode, img.size, img.tobytes())   # drops every info/metadata field
    buf = io.BytesIO()
    clean.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=False)
    return buf.getvalue()


def png_chunk_types(png: bytes) -> list[str]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    out, i = [], 8
    while i < len(png):
        (n,) = struct.unpack(">I", png[i:i + 4])
        out.append(png[i + 4:i + 8].decode("latin-1"))
        i += 12 + n
    return out


def canonicalize_image(img_or_png_bytes, th) -> tuple[bytes, dict]:
    """§9.7 steps 1-6. Returns (canonical PNG bytes, {s, W, H, w, h, offset, render_sha256, ...})."""
    canvas = int(th.value("canvas_px"))
    max_up = float(th.value("max_upscale"))
    img = _composite_white(_open(img_or_png_bytes))              # 1
    W, H = img.size
    s = min(canvas / max(W, H), max_up)                           # 2
    w = max(1, int(math.floor(W * s + 0.5)))
    h = max(1, int(math.floor(H * s + 0.5)))
    w, h = min(w, canvas), min(h, canvas)
    res = img if (w, h) == (W, H) else img.resize((w, h), Image.Resampling.LANCZOS)   # 3
    out = Image.new("RGB", (canvas, canvas), (255, 255, 255))    # 4
    off = ((canvas - w) // 2, (canvas - h) // 2)
    out.paste(res, off)
    png = encode_png(out)                                         # 5
    meta = {"s": s, "W": W, "H": H, "w": w, "h": h, "offset": list(off), "canvas": canvas,
            "render_sha256": sha256_bytes(png),                  # 6
            "canon_version": RENDER_CANON_VERSION, "resample": RESAMPLE,
            "pillow": pillow_version(), "png_compress_level": PNG_COMPRESS_LEVEL}
    return png, meta


def ink_fraction(canonical_png: bytes) -> float:
    """Share of canonical-canvas pixels that are not pure white (§9.5 V_EMPTY statistic)."""
    arr = np.asarray(Image.open(io.BytesIO(canonical_png)).convert("RGB"))
    return float((arr.min(axis=2) < 255).sum()) / float(arr.shape[0] * arr.shape[1])
