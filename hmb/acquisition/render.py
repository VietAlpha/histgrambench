"""Page render (§6.1): grayscale 8-bit, no auto-contrast / sharpen / despeckle, lossless PNG.

PDF pages are rasterised by MuPDF (pymupdf) straight to DeviceGray at `render_dpi`, with the
page's own geometry (physical DPI = render DPI, from the PDF page box). Image masters are
decoded by Pillow and converted to 8-bit luma (ITU-R 601, Pillow "L"); their pixels are used
as-is (no resampling, no EXIF re-orientation), and their physical DPI comes only from the
per-book `acquisition.source_dpi_nominal` set at census (None otherwise).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .imgstats import estimate_skew_deg, modal_text_height, sauvola_ink, sauvola_window

RENDER_VERSION = "render-1"
PNG_COMPRESSION = 6


class RenderError(RuntimeError):
    pass


def is_pdf(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def renderer_version(pdf: bool) -> str:
    if pdf:
        import fitz
        return f"pymupdf {fitz.VersionBind} / mupdf {fitz.VersionFitz}"
    import PIL
    return f"pillow {PIL.__version__}"


def page_count(path: str | Path) -> int:
    if is_pdf(path):
        import fitz
        with fitz.open(path) as doc:
            return doc.page_count
    from PIL import Image
    with Image.open(path) as im:
        return int(getattr(im, "n_frames", 1))


def load_page_gray(path: str | Path, page_index: int, dpi: int,
                   source_dpi_nominal: int | None = None) -> tuple[np.ndarray, dict]:
    """Return (gray uint8 HxW, meta). Raises RenderError on any decode failure."""
    path = Path(path)
    try:
        if is_pdf(path):
            import fitz
            with fitz.open(path) as doc:
                if not 0 <= page_index < doc.page_count:
                    raise RenderError(f"page {page_index} outside 0..{doc.page_count - 1}")
                pg = doc[page_index]
                z = dpi / 72.0
                pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False,
                                    annots=False)
                gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width].copy()
                meta = {"renderer": "pymupdf", "renderer_version": renderer_version(True),
                        "render_dpi": int(dpi), "physical_dpi": float(dpi),
                        "page_w_pt": float(pg.rect.width), "page_h_pt": float(pg.rect.height),
                        "page_rotation": int(pg.rotation)}
        else:
            from PIL import Image
            with Image.open(path) as im:
                n = int(getattr(im, "n_frames", 1))
                if not 0 <= page_index < n:
                    raise RenderError(f"frame {page_index} outside 0..{n - 1}")
                if n > 1:
                    im.seek(page_index)
                gray = np.asarray(im.convert("L"), dtype=np.uint8).copy()
            meta = {"renderer": "pillow", "renderer_version": renderer_version(False),
                    "render_dpi": None,
                    "physical_dpi": float(source_dpi_nominal) if source_dpi_nominal else None}
    except RenderError:
        raise
    except Exception as e:  # corrupt master, unsupported codec, decoder failure
        raise RenderError(f"{path.name} page {page_index}: {type(e).__name__}: {e}") from e
    meta.update(width=int(gray.shape[1]), height=int(gray.shape[0]), color_mode="gray8",
                render_version=RENDER_VERSION)
    return gray, meta


def encode_png(gray: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(gray), [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise RenderError("PNG encode failed")
    return buf.tobytes()


def page_measures(gray: np.ndarray, th, physical_dpi: float | None) -> dict:
    """Page-level measures: ink mask stats, x-height (G0b), page ink (G0c), skew (G0d)."""
    ink = sauvola_ink(gray, sauvola_window(th, physical_dpi or 300), th.value("sauvola_k"))
    return {"eff_res_xheight_px": modal_text_height(ink),
            "ink_frac_page": float(ink.mean()) if ink.size else 0.0,
            "skew_deg": estimate_skew_deg(ink), "_ink": ink}
