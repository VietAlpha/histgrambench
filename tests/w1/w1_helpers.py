"""Drawing helpers for B1 tests (imported as w1_helpers; unique name avoids module clashes)."""
import cv2
import numpy as np


def blank(h, w, v=255):
    return np.full((h, w), v, np.uint8)


def draw_text_rows(img, x0, y0, n_rows, n_chars, glyph_h=14, glyph_w=8, gap=4, row_gap=16):
    """Glyph-like filled blobs laid out as text lines (each blob is one component)."""
    for r in range(n_rows):
        y = y0 + r * (glyph_h + row_gap)
        for c in range(n_chars):
            x = x0 + c * (glyph_w + gap)
            cv2.rectangle(img, (x, y), (x + glyph_w - 1, y + glyph_h - 1), 0, -1)
    return img


def draw_diagram(img, cx, cy, r, thickness=3):
    """A circle with an inscribed triangle: line art with long strokes."""
    cv2.circle(img, (cx, cy), r, 0, thickness)
    pts = [(cx, cy - r), (cx - int(0.87 * r), cy + r // 2), (cx + int(0.87 * r), cy + r // 2)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cv2.line(img, pts[a], pts[b], 0, thickness)
    return img


def source_crop(box, snapshot_id="snap_0000000000000000"):
    """A valid §5.2 SourceCrop record (Paddle saved crop provenance) for fixture rows."""
    x0, y0, x1, y1 = (int(v) for v in box)
    return {"source": "paddle_layout@1.6", "snapshot_id": snapshot_id,
            "file": f"imgs/p00001/img_in_image_box_{x0}_{y0}_{x1}_{y1}.jpg", "file_sha256": "0" * 64,
            "block_bbox": [x0, y0, x1, y1], "frame": "paddle_preprocessed", "frame_wh": [1000, 1400],
            "rotation_deg": 0, "modifications": ["dewarped", "resampled", "jpeg_recompressed", "paddle_bounds"],
            "regenerable_from_master": False}
