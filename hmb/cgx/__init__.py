"""§10.2 CGX extraction: `extract_cgx(pdf_path, *, item_id, model_id, track, th) -> CgxRecord`."""
from .extract import EXTRACTOR_VERSION, cgx_geometry, extract_cgx, label_kind, polygons_from_strokes, write_cgx

__all__ = ["extract_cgx", "cgx_geometry", "write_cgx", "label_kind", "polygons_from_strokes", "EXTRACTOR_VERSION"]
