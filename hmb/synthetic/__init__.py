"""Known-geometry diagrams and the shared geometric predicate semantics.

Two roles, neither of them a benchmark stage (checkpoint simplification 2026-09-26: the Track C
control suite, its calibration and the evaluator-validation checkpoint were deleted):

  production  - `geo_predicates` and `spec.Geo`: the one geometric predicate semantics shared by the
                CGX extractor and adjudication (docs/CONTRACTS.md "Geometric predicate semantics").
  fixtures    - a deterministic generator of known-geometry diagrams (`families`, `build`, `spec`,
                `evaluator`, `tikz`, `admission`, `generate`, `render`) used by offline software
                tests, and `roundtrip`, the extractor round trip (generated TikZ -> sandbox -> CGX,
                compared with the geometry that produced it; no model call). These are engineering
                checks: they set no level, gate nothing and support no benchmark claim.
"""
from __future__ import annotations
