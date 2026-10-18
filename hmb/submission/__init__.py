"""Section 9: contestant protocol, extraction, wrapper normalization, sandboxed compilation, rendering.

Contract functions (docs/CONTRACTS.md):
    extract_candidate(raw, stop_reason) -> Extracted              # §9.2
    normalize(ex, th) -> Normalized                               # §9.3 / §9.4
    compile_in_sandbox(tex, th, workdir) -> CompileResult          # §9.6 (Docker only; no host TeX)
    canonicalize_image(img_or_png_bytes, th) -> (bytes, dict)      # §9.7 (shared with crops)
    process_submission(raw, stop_reason, *, item_id, model, track, layout, th, meta) -> SubmissionRecord
"""
from .extract import Extracted, extract_candidate, reextract_first_vs_last, split_inline_thinking
from .process import (finalize_submission, harness_object, harness_version, harness_version_parts,
                      model_dir_name, outcome_distribution, process_submission, run_contestant, run_t3_probe)
from .prompts import build_contestant_request, build_t3_request, is_t3_item, load_prompt, t3_item_id
from .render import canonicalize_image, ink_fraction
from .sandbox import (CompileResult, SandboxUnavailable, compile_in_sandbox, determinism_sample,
                      font_coverage_gate, harness_determinism_rerun)
from .wrapper import HarnessInvariantError, Normalized, normalize

__all__ = [
    "Extracted", "extract_candidate", "reextract_first_vs_last", "split_inline_thinking",
    "Normalized", "normalize", "HarnessInvariantError",
    "CompileResult", "SandboxUnavailable", "compile_in_sandbox", "harness_determinism_rerun",
    "determinism_sample", "font_coverage_gate",
    "canonicalize_image", "ink_fraction",
    "build_contestant_request", "load_prompt",
    "process_submission", "finalize_submission", "model_dir_name", "run_contestant", "run_t3_probe",
    "build_t3_request", "t3_item_id", "is_t3_item", "harness_object", "outcome_distribution", "harness_version",
    "harness_version_parts",
]
