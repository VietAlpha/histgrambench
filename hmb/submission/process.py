"""process_submission: raw response -> outcome code + every §5.3 artifact, written immutably.

Artifacts (paths relative to the layout root, §5.3 / hmb.store.Layout):
  submissions/<model>/<track>/<item>.txt            raw output (answer text as returned)
  submissions/<model>/<track>/<item>.thinking.txt   provider thinking, stored separately (if any)
  submissions/<model>/<track>/<item>.tex            normalized tex (only when normalization succeeded)
  compiled/<model>/<track>/<item>/cand.pdf, cand.log, render.png, compile.json, extraction.json
  compiled/<model>/<track>/<item>/submission_stage.json   validated compile-stage SubmissionRecord
  submissions/<model>/<track>/<item>.json           FINAL SubmissionRecord, written once by
                                                    finalize_submission (after CGX, verdicts, scores)
Lifecycle (lead decision, red-team #1): process_submission writes and returns the compile-stage
record; finalize_submission fills cgx/verdicts/scores paths + sha256 of the stored bytes and writes
the final §5.3 record once. Only outcome OK proceeds to adjudication; a non-OK cell finalizes with
scores only. Re-running a cell with the same raw output returns the stored stage record (resume); a
different raw output for the same cell raises ImmutableConflict (one output per item per track). `<model>` is the model id with characters outside [A-Za-z0-9._-]
replaced by "_".
"""
from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from hmb.hashing import H, sha256_bytes
from hmb.schemas import Attempt, ModelSpec, SubmissionRecord, validate_record
from hmb.store import Layout, read_json, write_immutable, write_json_immutable
from hmb.store import safe_model_dir as model_dir_name   # one sanitizer for every per-model path
from hmb.vocab import OUTCOME_GROUPS, ExtractionPath, Outcome, Track

from . import extract as _extract
from . import render as _render
from . import sandbox as _sandbox
from . import wrapper as _wrapper
from .extract import extract_candidate, split_inline_thinking
from .prompts import build_contestant_request, decoding_params, load_prompt
from .wrapper import normalize

PKG_DIR = Path(__file__).resolve().parent
CGX_DIR = PKG_DIR.parent / "cgx"
# Modules outside hmb/submission and hmb/cgx whose code decides CGX output: geometric predicates and
# their helpers, closure/MGS and the label normalization it applies, and the ontology.
CGX_DEPENDENCIES = ("synthetic/geo_predicates.py", "synthetic/spec.py", "cards/closure.py", "cards/normalize.py",
                    "ontology.py")


# ----------------------------------------------------------------------------- harness version
def code_hash() -> str:
    """Content hash standing in for 'evaluation code commit' (a working tree has no commit to cite):
    sha256 over the sorted (relative path, sha256) list of hmb/submission and hmb/cgx sources, prompts
    and wrapper, plus CGX_DEPENDENCIES."""
    root = PKG_DIR.parent
    paths = [p for base in (PKG_DIR, CGX_DIR) for p in base.rglob("*")
             if p.is_file() and p.suffix in (".py", ".txt", ".tex")]
    paths += [root / rel for rel in CGX_DEPENDENCIES]
    files = sorted([p.relative_to(root).as_posix(), sha256_bytes(p.read_bytes())] for p in paths)
    return sha256_bytes(json.dumps(files).encode())


def wrapper_version_id() -> str:
    """wrapper_v1 identity with the engine line folded in (C16): H(wrapper_v1.tex bytes, ENGINE_LINE)."""
    return f"{_wrapper.WRAPPER_VERSION}:" + H(_wrapper._WRAPPER_FILE.read_bytes(), _sandbox.ENGINE_LINE)


def harness_version_parts() -> dict:
    lock = _sandbox.read_lock()
    man = lock.get("manifest", {})
    return {"evaluation_code": code_hash(), "sandbox_image_digest": lock["image_id"],
            "wrapper_version": wrapper_version_id(),
            "extraction_rules_version": _extract.EXTRACTION_RULES_VERSION,
            "rasterizer": f"{man.get('rasterizer', 'pdftoppm')}|{_render.RENDER_CANON_VERSION}"
                          f"|pillow={_render.pillow_version()}"}


def harness_object(th) -> dict:
    """The §15.2 harness object (the same keys Opus 1's release build uses)."""
    p = harness_version_parts()
    c = th.value("canvas_px")
    return {"eval_commit": p["evaluation_code"], "container_digest": p["sandbox_image_digest"],
            "wrapper_version": p["wrapper_version"], "extraction_rules_version": p["extraction_rules_version"],
            "rasterizer": p["rasterizer"], "dpi": th.value("render_dpi_candidate"), "canvas": f"{c}x{c}"}


def harness_version(th) -> str:
    """§15.1 harness_version: the ONE formula, `hmb.manifests.harness_version_of` over the §15.2
    harness object (eval_commit = code_hash(), a content hash of the evaluation code)."""
    from hmb.manifests import harness_version_of
    return harness_version_of(harness_object(th))


# ----------------------------------------------------------------------------- main
_REQUIRED_META = ("dataset_version", "model_version_string", "access_date", "provider_terms_version",
                  "output_release", "image_encoding")


def _rel(layout: Layout, p: Path) -> str:
    return p.relative_to(layout.root).as_posix()


def process_submission(raw: str, stop_reason, *, item_id: str, model: ModelSpec, track, layout: Layout,
                       th, meta: dict) -> SubmissionRecord:
    missing = [k for k in _REQUIRED_META if k not in meta]
    if missing:
        raise KeyError(f"process_submission meta lacks {missing}")
    t = Track(track).value
    mdir = model_dir_name(model.model_id)
    rec_path = layout.compile_dir(mdir, t, item_id) / "submission_stage.json"
    raw_b = (raw or "").encode("utf-8")
    raw_path = layout.raw_output(mdir, t, item_id)
    if rec_path.exists():
        prev = SubmissionRecord.model_validate(read_json(rec_path))
        if prev.raw_output_sha256 == sha256_bytes(raw_b):
            return prev
    raw_sha = write_immutable(raw_path, raw_b)            # different raw -> ImmutableConflict
    answer, inline_thinking = split_inline_thinking(raw or "")
    thinking = meta.get("thinking") or inline_thinking
    if thinking:
        write_immutable(raw_path.with_suffix(".thinking.txt"), thinking.encode("utf-8"))
    pid, _txt, psha = load_prompt(t)
    cdir = layout.compile_dir(mdir, t, item_id)
    rec: dict = {
        "item_id": item_id, "dataset_version": meta["dataset_version"],
        "release_hash": meta.get("release_hash"), "model_id": model.model_id,
        "model_version_string": meta["model_version_string"], "model_family": model.family,
        "access_date": meta["access_date"], "provider_terms_version": meta["provider_terms_version"],
        "output_release": meta["output_release"], "track": t,
        "prompt_id": meta.get("prompt_id", pid), "prompt_sha256": meta.get("prompt_sha256", psha),
        "decoding_params": meta.get("decoding_params") or decoding_params(model, th),
        "image_encoding": meta["image_encoding"], "stop_reason": stop_reason,
        "attempts": [a if isinstance(a, dict) else a.model_dump() for a in meta.get("attempts", [])]
                    or [Attempt(index=1, utc=meta["access_date"]).model_dump()],
        "harness_version": harness_version(th), "sandbox_image_digest": _sandbox.sandbox_image_digest(),
        "raw_output": _rel(layout, raw_path), "raw_output_sha256": raw_sha,
        "extraction_path": ExtractionPath.none.value, "n_blocks": 0,
        "judge_config_id": meta.get("judge_config_id"),
    }
    ex = extract_candidate(answer, stop_reason)
    rec["extraction_path"], rec["n_blocks"] = ex.path.value, ex.n_blocks
    ext_info = {"path": ex.path.value, "n_blocks": ex.n_blocks, "n_fenced_all": ex.n_fenced_all,
                "outcome": ex.outcome.value if ex.outcome else None, "notes": ex.notes,
                "span_sha256": sha256_bytes(ex.span.encode("utf-8")) if ex.span is not None else None}
    if not ex.ok:
        rec["outcome_code"] = ex.outcome.value
        write_json_immutable(cdir / "extraction.json", ext_info)
        return _finish(layout, rec_path, rec)
    nz = normalize(ex, th)                                 # HarnessInvariantError propagates (fails run)
    rec["wrapper_class"] = nz.wrapper_class.value if nz.wrapper_class else None
    ext_info.update({"wrapper_class": rec["wrapper_class"], "reject_token": nz.reject_token})
    write_json_immutable(cdir / "extraction.json", ext_info)
    if nz.outcome is not None:
        rec["outcome_code"] = nz.outcome.value
        return _finish(layout, rec_path, rec)
    tex_b = nz.tex_bytes
    tex_path = layout.normalized_tex(mdir, t, item_id)
    rec["body_sha256"] = nz.body_sha256
    rec["normalized_tex"] = _rel(layout, tex_path)
    rec["normalized_tex_sha256"] = write_immutable(tex_path, tex_b)
    with tempfile.TemporaryDirectory(prefix="hmb_compile_") as tmp:
        cr = _sandbox.compile_in_sandbox(tex_b, th, Path(tmp))
        for name in ("cand.pdf", "cand.log", "render.png"):
            f = Path(tmp) / name
            if f.exists():
                write_immutable(cdir / name, f.read_bytes())
    stable = cr.to_json()
    for k in ("pdf_path", "log_path", "render_path"):
        stable[k] = _rel(layout, cdir / Path(stable[k]).name) if stable.get(k) else None
    write_json_immutable(cdir / "compile.json", stable)
    rec.update({"outcome_code": cr.outcome.value, "cpu_seconds": cr.cpu_seconds,
                "page_count": cr.page_count, "ink_fraction": cr.ink_fraction,
                "pdf_sha256": cr.pdf_sha256, "render_sha256": cr.render_sha256})
    return _finish(layout, rec_path, rec)


def _finish(layout: Layout, rec_path: Path, rec: dict) -> SubmissionRecord:
    sub = validate_record(SubmissionRecord, rec)
    write_json_immutable(rec_path, sub.model_dump(mode="json"))
    return sub


def run_contestant(*, item_id: str, track, model: ModelSpec, crop_path: Path, crop_sha256: str,
                   context_text: str | None, layout: Layout, th, ledger, allow_live: bool,
                   phase: str, cache_scope: str, meta: dict, provider_override=None) -> SubmissionRecord:
    """Assemble the §9.1 call, obtain ONE response via hmb.providers.call_model, then process it."""
    from hmb.providers import call_model
    req = build_contestant_request(track, model, crop_path, crop_sha256, context_text, th,
                                   phase=phase, cache_scope=cache_scope)
    resp = call_model(req, layout, ledger, allow_live=allow_live, provider_override=provider_override)
    if resp.fake and th.is_production:
        raise RuntimeError("production path refuses a fake provider response")
    m = dict(meta)
    m.setdefault("model_version_string", resp.model_version_string)
    m["attempts"] = [a.model_dump() for a in resp.attempts] or m.get("attempts", [])
    m["thinking"] = resp.thinking
    m["decoding_params"] = decoding_params(model, th)
    return process_submission(resp.text, resp.stop_reason, item_id=item_id, model=model, track=track,
                              layout=layout, th=th, meta=m)


def run_t3_probe(*, item_id: str, model: ModelSpec, context_text: str, layout: Layout, th, ledger,
                 allow_live: bool, phase: str, cache_scope: str, meta: dict,
                 provider_override=None) -> SubmissionRecord:
    """§13.6 T3 text-only recall probe: context text, NO image, same harness (extraction, wrapper,
    sandbox, render), stored under the derived item id `<item_id>__t3` with prompt_id
    reconstruction-T3-v2. It is a condition, not a track: the record's `track` is "B" (the
    context-conditioned protocol it mirrors), and every aggregation must drop `is_t3_item` ids
    (equivalently prompt_id == T3_PROMPT_ID); scores are reported per prominence tercile only.
    Production caller: `pipeline.run_t3_probe_model` (`hmb t3-probe`)."""
    from hmb.providers import call_model

    from .prompts import build_t3_request, load_t3_prompt, t3_item_id
    req = build_t3_request(model, context_text, th, phase=phase, cache_scope=cache_scope)
    resp = call_model(req, layout, ledger, allow_live=allow_live, provider_override=provider_override)
    if resp.fake and th.is_production:
        raise RuntimeError("production path refuses a fake provider response")
    pid, _t, psha = load_t3_prompt()
    m = dict(meta)
    m.setdefault("model_version_string", resp.model_version_string)
    m.setdefault("image_encoding", {"format": "none", "crop_sha256": None})
    m.update({"attempts": [a.model_dump() for a in resp.attempts] or m.get("attempts", []),
              "thinking": resp.thinking, "decoding_params": decoding_params(model, th),
              "prompt_id": pid, "prompt_sha256": psha})
    return process_submission(resp.text, resp.stop_reason, item_id=t3_item_id(item_id), model=model,
                              track="B", layout=layout, th=th, meta=m)


def outcome_distribution(records: Iterable) -> dict:
    """Full §9.5 code distribution and derived rates over the same denominator n."""
    codes = [Outcome(r.outcome_code if hasattr(r, "outcome_code") else r) for r in records]
    n = len(codes)
    cnt = Counter(codes)
    dist = {o.value: cnt.get(o, 0) for o in Outcome}
    rates = {g: (sum(cnt.get(o, 0) for o in members) / n if n else None)
             for g, members in OUTCOME_GROUPS.items()}
    return {"n": n, "distribution": dist, "rates": rates,
            "E_TRUNCATED_rate": (cnt.get(Outcome.E_TRUNCATED, 0) / n) if n else None}


def finalize_submission(stage: SubmissionRecord, layout: Layout, *, cgx_path=None, verdicts_path=None,
                        scores_path=None, judge_config_id: str | None = None) -> SubmissionRecord:
    """Fill the §5.3 cgx/verdicts/scores fields (relative path + sha256 of the stored bytes), validate,
    and write the final record once to layout.submission_record (immutable)."""
    ok = stage.outcome_code == Outcome.OK.value
    if ok and (cgx_path is None or verdicts_path is None or scores_path is None):
        raise ValueError("an OK cell finalizes with cgx, verdicts and scores")
    if not ok and (cgx_path is not None or verdicts_path is not None):
        raise ValueError(f"a {stage.outcome_code} cell never reaches adjudication; finalize with scores only")
    if scores_path is None:
        raise ValueError("finalize_submission requires the scores artifact")
    upd: dict = {}
    for field, path in (("cgx_record", cgx_path), ("verdicts", verdicts_path), ("scores", scores_path)):
        if path is None:
            continue
        path = Path(path)
        sha_field = "cgx_sha256" if field == "cgx_record" else f"{field}_sha256"
        upd[field] = _rel(layout, path.resolve() if not path.is_absolute() else path)
        upd[sha_field] = sha256_bytes(path.read_bytes())
    if judge_config_id is not None:
        upd["judge_config_id"] = judge_config_id
    data = stage.model_dump(mode="json")
    data.update(upd)
    final = validate_record(SubmissionRecord, data)
    mdir = model_dir_name(final.model_id)
    write_json_immutable(layout.submission_record(mdir, final.track, final.item_id), final.model_dump(mode="json"))
    return final
