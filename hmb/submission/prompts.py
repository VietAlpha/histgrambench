"""§9.1 frozen contestant prompts and contestant-call assembly.

One frozen prompt per track: `prompts/reconstruction-A-v2.txt`, `prompts/reconstruction-B-v2.txt`,
published with `prompt_id` and `prompt_sha256` (sha256 of the file bytes). Each states verbatim the six
§9.1 elements (task, output contract, allowlist, prohibitions, not-judged list = §7.6
allowed_variation, script note); v2 (2026-09-26) drops the v1 sentence "Reflection is not allowed.": with layout
off the cards (asc-2.0.0) no scored fact is chirality-bearing, so reflection is not judged. Track B appends the context sentence and the context text follows.
Track C control contestants (§12.9 robustness gap) use the Track A prompt (form choice).

Text-only contestants (`ModelSpec.input_mode == "text_only"`, the §2.7(c)/(d) floor, e.g. DeepSeek
V3.1, roster decision 2026-09-23) are never sent the image. On every track they get the frozen no-image
prompt `reconstruction-T3-v2` followed by the item's context text (stripped caption and nearby prose),
so their Track A and Track B inputs coincide; `prompt_id` and `image_encoding.format == "none"` record it.

Decoding: the ModelSpec's decoding (temperature 0, top_p 1 by default), its pinned `effort`, and
`contestant_max_tokens` from thresholds, identical across models. Transport retries live in hmb.providers; a parse or
compile failure is never retried here (one output per item per track).
"""
from __future__ import annotations

from pathlib import Path

from hmb.hashing import sha256_bytes
from hmb.schemas import ModelRequest, ModelSpec
from hmb.vocab import Track

PROMPT_DIR = Path(__file__).with_name("prompts")
PROMPT_IDS = {Track.A: "reconstruction-A-v2", Track.B: "reconstruction-B-v2", Track.C: "reconstruction-A-v2"}


def load_prompt(track) -> tuple[str, str, str]:
    """(prompt_id, text, prompt_sha256)."""
    pid = PROMPT_IDS[Track(track)]
    b = (PROMPT_DIR / f"{pid}.txt").read_bytes()
    return pid, b.decode("utf-8"), sha256_bytes(b)


def contestant_prompt(track, model: ModelSpec) -> tuple[str, str, str]:
    """(prompt_id, text, prompt_sha256) a contestant receives on `track`: the track prompt, or the
    no-image prompt for a text-only model."""
    return load_t3_prompt() if model.input_mode == "text_only" else load_prompt(track)


def image_encoding(model: ModelSpec, th, crop_sha256: str) -> dict:
    """The `image_encoding` recorded on a submission (format none when no image was sent)."""
    if model.input_mode == "text_only":
        return {"format": "none", "crop_sha256": None}
    c = int(th.value("canvas_px"))
    return {"format": "png", "width": c, "height": c, "crop_sha256": crop_sha256}


def build_contestant_request(track, model: ModelSpec, crop_path: str | Path, crop_sha256: str,
                             context_text: str | None, th, *, phase: str, cache_scope: str) -> ModelRequest:
    t = Track(track)
    pid, text, _sha = contestant_prompt(t, model)
    text_only = model.input_mode == "text_only"
    if t == Track.B or text_only:
        text = text + "\n" + (context_text or "")
    return ModelRequest(spec=model, system=None, text=text, image_paths=[] if text_only else [str(crop_path)],
                        image_sha256=[] if text_only else [crop_sha256],
                        max_tokens=int(th.value("contestant_max_tokens")),
                        purpose="contestant", phase=phase, cache_scope=f"{cache_scope}|{pid}")


# §13.6 T3 text-only recall probe: a diagnostic CONDITION, never a track. Same six elements, no
# image, context text appended; stored under the derived item id `<item_id>__t3`.
T3_PROMPT_ID = "reconstruction-T3-v2"
T3_SUFFIX = "__t3"


def load_t3_prompt() -> tuple[str, str, str]:
    b = (PROMPT_DIR / f"{T3_PROMPT_ID}.txt").read_bytes()
    return T3_PROMPT_ID, b.decode("utf-8"), sha256_bytes(b)


def t3_item_id(item_id: str) -> str:
    return item_id + T3_SUFFIX


def is_t3_item(item_id: str) -> bool:
    return item_id.endswith(T3_SUFFIX)


def build_t3_request(model: ModelSpec, context_text: str, th, *, phase: str, cache_scope: str) -> ModelRequest:
    pid, text, _sha = load_t3_prompt()
    return ModelRequest(spec=model, system=None, text=text + "\n" + (context_text or ""), image_paths=[],
                        image_sha256=[], max_tokens=int(th.value("contestant_max_tokens")),
                        purpose="contestant_t3", phase=phase, cache_scope=f"{cache_scope}|{pid}")


def decoding_params(model: ModelSpec, th) -> dict:
    d = dict(model.decoding)
    d.setdefault("temperature", 0.0)
    d.setdefault("top_p", 1.0)
    d["max_tokens"] = int(th.value("contestant_max_tokens"))
    d.setdefault("seed", None)
    d["effort"] = model.effort
    return d
