"""Benchmark model roster (plan §2.7, §7.3, §10.9, §6.3 G8b): owner-maintained YAML -> ModelSpecs.

The roster is protocol configuration chosen by the owner (§3.1), never by code. It is distinct from
the engineering agents that built this software. The template ships empty; every role that a
command needs must be filled before that command can run live.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .schemas import ModelSpec
from .vocab import Role

ROLES = ("builder", "judge", "contestant", "verifier")
# Roles cut by the model-role audit of 2026-09-23 (sorter G8a, tagger G8d, duplicate spotter G8c).
REMOVED_ROLES = ("classifier", "metadata", "embedder")
N_BUILDERS = 2                  # §7.3 as amended 2026-09-26 (owner): two answer-key writers in two families; since the
                                # owner decisions of 2026-09-26: GPT-6 Sol and Kimi K2.5 (Sonnet 5 + Kimi before; Opus 5.5 + Sol first)
EFFORT_REQUIRED_ROLES = ("builder", "judge", "verifier")   # effort pinned by the roster decision

# ---- Transports and the locked contestant roster (owner decisions 2026-09-25; D14 closed) ----
# Only the two Bedrock transports exist: one shared Runtime Converse transport and one shared Mantle
# Chat Completions transport. The direct Anthropic, OpenAI and Gemini connectors are removed, and a
# roster naming them is refused. Standard (default) service tier only; no cross-model fallback.
BEDROCK_REGION = "us-east-1"
MANTLE_BASE_URL = f"https://bedrock-mantle.{BEDROCK_REGION}.api.aws/v1"
TRANSPORTS = {"bedrock": "bedrock-runtime Converse", "bedrock_mantle": "bedrock-mantle Chat Completions"}
REMOVED_PROVIDERS = ("anthropic", "openai", "gemini")
SERVICE_TIER = "default"
_MANTLE_URL = re.compile(r"^https://bedrock-mantle\.[a-z0-9-]+\.api\.aws/")
_TIER_KEYS = ("service_tier", "serviceTier")


# Per-model request shape that differs from the transport default (repair 2026-09-26; live-verified on
# bedrock-mantle /openai/v1, reports/prompt_repair_20260926.md). GPT-6 models reject `max_tokens`
# ("Unsupported parameter: 'max_tokens' is not supported with this model") and any non-default
# temperature ("Only the default (1) value is supported"): the token cap goes in
# `max_completion_tokens` and no sampling control may be configured. Claude Sonnet 4.5 on Converse
# rejects temperature together with top_p ("`temperature` and `top_p` cannot both be specified"), so
# only one of them may be configured. Unlisted models keep the transport default.
# GPT-6 Sol and Luna moved to Global CRIS on bedrock-runtime Converse (owner decision 2026-09-26; published
# Global rate 10% below Mantle in-Region / US Geo CRIS, model cards); Converse sends `maxTokens`, and the
# effort goes in `decoding.additional_model_request_fields: {reasoning: {effort: ...}}` (live-verified).
MODEL_REQUEST_SHAPE: dict[str, dict] = {
    "global.openai.gpt-6-luna": {"sampling_controls": False},
    "global.openai.gpt-6-sol": {"sampling_controls": False},
    # GPT-6.1 Sol (writer since 2026-09-30, owner): its model card offers no global profile at launch, so it runs on
    # US CRIS Converse; request shape assumed equal to GPT-6 Sol (not live-checked yet).
    "us.openai.gpt-6.1-sol": {"sampling_controls": False},
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"sampling_exclusive": ["temperature", "top_p"]},
    # Kimi K2.5 (answer-key writer since 2026-09-26): its Bedrock model card documents no reasoning or effort
    # parameter, so a writer entry carries no effort (the writers' "medium" cannot be sent to it). Not live-checked.
    "moonshotai.kimi-k2.5": {"effort_control": False},
    # Claude Haiku 4.5 (contestant since 2026-10-02): Anthropic models on Converse reject temperature with top_p.
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": {"sampling_exclusive": ["temperature", "top_p"]},
}
# Routes retired by the Global CRIS migration of 2026-09-26: roster-check refuses them.
RETIRED_ROUTES = {"openai.gpt-6-luna": "global.openai.gpt-6-luna", "openai.gpt-6-sol": "global.openai.gpt-6-sol",
                  "us.openai.gpt-6-luna": "global.openai.gpt-6-luna", "us.openai.gpt-6-sol": "global.openai.gpt-6-sol"}
GLOBAL_CRIS_MODELS = ("global.openai.gpt-6-luna", "global.openai.gpt-6-sol")
SAMPLING_KEYS = ("temperature", "top_p", "seed")


def request_shape(model_id: str) -> dict:
    return dict(MODEL_REQUEST_SHAPE.get(model_id, {}))


@dataclass(frozen=True)
class Route:
    """One locked contestant: exact model or inference-profile id, transport, endpoint, whether it accepts
    image input (direct-figure track eligibility, Track A), and its declared family (company)."""
    logical_name: str
    model_id: str
    provider: str
    endpoint: str | None
    image_input: bool
    family: str


MANTLE_OPENAI_URL = f"https://bedrock-mantle.{BEDROCK_REGION}.api.aws/openai/v1"
# Owner decision 2026-09-26 (supersedes the 4-model lock of 2026-09-25 and the 15-model list of the same
# day): one image model per family plus one small anchor, and the DeepSeek V3.1 text-only floor. Trimmed
# for cost (judging scales with contestants): same-family duplicates dropped (Llama 4 Scout, Gemma 3
# 27B/12B/4B, Ministral 3 8B/3B, Magistral Small 1.2) and Amazon Nova Lite. Excluded as before: every
# grader model and the answer-key companies (Anthropic, OpenAI, xAI). Routes live-verified 2026-09-26.
# Owner decision 2026-09-26 (later): Kimi K2.5 became an answer-key writer, so it no longer competes (an
# answer-key company never competes): 5 image contestants + the DeepSeek V3.1 text-only floor.
LOCKED_CONTESTANTS = (
    Route("Llama 4 Maverick 17B-128E Instruct", "us.meta.llama4-maverick-17b-instruct-v1:0", "bedrock", None, True, "meta"),
    Route("Nemotron Nano 12B v2 VL", "nvidia.nemotron-nano-12b-v2", "bedrock", None, True, "nvidia"),
    Route("Gemma 4 26B A4B", "google.gemma-4-26b-a4b", "bedrock_mantle", MANTLE_OPENAI_URL, True, "google"),
    Route("Ministral 3 14B", "mistral.ministral-3-14b-instruct", "bedrock", None, True, "mistral"),
    # deliberately small low anchor (§2.7 capability spread)
    Route("Gemma 4 E2B", "google.gemma-4-e2b", "bedrock_mantle", MANTLE_OPENAI_URL, True, "google"),
    # text-only on this AWS offering: never sent an image, ineligible for the direct-figure track
    Route("DeepSeek V3.1", "deepseek.v3.1", "bedrock_mantle", MANTLE_BASE_URL, False, "deepseek"),
    # roster amendment (owner, 2026-10-02): Claude Haiku 4.5 on Global CRIS, contestant runs on batch
    Route("Claude Haiku 4.5", "global.anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock", None, True, "anthropic"),
)
# 2026-09-23 contestant rule; Moonshot added 2026-09-26 (Kimi K2.5 writes the answer key). OpenAI (the gatekeeper,
# a former writer) and xAI (a former writer) stay excluded. Anthropic removed by the owner's roster amendment of
# 2026-10-02 (Claude wrote only pilot cards; the test answer key is GPT-6.1 Sol + Kimi K2.5).
ANSWER_KEY_COMPANIES = ("openai", "xai", "moonshot")
LOCKED_BY_ID = {r.model_id: r for r in LOCKED_CONTESTANTS}


class RosterError(ValueError):
    pass


def load_roster(path: str | Path) -> dict[str, list[ModelSpec]]:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if d.get("version") != "roster-1":
        raise RosterError(f"{path}: expected version roster-1")
    removed = [r for r in REMOVED_ROLES if d.get(r)]
    if removed:
        raise RosterError(f"{path}: roles {removed} were removed by the 2026-09-23 role audit "
                          "(G8 is the gatekeeper alone)")
    out: dict[str, list[ModelSpec]] = {r: [] for r in ROLES}
    seen: set[tuple[str, str]] = set()
    for role in ROLES:
        for e in d.get(role) or []:
            if e.get("provider") in REMOVED_PROVIDERS:
                raise RosterError(f"{path}: {role} {e.get('model_id')}: provider {e['provider']!r} was removed "
                                  "(D14, 2026-09-25); only bedrock and bedrock_mantle exist")
            spec = ModelSpec(model_id=e["model_id"], provider=e["provider"], family=e["family"], role=Role(role),
                             snapshot=e.get("snapshot"), unit_price_id=e.get("unit_price_id"),
                             # an explicit `decoding: {}` is kept (GPT-6 accepts no sampling control)
                             decoding=(e["decoding"] or {}) if "decoding" in e else {"temperature": 0.0, "top_p": 1.0},
                             effort=e.get("effort"), input_mode=e.get("input_mode") or "vision",
                             endpoint=e.get("endpoint"), structured_output=e.get("structured_output"))
            if (role, spec.model_id) in seen:
                raise RosterError(f"duplicate {role} {spec.model_id}")
            seen.add((role, spec.model_id))
            out[role].append(spec)
    return out


def builder_rule_problems(builders: list[ModelSpec]) -> list[str]:
    """§7.3 builder rule, the one definition shared by `roster-check` and `cards-build` (D17), as
    amended 2026-09-26: exactly 2 builders in 2 distinct families (Grok 4.6 removed by the owner)."""
    n_fam = len({s.family for s in builders})
    if len(builders) != N_BUILDERS or n_fam != N_BUILDERS:
        return [f"§7.3: {N_BUILDERS} builders in {N_BUILDERS} distinct families required (got "
                f"{len(builders)} builders in {n_fam} families)"]
    return []


def effort_problems(r: dict[str, list[ModelSpec]]) -> list[str]:
    """Effort is pinned per model and consistent within a (role, family); roles whose effort the roster
    decision fixes must state it. A Converse model's effort must be encoded explicitly in
    decoding.additional_model_request_fields, because each family takes it differently."""
    probs: list[str] = []
    for role, specs in r.items():
        by_fam: dict[str, set] = {}
        for s in specs:
            by_fam.setdefault(s.family, set()).add(s.effort)
            no_control = request_shape(s.model_id).get("effort_control") is False
            if no_control and s.effort is not None:
                probs.append(f"{role} {s.model_id}: the model documents no effort control; leave effort unset")
            if role in EFFORT_REQUIRED_ROLES and s.effort is None and not no_control:
                probs.append(f"{role} {s.model_id}: no effort pinned (low|medium|high; roster decision 2026-09-23)")
            if s.effort is not None and s.provider == "bedrock"                     and not (s.decoding or {}).get("additional_model_request_fields"):
                probs.append(f"{role} {s.model_id}: effort {s.effort!r} has no Converse encoding "
                             "(decoding.additional_model_request_fields)")
        for fam, efforts in by_fam.items():
            if len(efforts) > 1:
                probs.append(f"{role} family {fam!r}: effort not pinned per family ({sorted(map(str, efforts))})")
    return probs


def request_shape_problems(r: dict[str, list[ModelSpec]]) -> list[str]:
    """Configured decoding the model's route rejects, and structured-output parity per role: every model
    of one role is constrained the same way (no route-specific advantage), and contestants never are."""
    probs: list[str] = []
    for role, specs in r.items():
        for s in specs:
            if request_shape(s.model_id).get("sampling_controls") is False:
                bad = sorted(k for k in SAMPLING_KEYS if k in (s.decoding or {}))
                if bad:
                    probs.append(f"{role} {s.model_id}: decoding {bad} is rejected by this model (only the default "
                                 "sampling is supported); set `decoding: {{}}`")
            excl = request_shape(s.model_id).get("sampling_exclusive") or []
            if excl and sum(k in (s.decoding or {}) for k in excl) > 1:
                probs.append(f"{role} {s.model_id}: configure only one of {excl} (the model rejects both together)")
        modes = {s.structured_output for s in specs}
        if len(modes) > 1:
            probs.append(f"{role}: structured_output differs between models {sorted(map(str, modes))} "
                         "(one constraint mode per role)")
        if role == "contestant" and any(m is not None for m in modes):
            probs.append("contestant: structured_output is not allowed (the output is raw TikZ)")
    return probs


def accepts_images(spec: ModelSpec) -> bool:
    """False for a text-only model: its spec says text_only, or its locked route has no image input."""
    route = LOCKED_BY_ID.get(spec.model_id)
    return spec.input_mode != "text_only" and (route is None or route.image_input)


def direct_figure_eligible(spec: ModelSpec) -> bool:
    """Track A (the direct-figure track) is open only to a locked contestant that accepts images."""
    route = LOCKED_BY_ID.get(spec.model_id)
    return route is not None and route.image_input and accepts_images(spec)


def mantle_endpoint_problem(url: str | None) -> str | None:
    if not url:
        return "bedrock_mantle needs its model card's endpoint URL pinned"
    if not _MANTLE_URL.match(url):
        return f"endpoint {url!r} is not a bedrock-mantle.<region>.api.aws URL (no other endpoint is allowed)"
    return None


def route_record(spec: ModelSpec) -> dict:
    """Run provenance for one model: resolved id, endpoint family, region, capability flags."""
    route = LOCKED_BY_ID.get(spec.model_id)
    return {"logical_name": route.logical_name if route else None, "model_id": spec.model_id,
            "snapshot": spec.snapshot, "transport": TRANSPORTS.get(spec.provider, spec.provider),
            "endpoint": spec.endpoint if spec.provider == "bedrock_mantle" else f"bedrock-runtime.{BEDROCK_REGION}",
            "region": BEDROCK_REGION, "service_tier": SERVICE_TIER, "locked": route is not None,
            "inference_scope": ("global_cris" if spec.model_id.startswith("global.") else
                                "us_geo_cris" if spec.model_id.startswith("us.") else "in_region"),
            "image_input": accepts_images(spec), "direct_figure_eligible": direct_figure_eligible(spec)}


def locked_roster_problems(contestants: list[ModelSpec]) -> list[str]:
    """The contestant list must be exactly the locked roster with its exact routes: no addition,
    upgrade, alias or substitution."""
    probs: list[str] = []
    got = {s.model_id: s for s in contestants}
    for mid in sorted(set(got) - set(LOCKED_BY_ID)):
        probs.append(f"contestant {mid}: not in the locked roster (no addition, alias or substitution)")
    for route in LOCKED_CONTESTANTS:
        s = got.get(route.model_id)
        if s is None:
            probs.append(f"locked contestant {route.logical_name} ({route.model_id}) is missing")
            continue
        if s.provider != route.provider:
            probs.append(f"contestant {route.model_id}: provider {s.provider!r}, locked route is {route.provider!r}")
        if route.provider == "bedrock_mantle" and s.endpoint != route.endpoint:
            probs.append(f"contestant {route.model_id}: endpoint {s.endpoint!r}, locked route is {route.endpoint!r}")
        if s.family != route.family:
            probs.append(f"contestant {route.model_id}: family {s.family!r}, locked family is {route.family!r}")
        want = "vision" if route.image_input else "text_only"
        if s.input_mode != want:
            probs.append(f"contestant {route.model_id}: input_mode {s.input_mode!r}, locked capability is {want!r}")
    return probs


def check_roster(r: dict[str, list[ModelSpec]]) -> list[str]:
    """Protocol checks that do not need data (§7.3, §10.9, §2.7, §6.3 G8b). Returns problems; empty = ok."""
    probs: list[str] = []
    for role in ROLES:
        if not r[role]:
            probs.append(f"role {role!r} is empty (owner roster decision required)")
    if len(r["verifier"]) > 1:
        probs.append("§6.3 G8b: exactly one gatekeeper (verifier) model")
    b_fams = {s.family for s in r["builder"]}
    j_fams = {s.family for s in r["judge"]}
    if r["builder"]:
        probs += builder_rule_problems(r["builder"])
    if r["judge"] and len(j_fams) < 3:
        probs.append("§10.9: >= 3 judge families required (third-judge escalation and LOJFO)")
    if b_fams & j_fams:
        probs.append(f"§10.9: builder and judge families overlap {sorted(b_fams & j_fams)} "
                     "(primary configuration requires disjointness; use per-item rotation otherwise)")
    probs += effort_problems(r)
    probs += request_shape_problems(r)
    judge_ids = {s.model_id for s in r["judge"]}
    for s in r["contestant"]:        # owner decisions 2026-09-23 / 2026-09-26
        if s.model_id in judge_ids:
            probs.append(f"contestant {s.model_id} is also a grader: a grader never competes in the benchmark it grades")
        if s.family in ANSWER_KEY_COMPANIES or s.family in {b.family for b in r["builder"]}:
            probs.append(f"contestant {s.model_id}: family {s.family!r} writes the answer key (answer-key companies excluded)")
    if r["contestant"]:
        probs += locked_roster_problems(r["contestant"])
    for role, specs in r.items():
        for s in specs:
            if s.model_id in RETIRED_ROUTES:
                probs.append(f"{role} {s.model_id}: retired route; use {RETIRED_ROUTES[s.model_id]} on provider "
                             "bedrock (Global CRIS, owner decision 2026-09-26)")
            if s.model_id in GLOBAL_CRIS_MODELS and s.provider != "bedrock":
                probs.append(f"{role} {s.model_id}: Global CRIS profiles are served on bedrock-runtime (provider bedrock)")
            if s.input_mode == "text_only" and role != "contestant":
                probs.append(f"{role} {s.model_id}: text_only input is allowed for contestants only "
                             "(builders, judges and the gatekeeper must see the image)")
            if s.provider == "bedrock_mantle" and mantle_endpoint_problem(s.endpoint):
                probs.append(f"{role} {s.model_id}: {mantle_endpoint_problem(s.endpoint)}")
            if any(k in (s.decoding or {}) for k in _TIER_KEYS):
                probs.append(f"{role} {s.model_id}: only the standard (default) service tier is allowed")
            if s.provider != "fake" and not s.unit_price_id:
                probs.append(f"{role} {s.model_id}: no unit_price_id (a live call needs a dated price)")
            if role in ("builder", "judge", "contestant") and not s.snapshot:
                probs.append(f"{role} {s.model_id}: no provider snapshot/version pinned (§2.7, §10.9)")
    return probs
