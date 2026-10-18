"""Hash chain (§15.1), release manifest (§15.2), run manifests (§15.3), tombstones and the
version trigger table (§15.4), and the plotting rule (one axis = one release_hash).

`H` is `hmb.hashing.H` (parts joined by 0x1F; structured parts as RFC 8785 canonical JSON; binary
files as raw bytes). Every chain value is a 64-hex digest; release.json prints `sha256:<hex>`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .hashing import H, sha256_bytes
from .store import Layout, read_json, write_json_immutable, write_mutable

BENCHMARK_PREFIX = "histmathbench-"
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
# §4.4 licensing layers carried by every release. L4 (project-created cards and metadata) is the owner's
# decision D9, 2026-09-23. Scans and crops are NOT covered by it: their terms are the per-book
# intersection of L1-L3 recorded in sources/rights_manifest.yaml (§4.4 standing rule).
DATA_LICENCE = {"L4_cards_and_metadata": "Apache-2.0", "L5_code_schemas_prompts": "Apache-2.0",
                "scans_and_crops": "per book: intersection of L1-L3 in sources/rights_manifest.yaml"}


class ManifestError(ValueError):
    pass


# --------------------------------------------------------------------------- §15.1 chain
def page_hash(page_image_bytes: bytes) -> str:
    return H(page_image_bytes)


def item_hash(crop_image_bytes: bytes, pipeline_version: str, page_hash_hex: str, crop_bbox) -> str:
    return H(crop_image_bytes, pipeline_version, page_hash_hex, [int(v) for v in crop_bbox])


def card_hash(frozen_eligible_fact_set: list, asc_version: str, consensus_rule_hash: str) -> str:
    return H(frozen_eligible_fact_set, asc_version, consensus_rule_hash)


def cardset_hash(pairs) -> str:
    """H(sorted list of (item_hash, card_hash))."""
    return H(sorted([str(a), str(b)] for a, b in pairs))


def file_hash(path: str | Path) -> str:
    """H over raw bytes of a file (split_groups.yaml, thresholds.json)."""
    return H(Path(path).read_bytes())


def harness_version(eval_commit: str, sandbox_image_digest: str, wrapper_version: str,
                    extraction_rules_version: str, rasterizer: dict) -> str:
    """rasterizer: {name, version, dpi, canvas}."""
    return H(eval_commit, sandbox_image_digest, wrapper_version, extraction_rules_version, rasterizer)


def release_hash(cardset: str, split: str, thresholds: str, judge_config: str, harness: str, metric: str) -> str:
    """§15.1 without controls_hash (the Track C control suite was deleted, 2026-09-26)."""
    return H(cardset, split, thresholds, judge_config, harness, metric)


def _hex(v: str) -> str:
    v = v[len("sha256:"):] if isinstance(v, str) and v.startswith("sha256:") else v
    if not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v):
        raise ManifestError(f"not a sha256 digest: {v!r}")
    return v


# --------------------------------------------------------------------------- §15.2 release.json
RELEASE_KEYS = ("cardset_hash", "split_hash", "thresholds_hash", "judge_config_hash", "metric_hash")
# §15.2 "harness" object. §15.1 harness_version also needs wrapper_version and
# extraction_rules_version, so the object carries them; tex_distribution is informational.
HARNESS_KEYS = ("eval_commit", "container_digest", "wrapper_version", "extraction_rules_version",
                "rasterizer", "dpi", "canvas")


def harness_version_of(harness: dict) -> str:
    """§15.1 harness_version computed from the §15.2 harness object."""
    miss = [k for k in HARNESS_KEYS if harness.get(k) in (None, "")]
    if miss:
        raise ManifestError(f"harness object lacks {miss}")
    return harness_version(harness["eval_commit"], harness["container_digest"], harness["wrapper_version"],
                           harness["extraction_rules_version"],
                           {"rasterizer": harness["rasterizer"], "dpi": harness["dpi"], "canvas": harness["canvas"]})


def _release_hash_of(rel: dict) -> str:
    h = {k: _hex(rel[k]) for k in RELEASE_KEYS}
    return release_hash(h["cardset_hash"], h["split_hash"], h["thresholds_hash"], h["judge_config_hash"],
                        harness_version_of(rel["harness"]), h["metric_hash"])


def build_release(*, version: str, created: str, hashes: dict, thresholds_frozen_at: str | None,
                  harness: dict, counts: dict, funnel_path: str, tombstones_path: str = "manifests/tombstones.json"
                  ) -> dict:
    """Assemble release.json exactly in the §15.2 shape; release_hash is computed from the five hash
    fields plus harness_version derived from the harness object (no top-level harness_version)."""
    if not _SEMVER.match(version):
        raise ManifestError(f"version {version!r} is not MAJOR.MINOR.PATCH")
    missing = [k for k in RELEASE_KEYS if not hashes.get(k)]
    if missing:
        raise ManifestError(f"release needs {missing}")
    if not thresholds_frozen_at:
        raise ManifestError("release needs thresholds_frozen_at (§15.5)")
    for k in ("books", "split_groups", "items_eligible", "facts_eligible"):
        if not isinstance(counts.get(k), int) or counts[k] < 0:
            raise ManifestError(f"counts.{k} must be a non-negative integer")
    h = {k: _hex(hashes[k]) for k in RELEASE_KEYS}
    rel = {"benchmark_version": BENCHMARK_PREFIX + version, "release_hash": None, "created": created,
           "cardset_hash": h["cardset_hash"], "split_hash": h["split_hash"], "thresholds_hash": h["thresholds_hash"],
           "thresholds_frozen_at": thresholds_frozen_at, "judge_config_hash": h["judge_config_hash"],
           "metric_hash": h["metric_hash"], "harness": dict(harness),
           "counts": counts, "funnel": funnel_path, "tombstones": tombstones_path,
           "data_licence": dict(DATA_LICENCE)}           # metadata, outside the §15.1 hash chain
    rel["release_hash"] = "sha256:" + _release_hash_of(rel)
    return rel


def verify_release(rel: dict) -> bool:
    try:
        return rel.get("release_hash") == "sha256:" + _release_hash_of(rel)
    except (KeyError, TypeError, ManifestError):
        return False


def write_release(layout: Layout, rel: dict) -> Path:
    """releases/<version>/release.json is immutable; manifests/release.json points at the current one."""
    if not verify_release(rel):
        raise ManifestError("release_hash does not match its inputs")
    ver = rel["benchmark_version"][len(BENCHMARK_PREFIX):]
    frozen = layout.p("releases", ver, "release.json")
    write_json_immutable(frozen, rel)
    write_mutable(layout.release_manifest, (json.dumps(rel, indent=1, sort_keys=True) + "\n").encode())
    return frozen


# --------------------------------------------------------------------------- §15.3 run.json
def build_run_manifest(*, release_hash_value: str, model_record: dict, prompt_sha256: str, scoring_window: dict,
                       drift_probe: dict, verdicts_hash: str, funnel_eligibility_rate: float | None,
                       output_release: str) -> dict:
    need = ("model_id", "provider_version", "family", "access_date", "decoding", "attempts")
    miss = [k for k in need if k not in model_record]
    if miss:
        raise ManifestError(f"model record lacks {miss}")
    if output_release not in ("public", "withheld"):
        raise ManifestError("output_release must be public|withheld")
    for k in ("open", "close"):
        if not scoring_window.get(k):
            raise ManifestError(f"scoring_window.{k} missing")
    for k in ("open_agreement", "close_agreement", "kappa_min", "pass"):
        if k not in drift_probe:
            raise ManifestError(f"drift_probe.{k} missing")
    return {"release_hash": "sha256:" + _hex(release_hash_value), "model": model_record,
            "model_hash": H(model_record), "prompt_sha256": _hex(prompt_sha256), "scoring_window": scoring_window,
            "drift_probe": drift_probe, "verdicts_hash": _hex(verdicts_hash),
            "funnel_eligibility_rate": funnel_eligibility_rate, "output_release": output_release}


def write_run_manifest(layout: Layout, model: str, run: dict) -> Path:
    p = layout.run_manifest(model)
    write_json_immutable(p, run)
    return p


# --------------------------------------------------------------------------- tombstones
def load_tombstones(layout: Layout) -> list[dict]:
    return read_json(layout.tombstones) if layout.tombstones.exists() else []


def add_tombstone(layout: Layout, item_id: str, retired_in_version: str, reason: str, evidence: str) -> dict:
    """Append {item_id, retired_in_version, reason, evidence}; existing entries are preserved verbatim."""
    if not (item_id and retired_in_version and reason and evidence):
        raise ManifestError("tombstone needs item_id, retired_in_version, reason and evidence")
    stones = load_tombstones(layout)
    if any(s["item_id"] == item_id for s in stones):
        raise ManifestError(f"{item_id} is already tombstoned")
    entry = {"item_id": item_id, "retired_in_version": retired_in_version, "reason": reason, "evidence": evidence}
    stones.append(entry)
    write_mutable(layout.tombstones, (json.dumps(stones, indent=1, ensure_ascii=False) + "\n").encode())
    return entry


def assert_id_usable(layout: Layout, item_id: str) -> None:
    """Ids are never reused: a tombstoned id may never be issued again."""
    if any(s["item_id"] == item_id for s in load_tombstones(layout)):
        raise ManifestError(f"{item_id} is tombstoned; ids are never reused (§15.4)")


# --------------------------------------------------------------------------- §15.4 triggers
VERSION_TRIGGERS: dict[str, str] = {
    "card_content": "MAJOR", "ontology": "MAJOR", "rules_R_E": "MAJOR", "consensus_rule": "MAJOR",
    "eligibility_levels": "MAJOR",
    "judge_model": "MAJOR", "judge_snapshot": "MAJOR", "judge_prompt": "MAJOR", "extractor": "MAJOR",
    "judge_protocol_constants": "MAJOR",
    "metric_definitions": "MAJOR", "metric_caps": "MAJOR",
    "compile_container": "MAJOR", "tex_pin": "MAJOR", "rasterizer": "MAJOR", "dpi": "MAJOR", "wrapper": "MAJOR",
    "extraction_rules": "MAJOR",
    "split_assignment": "MAJOR", "split_group_graph": "MAJOR",
    "thresholds": "MAJOR",
    "add_books_or_items": "MINOR", "add_analysis_slice": "MINOR", "rights_retirement": "MINOR",
    "documentation": "PATCH", "paths": "PATCH", "non_scoring_metadata": "PATCH",
}
_ORDER = {"PATCH": 0, "MINOR": 1, "MAJOR": 2}


def version_bump(change_type: str) -> str:
    try:
        return VERSION_TRIGGERS[change_type]
    except KeyError:
        raise ManifestError(f"unknown change type {change_type!r}; the trigger table is closed") from None


def next_version(current: str, changes: list[str]) -> str:
    """Apply the largest bump implied by `changes` (a rights retirement also needs a tombstone)."""
    m = _SEMVER.match(current)
    if not m:
        raise ManifestError(f"version {current!r} is not MAJOR.MINOR.PATCH")
    if not changes:
        raise ManifestError("no change listed; a release without a change keeps its version")
    bump = max((version_bump(c) for c in changes), key=_ORDER.__getitem__)
    a, b, c = (int(x) for x in m.groups())
    return {"MAJOR": f"{a + 1}.0.0", "MINOR": f"{a}.{b + 1}.0", "PATCH": f"{a}.{b}.{c + 1}"}[bump]


# --------------------------------------------------------------------------- plotting rule
def check_plot_axis(points: list[dict]) -> str:
    """One leaderboard axis carries exactly one release_hash; returns it or raises."""
    hs = {p.get("release_hash") for p in points}
    if len(hs) != 1 or None in hs:
        raise ManifestError(f"one axis must carry exactly one release_hash, got {sorted(map(str, hs))}")
    return hs.pop()


# =========================================================================== MS1 form freeze
FORM_FREEZE_VERSION = "form-freeze-1"
FORM_FREEZE_STATUS = "awaiting_owner_hash_and_commit"
_PKG = Path(__file__).resolve().parent


def _file_shas(paths) -> dict[str, str]:
    return {p.relative_to(_PKG.parent).as_posix(): sha256_bytes(p.read_bytes()) for p in sorted(paths)}


def _component(fn) -> dict:
    """{sha256, detail} for one frozen form; an import/IO failure is recorded, never hidden."""
    try:
        detail = fn()
    except Exception as e:                       # a component owned by another worker may not exist yet
        return {"sha256": None, "error": f"{type(e).__name__}: {e}"}
    from .hashing import sha256_json
    return {"sha256": sha256_json(detail), "detail": detail}


def _ontology_table() -> dict:
    from .ontology import ARG_TYPES, ASC_MAJOR_MINOR_PATCH, CARD_CLASSES, ENTITY_TYPES, ENUMS, PREDICATES
    return {"predicates": [[p.name, list(p.args), p.fclass.value, p.strictness.value, p.anchor.value,
                            p.decidability.value, [list(x) for x in p.symmetric_pairs]] for p in PREDICATES.values()],
            "entity_types": list(ENTITY_TYPES), "enums": {k: list(v) for k, v in ENUMS.items()},
            # asc-2.0.0 (2026-09-26): the classes a card may assert (layout is extractor-internal)
            "version": ASC_MAJOR_MINOR_PATCH, "card_classes": sorted(c.value for c in CARD_CLASSES),
            # §7.1 typed argument positions (C41, 2026-09-26)
            "arg_types": {k: [sorted(d) if d is not None else None for d in v] for k, v in sorted(ARG_TYPES.items())}}


def _gate_forms() -> dict:
    import inspect
    from . import gates as G
    names = [n for n in sorted(vars(G)) if inspect.isfunction(getattr(G, n))
             and getattr(G, n).__module__ == G.__name__
             and (re.fullmatch(r"(g\d|ms\d_exit|books_ready)", n) or n in ("decide", "evaluate_all"))]
    fns = {n: sha256_bytes(inspect.getsource(getattr(G, n)).encode()) for n in names}
    return {"hmb/gates.py": sha256_bytes((_PKG / "gates.py").read_bytes()), "functions": fns,
            "acquisition_gates": _file_shas([_PKG / "acquisition" / "gates.py",
                                             *(_PKG / "acquisition" / "prompts").glob("*.txt")])}


def form_freeze_components() -> dict[str, dict]:
    def registry():
        from .config import registry_hash
        return {"registry_hash": registry_hash()}

    def rules():
        from .cards.closure import closure_spec
        return closure_spec()

    def normalization():
        from .cards.normalize import normalization_spec
        return normalization_spec()

    def judges():
        from .adjudication.judges import VERDICT_SCHEMA_VERSION, prompt_hashes, tiebreak_prompt_hash
        from .adjudication.serialize import serializer_identity
        return {"prompts": {**prompt_hashes(), **tiebreak_prompt_hash()}, "verdict_schema_version": VERDICT_SCHEMA_VERSION,
                "statement_serializer": serializer_identity()}   # code-built judge text (2026-09-26)

    def outcomes():
        from .vocab import OUTCOME_GROUPS, Outcome
        return {"outcomes": [o.value for o in Outcome],
                "groups": {k: sorted(x.value for x in v) for k, v in OUTCOME_GROUPS.items()}}

    def metric():
        from .grading.cell import AGGREGATION_RULES, _src_hashes
        return {"sources": _src_hashes(), "aggregation_rules": list(AGGREGATION_RULES)}

    return {
        "parameter_registry": _component(registry),
        "ontology_table": _component(_ontology_table),
        "rules_R_E": _component(rules),
        "normalization_and_confusables": _component(normalization),
        # templates plus the code that builds model-visible text around them (OCR/context blocks,
        # verification statements; the Track B / T3 context concatenation), 2026-09-26
        "builder_prompts": _component(lambda: _file_shas([*(_PKG / "cards" / "prompt_templates").glob("*.txt"),
                                                          _PKG / "cards" / "prompts.py", _PKG / "cards" / "verify.py"])),
        "contestant_prompts_and_wrapper": _component(lambda: _file_shas(
            [*(_PKG / "submission" / "prompts").glob("*.txt"), _PKG / "submission" / "wrapper_v1.tex",
             _PKG / "submission" / "prompts.py"])),
        "judge_prompts_and_verdict_schema": _component(judges),
        "outcome_vocabulary": _component(outcomes),
        "metric_code": _component(metric),
        "estimator_code": _component(lambda: _file_shas([_PKG / "stats.py"])),
        "gate_forms": _component(_gate_forms),
    }


def test_verdict_judge_hashes(layout: Layout) -> set:
    """judge_config_hash values of stored verdicts on test-split items. Development items judged by a pre-freeze
    run (MS2/MS3, provisional levels) are left out: the release binds the configuration of test judging only."""
    tdir, vdir = layout.p("data", "test"), layout.p("verdicts")
    test_ids = {p.stem for p in tdir.glob("*.json")} if tdir.exists() else set()
    return {v.get("judge_config_hash") for p in vdir.rglob("*.json")
            if (v := read_json(p)).get("item_id") in test_ids} if vdir.exists() else set()


def awaiting_level_freeze(layout: Layout) -> list[dict]:
    """Every calibrated/provisional parameter without a level-frozen value in thresholds.json (or no file yet)."""
    from .config import REGISTRY, load_thresholds
    th = load_thresholds(layout.thresholds) if layout.thresholds.exists() else None
    out = []
    for s in REGISTRY:
        if s.kind == "form_constant":
            continue
        st = (th.params.get(s.name) or {}).get("status") if th else None
        if st != "frozen":
            out.append({"name": s.name, "section": s.section, "kind": s.kind, "status": st or "no thresholds.json"})
    return out


def _approvals(layout: Layout) -> list[dict]:
    from .store import iter_jsonl
    return [r for r in iter_jsonl(layout.human_audit_log) if r.get("action") == "cp1_form_freeze_approval"]


def _form_freeze_sha(comps: dict[str, dict]) -> str:
    from .hashing import sha256_json
    return sha256_json({k: v["sha256"] for k, v in comps.items()})


def _approved_sha(row: dict) -> str:
    return row.get("description", "").split("form_freeze_sha256=")[-1]


def cp1_approval_problems(layout: Layout, level_frozen_at: str | None = None) -> list[str]:
    """MS1 is complete only when the owner has approved the forms AS THEY ARE NOW (the current code,
    prompts and registry hash to the approved form_freeze_sha256) and, when a level freeze exists,
    approved them before it (form freeze precedes level freeze, §17.3). Empty list = approved."""
    comps = form_freeze_components()
    incomplete = sorted(k for k, v in comps.items() if v["sha256"] is None)
    if incomplete:
        return [f"form-freeze components incomplete: {incomplete}"]
    sha = _form_freeze_sha(comps)
    rows = [r for r in _approvals(layout) if _approved_sha(r) == sha]
    if not rows:
        return [f"no MS1 owner approval of the current forms (form_freeze_sha256={sha[:12]}…): run `form-freeze`, "
                "review it, then `form-freeze --record-owner-approval --actor <name>`"]
    if level_frozen_at and min(r["utc"] for r in rows) > level_frozen_at:
        return ["the MS1 approval is later than the level freeze; the forms must be approved first (§17.3)"]
    return []


def write_form_freeze(layout: Layout, utc: str) -> dict:
    """manifests/form_freeze.json. Never claims MS1 is frozen: status stays awaiting the owner's
    hash-and-commit. Refuses to change content that an owner approval already references."""
    from .hashing import sha256_json
    comps = form_freeze_components()
    doc = {"artifact": "cp1_form_freeze", "version": FORM_FREEZE_VERSION, "created_utc": utc,
           "status": FORM_FREEZE_STATUS,
           "note": "MS1 is frozen only by the owner's hash-and-commit act; this file is its input.",
           "components": comps,
           "incomplete_components": sorted(k for k, v in comps.items() if v["sha256"] is None),
           "awaiting_level_freeze": awaiting_level_freeze(layout)}
    doc["form_freeze_sha256"] = _form_freeze_sha(comps)
    approved = {_approved_sha(r) for r in _approvals(layout)}
    if approved and doc["form_freeze_sha256"] not in approved:
        raise ManifestError("an owner approval references different forms; a form change after approval "
                            "needs a new evaluator version")
    write_mutable(layout.p("manifests", "form_freeze.json"),
                  (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode())
    return doc


def record_form_freeze_approval(layout: Layout, actor: str, utc: str) -> dict:
    """Append the owner's approval (HumanAction, scope protocol) for the current form_freeze.json."""
    from .schemas import HumanAction
    from .store import append_jsonl
    p = layout.p("manifests", "form_freeze.json")
    if not p.exists():
        raise ManifestError("manifests/form_freeze.json missing: run form-freeze first")
    if not actor or not actor.strip():
        raise ManifestError("--actor is required")
    doc = read_json(p)
    if doc.get("incomplete_components"):
        raise ManifestError(f"cannot approve an incomplete form freeze: {doc['incomplete_components']}")
    row = HumanAction(utc=utc, actor=actor.strip(), scope="protocol", action="cp1_form_freeze_approval",
                      description="owner approval of manifests/form_freeze.json form_freeze_sha256="
                                  + doc["form_freeze_sha256"]).model_dump(mode="json")
    append_jsonl(layout.human_audit_log, row)
    return row


# =========================================================================== release build
def _consensus_rule_hash(th) -> str:
    from .cards import consensus_rule_hash      # Opus 2 owns the consensus rule (§15.1 card_hash)
    return consensus_rule_hash()


def _eligible_fact_set(card) -> list:
    from .ontology import canonical_args
    out = []
    for f in card.facts:
        if f.eligible:
            args = canonical_args(f.predicate, f.args)
            out.append([f.predicate, [list(a) if isinstance(a, tuple) else a for a in args]])
    return sorted(out, key=lambda x: json.dumps(x, sort_keys=True))


def _placeholder(name: str) -> str:
    return H("fixture-placeholder", name)


def _harness_object(th) -> dict:
    """§15.2 harness object from Opus 3's parts; asserts the single §15.1 formula."""
    from .submission.process import harness_version as sub_hv, harness_version_parts
    hp = harness_version_parts()
    harness = {"eval_commit": hp["evaluation_code"], "container_digest": hp["sandbox_image_digest"],
               "wrapper_version": hp["wrapper_version"], "extraction_rules_version": hp["extraction_rules_version"],
               "rasterizer": hp["rasterizer"], "dpi": th.value("render_dpi_candidate"),
               "canvas": f"{th.value('canvas_px')}x{th.value('canvas_px')}"}
    harness["submission_harness_version_equal"] = sub_hv(th) == harness_version_of(harness)
    return harness


def build_release_from_root(layout: Layout, version: str, *, roster_path: str | None = None,
                            fixture: bool = False, created: str) -> dict:
    """§15.1 chain from the artifacts under `layout.root`; writes releases/<v>/release.json.

    Production: the owner must have approved the current forms (MS1) before the level freeze,
    thresholds.json must be production and level-frozen, frozen_at must predate the
    first-test-artifact marker, split groups must exist, and every chain input must be real.
    `fixture=True` builds an explicitly non-releasable release (placeholders listed)."""
    import yaml

    from .config import load_thresholds
    from .schemas import Card, ItemRecord
    from .store import iter_jsonl
    problems: list[str] = []
    placeholders: list[str] = []
    th_path = layout.thresholds if layout.thresholds.exists() else layout.p("thresholds.fixture.json")
    if not th_path.exists():
        raise ManifestError("no thresholds.json under the root")
    th = load_thresholds(th_path)
    if not fixture:
        if not th.is_production:
            problems.append("thresholds are not the production profile (use --fixture for engineering roots)")
        elif not th.frozen_at or th.missing_for_production():
            problems.append("thresholds.json is not level-frozen")
        if not layout.test_artifacts_exist():
            problems.append("no first-test-artifact marker: nothing to release")
        elif th.frozen_at and read_json(layout.first_test_artifact_marker)["utc"] <= th.frozen_at:
            problems.append("thresholds frozen_at does not predate the first test-book artifact")
        problems += cp1_approval_problems(layout, th.frozen_at)
    if not layout.split_groups.exists():
        problems.append("sources/split_groups.yaml missing")
    if problems:
        raise ManifestError("; ".join(problems))
    pairs, books, facts_eligible, n_items = [], set(), 0, 0
    crh = None                                   # consensus_rule_hash, computed once per release
    non_prod: list[str] = []
    for split in ("development", "test"):
        d = layout.p("data", split)
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            it = ItemRecord.model_validate(read_json(p))
            if not it.quality.eligible:
                continue
            card = Card.model_validate(read_json(layout.card(it.id)))
            if card.profile != "production":
                non_prod.append(f"card:{it.id}")
            page_b = layout.p("images", *Path(it.assets["page_image"]).parts).read_bytes()
            crop_b = layout.p("images", *Path(it.assets["crop_image"]).parts).read_bytes()
            ih = item_hash(crop_b, it.extraction.pipeline_version, page_hash(page_b), it.extraction.crop_bbox)
            efs = _eligible_fact_set(card)
            crh = crh or _consensus_rule_hash(th)
            pairs.append((ih, card_hash(efs, card.asc_version, crh)))
            books.add(it.source["book_id"])
            facts_eligible += len(efs)
            n_items += 1
    for kind in ("verdicts", "scores"):
        d = layout.p(kind)
        for p in sorted(d.rglob("*.json")) if d.exists() else []:
            if p.name == "run.json":
                continue
            prof = read_json(p).get("profile")
            if prof != "production":
                non_prod.append(f"{kind}:{p.relative_to(d).as_posix()}")
    if non_prod and not fixture:
        raise ManifestError(f"non-production artifacts cannot enter a production release: {non_prod}")
    groups = (yaml.safe_load(layout.split_groups.read_text(encoding="utf-8")) or {}).get("groups", {})
    hashes = {"cardset_hash": cardset_hash(pairs), "split_hash": file_hash(layout.split_groups),
              "thresholds_hash": file_hash(th_path)}
    try:
        if not roster_path:
            raise ManifestError("no --roster")
        from .adjudication.protocol import judge_config_hash as jch
        from .cgx.extract import EXTRACTOR_VERSION
        from .roster import load_roster
        hashes["judge_config_hash"] = jch(load_roster(roster_path)["judge"], EXTRACTOR_VERSION, th)
        seen = test_verdict_judge_hashes(layout)
        if seen - {hashes["judge_config_hash"]}:
            raise ManifestError("verdict files carry a different judge_config_hash")
    except Exception as e:
        if not fixture:
            raise ManifestError(f"judge_config_hash: {e}") from e
        hashes["judge_config_hash"] = _placeholder("judge_config_hash")
        placeholders.append(f"judge_config_hash ({e})")
    try:
        harness = _harness_object(th)
        if not harness["submission_harness_version_equal"] and not fixture:
            raise ManifestError("hmb.submission.process.harness_version(th) differs from the §15.1 value "
                                "computed from the harness object (one formula required)")
    except Exception as e:
        if not fixture:
            raise ManifestError(f"harness: {e}") from e
        harness = {k: f"fixture-placeholder:{k}" for k in HARNESS_KEYS}
        placeholders.append(f"harness ({type(e).__name__}: {e})")
    try:
        from .grading.cell import metric_hash as grading_metric_hash
        hashes["metric_hash"] = grading_metric_hash(th)
    except Exception as e:
        if not fixture:
            raise ManifestError(f"metric_hash: {e}") from e
        hashes["metric_hash"] = _placeholder("metric_hash")
        placeholders.append(f"metric_hash ({e})")
    from .sampling import funnel_table, stages_of_row
    latest: dict[str, dict] = {}
    for r in iter_jsonl(layout.funnel):
        latest[r["proposal_id"]] = r
    units = [{"stages": stages_of_row(r), "book_id": r["book_id"]} for r in latest.values()]
    funnel_rel = f"manifests/funnel_{version}.json"
    write_json_immutable(layout.p(*funnel_rel.split("/")), {"version": version, "overall": funnel_table(units),
                                                            "by_book": funnel_table(units, "book_id")})
    rel = build_release(version=version, created=created, hashes=hashes,
                        thresholds_frozen_at=th.frozen_at or ("fixture" if fixture else None), harness=harness,
                        counts={"books": len(books), "split_groups": len(groups), "items_eligible": n_items,
                                "facts_eligible": facts_eligible}, funnel_path=funnel_rel)
    rel["profile"] = "fixture" if fixture else "production"
    rel["releasable"] = not fixture
    if fixture:
        rel["placeholders"] = placeholders
        rel["label"] = "FIXTURE RELEASE - engineering only, not releasable, not a benchmark result"
    write_release(layout, rel)
    return rel


# =========================================================================== CLI
def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cmd_form_freeze(args) -> int:
    lay = Layout.from_env(args.root)
    try:
        if args.record_owner_approval:
            print(json.dumps(record_form_freeze_approval(lay, args.actor or "", _utc_now()), indent=1))
            return 0
        doc = write_form_freeze(lay, _utc_now())
    except ManifestError as e:
        print(f"form-freeze refused: {e}")
        return 2
    print(json.dumps({"path": str(lay.p("manifests", "form_freeze.json")), "status": doc["status"],
                      "form_freeze_sha256": doc["form_freeze_sha256"],
                      "incomplete_components": doc["incomplete_components"],
                      "n_awaiting_level_freeze": len(doc["awaiting_level_freeze"])}, indent=1))
    return 0


def _cmd_release_build(args) -> int:
    lay = Layout.from_env(args.root)
    try:
        rel = build_release_from_root(lay, args.version, roster_path=args.roster, fixture=args.fixture,
                                      created=_utc_now())
    except ManifestError as e:
        print(f"release-build refused: {e}")
        return 2
    print(json.dumps({"release_hash": rel["release_hash"], "releasable": rel["releasable"],
                      "counts": rel["counts"], "placeholders": rel.get("placeholders", [])}, indent=1))
    return 0


def register_cli(sub) -> None:
    p = sub.add_parser("form-freeze", help="write the MS1 form-freeze artifact (never claims MS1 frozen)")
    p.add_argument("--record-owner-approval", action="store_true")
    p.add_argument("--actor", default=None)
    p.set_defaults(func=_cmd_form_freeze)
    p = sub.add_parser("release-build", help="compute the §15.1 chain and write manifests/release.json")
    p.add_argument("--version", required=True)
    p.add_argument("--roster", default=None)
    p.add_argument("--fixture", action="store_true", help="engineering root: non-releasable release")
    p.set_defaults(func=_cmd_release_build)


__all__ = ["page_hash", "item_hash", "card_hash", "cardset_hash", "file_hash",
           "harness_version", "harness_version_of", "release_hash", "build_release", "verify_release", "write_release",
           "build_run_manifest", "write_run_manifest", "add_tombstone", "load_tombstones", "assert_id_usable",
           "VERSION_TRIGGERS", "version_bump", "next_version", "check_plot_axis", "sha256_bytes",
           "write_form_freeze", "record_form_freeze_approval", "form_freeze_components", "cp1_approval_problems",
           "build_release_from_root"]
