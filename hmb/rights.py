"""Source rights, provenance, and the R3 extraction gate (plan §4).

What code may do here (R1): compute mechanical fields (hashes, byte count, page count, codec,
bit depth, colour space), the arithmetic of the named rule `us-pd-pre1931-v1`, and the R2
defaults. `release_mode`, `approved_by`, the terms booleans, and every determination other
than the named rule are human-set; code never writes them.

R3 (§4.5) is value-checked: every one of the five conditions is tested on its value, never
on presence. The production gate refuses any `fixture:` approver; the fixture profile accepts
one so engineering tests can run on real pages without pretending a book is approved.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .hashing import book_id as make_book_id, sha256_bytes, sha256_file, sha256_json
from .schemas import RightsManifest
from .store import Layout, write_immutable, write_mutable
from .vocab import Profile, ReleaseMode, UsStatus, YesNoUnknown

RULE_ID = "us-pd-pre1931-v1"
RULE_LAST_PD_YEAR = 1930          # §4.1: published before 1 January 1931
MANIFEST_VERSION = "rights-1"
FIXTURE_APPROVER_PREFIX = "fixture:"

# §4.7 benchmark-card sentence (mandatory, verbatim).
BENCHMARK_CARD_SENTENCE = (
    "Rights status was determined once per source book. The determination covers United States "
    "copyright status only and rests on the first-publication year evidenced by the volume's own "
    "title page and catalog record together with the term rules summarised in U.S. Copyright Office "
    "Circulars 15A and 38B. Status in the source country and in any other jurisdiction was not "
    "determined. Repository terms of use are recorded separately from copyright status and reproduced "
    "verbatim in `sources/rights_manifest.yaml`; where a repository conditions reuse, that condition "
    "applies to the corresponding items independently of their public-domain status.")

PROVENANCE_COLUMNS = (
    "book_id", "work_id", "display_title", "author", "publisher", "publication_year", "published",
    "language", "first_publication_country", "source_repository", "source_url", "source_file_sha256",
    "source_codec", "source_bitdepth", "source_colorspace", "us_status", "us_rule_id",
    "release_mode", "redistribution_permitted", "commercial_reuse_permitted", "ai_use_clause",
    "required_attribution_string", "approved_by", "approved_utc", "manifest_sha256")


class RightsError(ValueError):
    pass


# --------------------------------------------------------------------------- §4.1 rule
def us_status_by_rule(first_publication_year: Any, published: Any) -> str:
    """Rule us-pd-pre1931-v1: public_domain iff evidenced year <= 1930 AND published == yes.

    Anything else (unknown year, non-integer year, unpublished, unknown publication) is
    `not_determined`. The rule never yields `in_copyright`; that is a human determination.
    """
    if isinstance(first_publication_year, bool) or not isinstance(first_publication_year, int):
        return UsStatus.not_determined.value
    if str(published) != YesNoUnknown.yes.value:
        return UsStatus.not_determined.value
    if first_publication_year <= RULE_LAST_PD_YEAR:
        return UsStatus.public_domain.value
    return UsStatus.not_determined.value


def apply_us_rule(record: dict) -> dict:
    """Return a copy of a manifest record with `determination.us_status` set by the named rule.

    The rule reads only the evidenced year: when `publication_year_evidence.agree` is not True
    the year is not evidenced and the result is `not_determined`. The R2 fields are reset to
    their only legal value. `manifest_sha256` is recomputed.
    """
    out = _deepcopy(record)
    ev = (out.get("publication_year_evidence") or {})
    year = out.get("publication_year") if ev.get("agree") is True else None
    det = dict(out.get("determination") or {})
    det.update(us_status=us_status_by_rule(year, out.get("published")), us_rule_id=RULE_ID,
               us_determined_by="rule", source_country_status="not_determined",
               release_country_status="not_determined", determined_by_model=False)
    det.setdefault("source_country", out.get("first_publication_country"))
    out["determination"] = det
    out["manifest_sha256"] = manifest_sha256(out)
    return out


# --------------------------------------------------------------------------- hashing
def manifest_sha256(record: dict) -> str:
    """§4.5: hash of the record with the `manifest_sha256` field removed (RFC 8785 JSON)."""
    body = {k: v for k, v in record.items() if k != "manifest_sha256"}
    return sha256_json(body)


# --------------------------------------------------------------------------- R1 mechanical draft
def probe_master(path: str | Path) -> dict:
    """Mechanical acquisition fields of a master file. Raises RightsError on unsupported files."""
    path = Path(path)
    out: dict[str, Any] = {"source_file_sha256": sha256_file(path),
                           "source_file_bytes": path.stat().st_size}
    if path.suffix.lower() == ".pdf":
        import fitz  # pymupdf
        try:
            doc = fitz.open(path)
        except Exception as e:  # corrupt or not a PDF
            raise RightsError(f"{path.name}: cannot open PDF ({e})") from e
        with doc:
            out["page_count"] = doc.page_count
            counts: dict[tuple[int, str], int] = {}
            for pno in range(doc.page_count):
                for img in doc.get_page_images(pno, full=True):
                    bpc, cs = int(img[4]), str(img[5])
                    key = _pdf_image_kind(bpc, cs, _icc_components(doc, img[0]) if cs == "ICCBased" else None)
                    if key:
                        counts[key] = counts.get(key, 0) + 1
        out["source_codec"] = "pdf_mixed"
        if counts:
            (bd, space), _ = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
            out["source_bitdepth"], out["source_colorspace"] = bd, space
            out["image_kind_counts"] = {f"{k[1]}/{k[0]}": v for k, v in sorted(counts.items())}
        else:
            out["source_bitdepth"] = out["source_colorspace"] = None
        return out
    from PIL import Image
    try:
        with Image.open(path) as im:
            fmt, mode = im.format, im.mode
            compression = im.info.get("compression")
            n = getattr(im, "n_frames", 1)
    except Exception as e:
        raise RightsError(f"{path.name}: cannot open image ({e})") from e
    codec = {"JPEG": "jpeg", "PNG": "png", "JPEG2000": "jp2"}.get(fmt or "")
    if fmt == "TIFF" and compression == "group4":
        codec = "tiff_g4"
    if codec is None:
        raise RightsError(f"{path.name}: codec {fmt}/{compression} is outside the §4.5 codec list")
    kind = {"1": (1, "bitonal"), "L": (8, "gray"), "LA": (8, "gray"), "RGB": (24, "rgb"),
            "RGBA": (24, "rgb"), "P": (24, "rgb"), "CMYK": (24, "rgb"), "YCbCr": (24, "rgb")}.get(mode)
    out.update(page_count=n, source_codec=codec,
               source_bitdepth=kind[0] if kind else None, source_colorspace=kind[1] if kind else None)
    return out


def _icc_components(doc, xref: int) -> int | None:
    """Number of colour components of an ICCBased image colour space (/N of the ICC stream)."""
    import re as _re
    try:
        kind, val = doc.xref_get_key(xref, "ColorSpace")
        if kind == "xref":                       # indirect: resolve the colour-space array object
            val = doc.xref_object(int(val.split()[0]))
        m = _re.search(r"/ICCBased\s+(\d+)\s+0\s+R", val or "")
        if not m:
            return None
        kind, n = doc.xref_get_key(int(m.group(1)), "N")
        return int(n) if kind == "int" else None
    except Exception:
        return None


def _pdf_image_kind(bpc: int, cs: str, icc_n: int | None = None) -> tuple[int, str] | None:
    if bpc == 1:
        return (1, "bitonal")
    if cs == "ICCBased":
        return {1: (8, "gray"), 3: (24, "rgb"), 4: (24, "rgb")}.get(icc_n or 0)
    if "Gray" in cs:
        return (8, "gray")
    if "RGB" in cs or "CMYK" in cs or "ICC" in cs or "Lab" in cs:
        return (24, "rgb")
    return None


def draft_manifest(master_path: str | Path, *, source_repository: str | None = None,
                   source_url: str | None = None, download_utc: str | None = None,
                   scanning_agent: str | None = None, probe: dict | None = None) -> dict:
    """R1 draft: mechanical fields filled, every human field left null, R2 defaults set.
    `probe` replaces `probe_master` (a pre-cropped source: `probe_precropped`).

    The draft is deliberately NOT a valid RightsManifest: approval, terms booleans, the year
    evidence comparison, and `release_mode` are for a human. `determination.us_status` is
    `not_determined` until `apply_us_rule` runs on an approved, evidenced record.
    """
    probe = probe or probe_master(master_path)
    bid = make_book_id(probe["source_file_sha256"])
    rec: dict[str, Any] = {
        "book_id": bid, "work_id": None, "display_title": None, "author": None, "publisher": None,
        "publisher_id": None, "edition_statement": None, "is_reprint_of_earlier_work": None,
        "publication_year": None, "published": None, "publication_place": None,
        "first_publication_country": None, "language": None,
        "acquisition": {
            "source_repository": source_repository, "source_url": source_url,
            "download_utc": download_utc, "source_file_sha256": probe["source_file_sha256"],
            "source_file_bytes": probe["source_file_bytes"], "page_count": probe["page_count"],
            "source_codec": probe["source_codec"], "source_bitdepth": probe["source_bitdepth"],
            "source_colorspace": probe["source_colorspace"], "scanning_agent": scanning_agent,
            "source_dpi_nominal": None},
        "publication_year_evidence": {
            "catalog_record": None, "catalog_record_sha256": None, "titlepage_asset": None,
            "titlepage_asset_sha256": None, "titlepage_page_index": None, "agree": None},
        "terms": {"terms_url": None, "terms_snapshot": None, "terms_snapshot_sha256": None,
                  "terms_snapshot_utc": None, "repository_rights_statement_verbatim": None,
                  "redistribution_permitted": None, "commercial_reuse_permitted": None,
                  "ai_use_clause": None, "required_attribution_string": None},
        "determination": {"us_status": UsStatus.not_determined.value, "us_rule_id": RULE_ID,
                          "us_determined_by": "rule", "source_country": None,
                          "source_country_status": "not_determined",
                          "release_country_status": "not_determined",
                          "determined_by_model": False},
        "prominence": {"hosted_by_major_aggregators": None, "public_fulltext_layer": None},
        "release_mode": None, "approved_by": None, "approved_utc": None,
        "manifest_version": MANIFEST_VERSION,
    }
    rec["manifest_sha256"] = manifest_sha256(rec)
    return rec


# --------------------------------------------------------------------------- R3 gate
def r3_gate(record: dict, profile: Profile | str) -> dict:
    """§4.5 R3 extraction gate. Returns {passed, failures[], profile}.

    Pages of a book are rendered only if all of: the record validates as a RightsManifest,
    its manifest_sha256 matches, us_status == public_domain, release_mode != excluded,
    publication_year_evidence.agree is True, terms.redistribution_permitted != no, and
    approved_by is a non-empty string. Production refuses a `fixture:` approver; the fixture
    profile requires one (a real approval never enters a fixture run by accident).
    """
    profile = Profile(profile)
    failures: list[str] = []
    if not isinstance(record, dict):
        return {"passed": False, "failures": ["record_not_a_mapping"], "profile": profile.value}
    try:
        RightsManifest.model_validate(record)
    except ValidationError as e:
        failures.append("schema_invalid: " + "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()))
    if record.get("manifest_sha256") != manifest_sha256(record):
        failures.append("manifest_sha256_mismatch")
    det = record.get("determination") if isinstance(record.get("determination"), dict) else {}
    ev = record.get("publication_year_evidence") if isinstance(
        record.get("publication_year_evidence"), dict) else {}
    terms = record.get("terms") if isinstance(record.get("terms"), dict) else {}
    if det.get("us_status") != UsStatus.public_domain.value:
        failures.append(f"us_status={det.get('us_status')!r}")
    if record.get("release_mode") not in {m.value for m in ReleaseMode} or \
            record.get("release_mode") == ReleaseMode.excluded.value:
        failures.append(f"release_mode={record.get('release_mode')!r}")
    if ev.get("agree") is not True:
        failures.append(f"year_evidence_agree={ev.get('agree')!r}")
    if terms.get("redistribution_permitted") not in {"yes", "unknown"}:
        failures.append(f"redistribution_permitted={terms.get('redistribution_permitted')!r}")
    approver = record.get("approved_by")
    if not isinstance(approver, str) or not approver.strip():
        failures.append("approved_by_unset")
    elif profile == Profile.production and approver.strip().startswith(FIXTURE_APPROVER_PREFIX):
        failures.append("fixture_approver_in_production")
    elif profile == Profile.fixture and not approver.strip().startswith(FIXTURE_APPROVER_PREFIX):
        failures.append("fixture_profile_requires_fixture_approver")
    return {"passed": not failures, "failures": failures, "profile": profile.value}


# --------------------------------------------------------------------------- manifest file
def load_manifests(path: str | Path) -> dict[str, dict]:
    """Read `sources/rights_manifest.yaml` (a YAML list of records). Duplicate book ids refuse."""
    path = Path(path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, list):
        raise RightsError(f"{path}: expected a YAML list of manifest records")
    out: dict[str, dict] = {}
    for rec in data:
        if not isinstance(rec, dict) or not isinstance(rec.get("book_id"), str):
            raise RightsError(f"{path}: record without book_id")
        if rec["book_id"] in out:
            raise RightsError(f"{path}: duplicate book_id {rec['book_id']}")
        out[rec["book_id"]] = rec
    return out


# --------------------------------------------------------------------------- §4.6 evidence bundle
def write_evidence(layout: Layout, book: str, name: str, data: bytes) -> dict:
    """Store one evidence file immutably under sources/evidence/<book_id>/ and index it."""
    if "/" in name or "\\" in name or name.startswith("."):
        raise RightsError(f"evidence name must be a plain file name, got {name!r}")
    path = layout.evidence_dir(book) / name
    digest = write_immutable(path, data)
    rel = f"evidence/{book}/{name}"
    entry = {"path": rel, "sha256": digest, "bytes": len(data)}
    index_path = layout.evidence_dir(book) / "index.json"
    index = _read_json_or(index_path, {})
    if index.get(name) not in (None, entry):
        raise RightsError(f"{rel}: index already records different content")
    index[name] = entry
    import json
    write_mutable(index_path, (json.dumps(index, indent=1, sort_keys=True) + "\n").encode())
    return entry


def write_master_evidence(layout: Layout, master_path: str | Path) -> dict:
    """Evidence item (1): hash and byte count of the master download."""
    p = Path(master_path)
    rec = {"file_name": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size}
    import json
    return write_evidence(layout, make_book_id(rec["sha256"]), "master.json",
                          (json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode())


def probe_precropped(snap: dict) -> dict:
    """Mechanical acquisition fields of a pre-cropped source: the master is its image listing
    (`collection.precropped_listing`); codec, bit depth and colour space are the most common over its images."""
    import io

    from PIL import Image

    from .acquisition import collection as col
    listing = col.read_verified(snap, "", master=True)
    codecs: dict[str, int] = {}
    kinds: dict[tuple, int] = {}
    for rel in snap["precropped"]["items"]:
        try:
            with Image.open(io.BytesIO(col.read_verified(snap, rel))) as im:
                fmt, mode = im.format, im.mode
        except col.SnapshotChanged:
            raise
        except Exception:
            continue
        c = {"JPEG": "jpeg", "PNG": "png", "JPEG2000": "jp2"}.get(fmt or "")
        if c:
            codecs[c] = codecs.get(c, 0) + 1
        k = {"1": (1, "bitonal"), "L": (8, "gray"), "LA": (8, "gray"), "RGB": (24, "rgb"), "RGBA": (24, "rgb"),
             "P": (24, "rgb"), "CMYK": (24, "rgb")}.get(mode)
        if k:
            kinds[k] = kinds.get(k, 0) + 1
    if not codecs or not kinds:
        raise RightsError(f"{snap['source_name']}: no decodable png/jpeg image")
    kind = max(kinds.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return {"source_file_sha256": sha256_bytes(listing), "source_file_bytes": len(listing),
            "page_count": len(snap["precropped"]["items"]),
            "source_codec": max(codecs.items(), key=lambda kv: (kv[1], kv[0]))[0],
            "source_bitdepth": kind[0], "source_colorspace": kind[1]}


def write_precropped_master_evidence(layout: Layout, snap: dict) -> dict:
    """Evidence item (1) for a pre-cropped source: the image listing (its master) and master.json naming the
    source directory, which `build` runs as `acq-run <file_name>`."""
    import json

    from .acquisition import collection as col
    listing = col.read_verified(snap, "", master=True)
    bid = make_book_id(sha256_bytes(listing))
    write_evidence(layout, bid, "listing.json", listing)
    rec = {"file_name": snap["source_name"], "sha256": sha256_bytes(listing), "bytes": len(listing),
           "kind": "precropped_listing", "listing": f"evidence/{bid}/listing.json"}
    return write_evidence(layout, bid, "master.json",
                          (json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode())


def precropped_titlepage_png(snap: dict) -> bytes:
    """Title-page asset of a pre-cropped source: its first image, grayscale lossless PNG."""
    from .acquisition import collection as col
    from .acquisition.render import encode_png
    return encode_png(col.load_crop_gray(snap, snap["precropped"]["items"][0]))


def render_titlepage_png(master_path: str | Path, page_index: int, dpi: int) -> bytes:
    """Automatic title-page asset (§4.5): the page rendered grayscale, lossless PNG."""
    from .acquisition.render import encode_png, load_page_gray
    gray, _meta = load_page_gray(master_path, page_index, dpi)
    return encode_png(gray)


def verify_evidence(layout: Layout, record: dict) -> list[str]:
    """Re-hash every evidence file named in a manifest; return a list of problems (empty = ok)."""
    problems: list[str] = []
    pairs = [((record.get("publication_year_evidence") or {}).get("catalog_record"),
              (record.get("publication_year_evidence") or {}).get("catalog_record_sha256")),
             ((record.get("publication_year_evidence") or {}).get("titlepage_asset"),
              (record.get("publication_year_evidence") or {}).get("titlepage_asset_sha256")),
             ((record.get("terms") or {}).get("terms_snapshot"),
              (record.get("terms") or {}).get("terms_snapshot_sha256"))]
    for rel, want in pairs:
        if not rel or not want:
            problems.append(f"missing evidence reference {rel!r}")
            continue
        p = layout.p("sources", *Path(rel).parts)
        if not p.exists():
            problems.append(f"{rel}: file missing")
        elif sha256_file(p) != want:
            problems.append(f"{rel}: sha256 mismatch")
    return problems


# --------------------------------------------------------------------------- provenance view
def provenance_rows(manifests: dict[str, dict]) -> list[dict]:
    rows = []
    for bid in sorted(manifests):
        r = manifests[bid]
        acq, det, terms = r.get("acquisition") or {}, r.get("determination") or {}, r.get("terms") or {}
        flat = {**{k: r.get(k) for k in ("book_id", "work_id", "display_title", "author", "publisher",
                                         "publication_year", "published", "language",
                                         "first_publication_country", "release_mode", "approved_by",
                                         "approved_utc", "manifest_sha256")},
                **{k: acq.get(k) for k in ("source_repository", "source_url", "source_file_sha256",
                                           "source_codec", "source_bitdepth", "source_colorspace")},
                "us_status": det.get("us_status"), "us_rule_id": det.get("us_rule_id"),
                **{k: terms.get(k) for k in ("redistribution_permitted", "commercial_reuse_permitted",
                                             "ai_use_clause", "required_attribution_string")}}
        rows.append({c: ("" if flat.get(c) is None else flat[c]) for c in PROVENANCE_COLUMNS})
    return rows


def write_provenance_csv(layout: Layout, manifests: dict[str, dict] | None = None) -> str:
    """Regenerate `sources/provenance.csv` from the manifest file (§15.6: never hand-edited)."""
    manifests = load_manifests(layout.rights_manifest) if manifests is None else manifests
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(PROVENANCE_COLUMNS), lineterminator="\n")
    w.writeheader()
    for row in provenance_rows(manifests):
        w.writerow(row)
    return write_mutable(layout.provenance_csv, buf.getvalue().encode("utf-8"))


# --------------------------------------------------------------------------- helpers
def _deepcopy(x):
    import copy
    return copy.deepcopy(x)


def _read_json_or(path: Path, default):
    import json
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def dump_yaml(obj: Any) -> bytes:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).encode("utf-8")


# --------------------------------------------------------------------------- CLI
def _cmd_draft(args) -> int:
    lay = Layout.from_env(args.root)
    rec = draft_manifest(args.master, source_repository=args.repository, source_url=args.url,
                         download_utc=args.download_utc)
    entry = write_evidence(lay, rec["book_id"], "manifest_draft.yaml", dump_yaml(rec))
    write_master_evidence(lay, args.master)
    print(f"{rec['book_id']} draft -> sources/{entry['path']} (human fields left null)")
    return 0


def _cmd_check(args) -> int:
    import json
    lay = Layout.from_env(args.root)
    mans = load_manifests(lay.rights_manifest)
    out = {bid: r3_gate(rec, args.profile) for bid, rec in mans.items()}
    evidence = {bid: verify_evidence(lay, rec) for bid, rec in mans.items()}      # re-hash every evidence file
    bad = sorted(b for b in out if out[b]["passed"] and evidence[b])
    print(json.dumps({"n_books": len(out), "n_pass": sum(v["passed"] for v in out.values()),
                      "results": out, "evidence_problems": {b: v for b, v in evidence.items() if v},
                      "approved_books_with_evidence_problems": bad}, indent=1))
    return 2 if bad else 0


def _cmd_provenance(args) -> int:
    lay = Layout.from_env(args.root)
    print(f"wrote {lay.provenance_csv} sha256={write_provenance_csv(lay)}")
    return 0


def register_cli(sub) -> None:
    p = sub.add_parser("rights-draft", help="R1 mechanical manifest draft + master evidence (§4.5-4.6)")
    p.add_argument("master")
    p.add_argument("--repository", default=None)
    p.add_argument("--url", default=None)
    p.add_argument("--download-utc", default=None)
    p.set_defaults(func=_cmd_draft)
    p = sub.add_parser("rights-check", help="R3 gate over sources/rights_manifest.yaml and the evidence re-hash "
                                            "(exit 2 when an approved book's evidence is missing or changed)")
    p.add_argument("--profile", choices=["fixture", "production"], default="production")
    p.set_defaults(func=_cmd_check)
    p = sub.add_parser("rights-provenance", help="regenerate sources/provenance.csv")
    p.set_defaults(func=_cmd_provenance)


__all__ = ["RULE_ID", "BENCHMARK_CARD_SENTENCE", "us_status_by_rule", "apply_us_rule",
           "manifest_sha256", "probe_master", "draft_manifest", "r3_gate", "load_manifests",
           "write_evidence", "write_master_evidence", "verify_evidence",
           "render_titlepage_png", "write_provenance_csv", "provenance_rows"]
