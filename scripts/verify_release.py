"""Integrity checks over a local copy of the HistGramBench dataset and this repository. Offline; no network.

    python scripts/verify_release.py --dataset DIR [--repo .] [--out report.json]

Checks: every configuration loads with the Hugging Face `datasets` library from the dataset card's own YAML;
images decode and match their sha256; native records match their sha256; identifiers are unique; joins between
items, works, split groups, answer keys, answers, verdicts and scores are complete; coverage equals the paper's
counts; the development/test split keeps every source group, work, book, hard duplicate edge and near-duplicate
cluster on one side; every figure has its image and every work its rights basis; mirrored files are identical.
Exit status 0 when every check passes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

EXPECT = {"test_items": 2507, "test_works": 95, "test_groups": 87, "test_key_facts": 114403, "models": 6,
          "tracks": 2, "test_answers": 30084, "development_items": 150, "development_works": 5}


def sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8") if isinstance(b, str) else b).hexdigest()


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def check(self, name: str, ok: bool, detail: object = "") -> None:
        try:
            detail = json.loads(json.dumps(detail, default=str))
        except TypeError:                     # e.g. dict keys that are tuples
            detail = str(detail)
        self.checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail != "" else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from datasets import load_dataset
    global CACHE
    CACHE = tempfile.mkdtemp(prefix="hgb_verify_cache_")
    from PIL import Image
    R = Report()
    D = a.dataset

    card = (D / "README.md").read_text(encoding="utf-8")
    meta = yaml.safe_load(card.split("---")[1])
    configs = {c["config_name"]: c for c in meta["configs"]}
    R.check("dataset card YAML parses and declares configs", bool(configs), sorted(configs))
    on_disk = {p.parent.name for p in (D / "data").glob("*/*.parquet")}
    R.check("every data folder is a declared config", on_disk == set(configs), sorted(on_disk ^ set(configs)))

    ds = {}
    for name in configs:
        ds[name] = load_dataset(str(D), name, cache_dir=CACHE)   # fresh cache: never a stale earlier build
        R.check(f"load_dataset(..., {name!r})", True, {s: ds[name][s].num_rows for s in ds[name]})

    def rows(cfg, split, cols=None):
        d = ds[cfg][split]
        return d.select_columns(cols) if cols else d

    # ------------------------------------------------------------------ items
    items = {}
    img_bad, img_n, rec_bad = [], 0, []
    for split in ds["items"]:
        d = ds["items"][split].cast_column("image", ds["items"][split].features["image"].__class__(decode=False))
        d = d.cast_column("original_crop", d.features["original_crop"].__class__(decode=False))
        for r in d:
            items[r["item_id"]] = {**{k: r[k] for k in ("split", "split_group", "book_id", "work_id", "scored",
                                                       "rights_basis", "n_key_facts", "has_page_text",
                                                       "phash_cluster_id", "page_text")}}
            if sha(r["item_record_json"]) != r["item_record_sha256"]:
                rec_bad.append(r["item_id"])
            for col, scol in (("image", "image_sha256"), ("original_crop", "original_crop_sha256")):
                if r[col] is not None:
                    img_n += 1
                    b = r[col]["bytes"]
                    try:
                        with Image.open(io.BytesIO(b)) as im:
                            im.load()
                            if col == "image" and (im.width, im.height) != (r["image_width"], r["image_height"]):
                                img_bad.append(f"{r['item_id']}:{col}:size")
                    except Exception as e:  # noqa: BLE001
                        img_bad.append(f"{r['item_id']}:{col}:{e}")
                    if sha(b) != r[scol]:
                        img_bad.append(f"{r['item_id']}:{col}:sha256")
            if r["page_text"] is not None and sha(r["page_text"]) != r["page_text_sha256"]:
                img_bad.append(f"{r['item_id']}:page_text:sha256")
    R.check("item ids unique", len(items) == sum(ds["items"][s].num_rows for s in ds["items"]))
    R.check("item records match sha256", not rec_bad, rec_bad[:5])
    R.check("images decode, match sha256 and stated size", not img_bad, f"{img_n} images; bad {img_bad[:5]}")
    by = defaultdict(list)
    for i, r in items.items():
        by[r["split"]].append(i)
    test = set(by["test"])
    R.check("test items = 2,507", len(test) == EXPECT["test_items"], len(test))
    R.check("development items = 150", len(by["development"]) == EXPECT["development_items"], len(by["development"]))
    R.check("only test and development items", set(by) <= {"test", "development"}, sorted(by))
    R.check("test works = 95", len({items[i]["book_id"] for i in test}) == EXPECT["test_works"])
    R.check("test source groups = 87", len({items[i]["split_group"] for i in test}) == EXPECT["test_groups"])
    R.check("development works = 5", len({items[i]["book_id"] for i in by["development"]}) == EXPECT["development_works"])
    R.check("key facts on test items = 114,403", sum(items[i]["n_key_facts"] for i in test) == EXPECT["test_key_facts"])
    R.check("every item has its model-input image and original crop", img_n == 2 * len(items), f"{img_n} images")
    no_text = [i for i, r in items.items() if r["has_page_text"] and not r["page_text"]]
    R.check("page text present wherever the item had page text", not no_text, no_text[:5])
    R.check("every item has a rights basis", all(r["rights_basis"] for r in items.values()),
            dict(Counter(r["rights_basis"] for r in items.values())))

    # ------------------------------------------------------------------ works and groups
    works = {r["book_id"]: r for r in ds["works"]["records"]}
    R.check("every item's book is in works", all(r["book_id"] in works for r in items.values()))
    R.check("works rights basis agrees with items", all(works[r["book_id"]]["rights_basis"] == r["rights_basis"] for r in items.values()))
    groups = {r["split_group"]: r for r in ds["split_groups"]["records"]}
    R.check("every item's split group is in split_groups", all(r["split_group"] in groups for r in items.values()))
    side = {"test": "test", "development": "development"}
    R.check("item split equals its group's split", all(groups[r["split_group"]]["split"] == side[r["split"]]
                                                       for r in items.values()))
    for key in ("book_id", "work_id", "split_group"):
        cross = defaultdict(set)
        for r in items.values():
            cross[r[key]].add(side[r["split"]])
        bad = [k for k, v in cross.items() if len(v) > 1]
        R.check(f"no {key} on both sides of the split", not bad, bad[:5])
    clusters = defaultdict(set)
    for r in items.values():
        if r["phash_cluster_id"] and r["phash_cluster_id"] != "singleton":
            clusters[r["phash_cluster_id"]].add(side[r["split"]])
    bad = [k for k, v in clusters.items() if len(v) > 1]
    R.check("no near-duplicate cluster on both sides", not bad, f"{len(clusters)} clusters; crossing {bad[:5]}")
    sg = yaml.safe_load((D / "metadata" / "split_groups.yaml").read_text(encoding="utf-8"))
    g_of = {b: g for g, v in sg["groups"].items() for b in v["books"]}
    bad = [e for e in sg["hard_edges"] if g_of.get(e["a"]) != g_of.get(e["b"])]
    R.check("every hard edge (duplicate / work identity) inside one group", not bad,
            f"{len(sg['hard_edges'])} edges; soft-edge crossings recorded by design: {len(sg['soft_edge_crossings'])}")

    # ------------------------------------------------------------------ answer keys
    nk = Counter()
    fk_bad = []
    seen = set()
    for split in ds["answer_key_facts"]:
        for r in rows("answer_key_facts", split, ["item_id", "fact_id", "in_key"]):
            k = (r["item_id"], r["fact_id"])
            if k in seen or r["item_id"] not in items:
                fk_bad.append(k)
            seen.add(k)
            nk[r["item_id"]] += r["in_key"]
    R.check("fact ids unique and joined to items", not fk_bad, fk_bad[:5])
    R.check("facts in the key equal items.n_key_facts", all(nk[i] == items[i]["n_key_facts"] for i in items))
    kr_bad = []
    for split in ds["answer_key_records"]:
        for r in ds["answer_key_records"][split]:
            for col in ("card", "generation_manifest", "card_gate"):
                if r[f"{col}_json"] is not None and sha(r[f"{col}_json"]) != r[f"{col}_sha256"]:
                    kr_bad.append(f"{r['item_id']}:{col}")
    R.check("answer-key records match sha256", not kr_bad, kr_bad[:5])

    # ------------------------------------------------------------------ run records
    def cells(cfg, split):
        return {(r["model_dir"], r["track"], r["item_id"]): r for r in ds[cfg][split]}
    ans = cells("answers", "test")
    R.check("answers on test = 30,084 (2,507 x 6 models x 2 tracks)", len(ans) == EXPECT["test_answers"], len(ans))
    R.check("answers cover every (model, track, test item)",
            {(m, t, i) for m in {k[0] for k in ans} for t in ("A", "B") for i in test} == set(ans))
    nat_bad = []
    for k, r in ans.items():
        for col in ("submission_record", "submission_stage", "extraction", "compile", "raw_output", "normalized_tex"):
            v = r[col] if col in ("raw_output", "normalized_tex") else r[f"{col}_json"]
            if v is not None and sha(v) != r[f"{col}_sha256"]:
                nat_bad.append(f"{k}:{col}")
    R.check("answer records match sha256", not nat_bad, nat_bad[:5])
    no_raw = [k for k, r in ans.items() if r["raw_output"] is None and r["raw_output_sha256"] is not None]
    R.check("every stored raw answer is present", not no_raw, no_raw[:3])
    oc = Counter(r["outcome_code"] for r in ans.values())
    R.check("outcome codes recorded (invalid outcomes kept)", True, dict(oc))
    ver = cells("verdicts", "test")
    ok = {k for k, r in ans.items() if r["outcome_code"] == "OK"}
    R.check("a verdict file exists exactly for every valid drawing", set(ver) == ok, f"{len(ver)} verdicts, {len(ok)} OK")
    v_bad = [k for k, r in ver.items() if sha(r["verdict_json"]) != r["verdict_sha256"] or
             (r["alignment_json"] is not None and sha(r["alignment_json"]) != r["alignment_sha256"])]
    R.check("verdict and alignment records match sha256", not v_bad, v_bad[:5])
    cg = {(r["model_dir"], r["track"], r["item_id"]): r["cgx_sha256"] for r in rows("cgx", "test", ["model_dir", "track", "item_id", "cgx_sha256"])}
    R.check("CGX geometry exists for every valid drawing", set(cg) >= ok, f"{len(cg)} records")
    pdfs = {(r["model_dir"], r["track"], r["item_id"]) for r in rows("compiled_pdfs", "test", ["model_dir", "track", "item_id"])}
    R.check("a compiled PDF exists for every valid drawing", pdfs >= ok, f"{len(pdfs)} PDFs, {len(ok)} OK")
    sc = cells("scores", "test")
    R.check("scores join answers one to one", set(sc) == set(ans))
    R.check("score outcome equals answer outcome", all(sc[k]["outcome_code"] == ans[k]["outcome_code"] for k in sc))

    # ------------------------------------------------------------------ mirrors and canonical files in the repository
    canon = {}
    with gzip.open(a.repo / "results" / "score_cells.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            canon[d["path"]] = d
    mirror_bad = []
    n_m = 0
    for split in ds["scores"]:
        for r in ds["scores"][split]:
            n_m += 1
            c = canon.get(f"scores/{r['model_dir']}/{r['track']}/{r['item_id']}.json")
            if c is None or c["content"] != r["score_json"] or c["sha256"] != r["score_sha256"] or sha(r["score_json"]) != r["score_sha256"]:
                mirror_bad.append(r["item_id"])
    R.check("dataset `scores` mirrors repository results/score_cells.jsonl.gz", not mirror_bad and n_m == len(canon),
            f"{n_m} rows vs {len(canon)} canonical")
    for rel in ("sources/split_groups.yaml", "manifests/release.json"):
        m = D / "metadata" / Path(rel).name
        R.check(f"metadata/{m.name} identical to repository frozen/{rel}",
                m.read_bytes() == (a.repo / "frozen" / rel).read_bytes())
    n_pass = sum(c["pass"] for c in R.checks)
    print(f"{n_pass} of {len(R.checks)} checks pass")
    if a.out:
        a.out.write_text(json.dumps(R.checks, indent=1, default=str), encoding="utf-8")
    import shutil
    shutil.rmtree(CACHE, ignore_errors=True)
    return 0 if n_pass == len(R.checks) else 1


if __name__ == "__main__":
    sys.exit(main())
