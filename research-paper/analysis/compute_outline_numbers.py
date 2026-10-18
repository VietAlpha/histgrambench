"""Read-only computation of the outline's [compute] numbers from stored HistMathBench artifacts.
No model call, no write under the repository. Output: JSON next to this script."""
import collections, glob, json, os, sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hmb import stats  # noqa: E402
from hmb.config import load_thresholds  # noqa: E402

D = Path(__import__("os").environ.get("HMB_ROOT", REPO / "hmb_data"))
TH = load_thresholds(D / "thresholds.json")
ALPHA = float(TH.value("alpha"))
OUT = Path(__file__).with_name("outline_numbers.json")

SR = json.loads((D / "reports" / "scores_test_histmathbench-1.0.0.json").read_text(encoding="utf-8"))
MODELS = sorted(SR["tables"]["A"]["models"])
print(MODELS, file=sys.stderr)
SHORT = {"google.gemma-4-26b-a4b": "Gemma 4 26B", "us.meta.llama4-maverick-17b-instruct-v1:0": "Llama 4 Maverick",
         "mistral.ministral-3-14b-instruct": "Ministral 3 14B", "google.gemma-4-e2b": "Gemma 4 E2B",
         "global.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5",
         "nvidia.nemotron-nano-12b-v2": "Nemotron Nano 12B v2"}
SHORT.update({k.replace(":", "_"): v for k, v in list(SHORT.items())})

# ---- items, groups, context, year
items, group, ctx, year, nfacts = [], {}, {}, {}, {}
rights = {b["book_id"]: b for b in yaml.safe_load((D / "sources" / "rights_manifest.yaml").read_text(encoding="utf-8"))}
for p in sorted((D / "data" / "test").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if not r["quality"]["eligible"]:
        continue
    i = r["id"]
    items.append(i)
    group[i] = r["split_group"]
    ctx[i] = int((r.get("extraction") or {}).get("context_char_count", 0))
    year[i] = rights[r["source"]["book_id"]]["publication_year"]
assert len(items) == SR["tables"]["A"]["N"], (len(items), SR["tables"]["A"]["N"])
cards = {i: json.loads((D / "semantic_cards" / f"{i}.json").read_text(encoding="utf-8")) for i in items}
for i in items:
    nfacts[i] = sum(f["eligible"] for f in cards[i]["facts"])


def model_dir(m):
    return m.replace(":", "_")


cells = {t: {m: {} for m in MODELS} for t in "AB"}
for t in "AB":
    for m in MODELS:
        for i in items:
            cells[t][m][i] = json.loads((D / "scores" / model_dir(m) / t / f"{i}.json").read_text(encoding="utf-8"))


def gest(values, its):
    by = collections.defaultdict(list)
    for i in its:
        by[group[i]].append(values[i])
    gm = [float(np.mean(v)) for v in by.values()]
    im = stats.im_group_t(gm, alpha=ALPHA)
    return {"estimate": float(np.mean(gm)), "ci": im.get("ci"), "p": im.get("p_value"), "G": len(gm), "n": len(its)}


def S(t, m):
    return {i: float(cells[t][m][i]["S"] or 0.0) for i in items}


res = {}
# check: headline reproduces the report
res["check_headline"] = {SHORT[m]: [round(gest(S("A", m), items)["estimate"], 4),
                                    round(SR["tables"]["A"]["models"][m]["headline"]["estimate"], 4)] for m in MODELS}

# F2: H restricted to valid outputs
res["H_valid"] = {}
for t in "AB":
    for m in MODELS:
        ok = [i for i in items if cells[t][m][i]["outcome_code"] == "OK"]
        res["H_valid"][f"{SHORT[m]}|{t}"] = gest(S(t, m), ok)

# F3: per-model paired B-A (all items), on items with text, per-class B-A
with_text = [i for i in items if ctx[i] > 0]
res["n_with_text"] = len(with_text)
res["paired_BA_all"], res["paired_BA_text"] = {}, {}
for m in MODELS:
    a, b = S("A", m), S("B", m)
    d = {i: b[i] - a[i] for i in items}
    res["paired_BA_all"][SHORT[m]] = gest(d, items)
    res["paired_BA_text"][SHORT[m]] = gest(d, with_text)
for name, its in (("pooled_BA_all", items), ("pooled_BA_text", with_text), ("pooled_BA_notext", [i for i in items if ctx[i] == 0])):
    d = {i: float(np.mean([S("B", m)[i] - S("A", m)[i] for m in MODELS])) for i in its}
    res[name] = gest(d, its)
res["H_text_only_items"] = {f"{SHORT[m]}|{t}": gest(S(t, m), with_text)["estimate"] for m in MODELS for t in "AB"}


def class_recall(t, m, c, its):
    rows = [cells[t][m][i]["per_class"][c] for i in its if cells[t][m][i]["per_class"][c]["reportable"]]
    return float(np.mean([r["R"] or 0.0 for r in rows])) if rows else None


res["per_class_BA"] = {}
for m in MODELS:
    res["per_class_BA"][SHORT[m]] = {c: {"A": class_recall("A", m, c, items), "B": class_recall("B", m, c, items),
                                         "B_minus_A": class_recall("B", m, c, items) - class_recall("A", m, c, items)}
                                     for c in "ERL"}
# per-class B-A among items valid in both tracks (content change, not validity change)
res["per_class_BA_valid_both"] = {}
for m in MODELS:
    vb = [i for i in items if cells["A"][m][i]["outcome_code"] == "OK" and cells["B"][m][i]["outcome_code"] == "OK"]
    res["per_class_BA_valid_both"][SHORT[m]] = {"n": len(vb), **{c: class_recall("B", m, c, vb) - class_recall("A", m, c, vb) for c in "ERL"},
                                                "S_B_minus_A": float(np.mean([S("B", m)[i] - S("A", m)[i] for i in vb]))}

# F4: drop largest group; item-weighted; period strata
sizes = collections.Counter(group[i] for i in items)
big = max(sizes, key=sizes.get)
res["largest_group"] = {"id": big, "n": sizes[big]}
keep = [i for i in items if group[i] != big]
res["H_drop_largest"] = {f"{SHORT[m]}|{t}": gest(S(t, m), keep) for m in MODELS for t in "AB"}
full_A = [SR["tables"]["A"]["models"][m]["headline"]["estimate"] for m in MODELS]
res["tau_drop_largest_A"] = stats.kendall_tau_b(full_A, [res["H_drop_largest"][f"{SHORT[m]}|A"]["estimate"] for m in MODELS])
res["item_weighted"] = {f"{SHORT[m]}|{t}": float(np.mean(list(S(t, m).values()))) for m in MODELS for t in "AB"}

bins = {"before 1850": lambda y: y < 1850, "1850-1930": lambda y: 1850 <= y <= 1930, "after 1930": lambda y: y > 1930}
res["period"] = {}
for name, f in bins.items():
    its = [i for i in items if f(year[i])]
    g = len({group[i] for i in its})
    nf = sum(nfacts[i] for i in its)
    publish = len(its) >= 30 and g >= 3 and nf >= 150
    res["period"][name] = {"n": len(its), "groups": g, "facts": nf,
                           "publishable": publish,
                           "H": {f"{SHORT[m]}|{t}": (gest(S(t, m), its)["estimate"] if publish else None) for m in MODELS for t in "AB"}}

# F5: routing of judged facts
model_class = {}
for i in items:
    for f in cards[i]["facts"]:
        model_class[(i, f["fact_id"])] = f.get("decidability") != "geo"
res["routing"] = {}
for t in "AB":
    for m in MODELS:
        cnt = collections.Counter()
        for i in items:
            vp = D / "verdicts" / model_dir(m) / t / f"{i}.json"
            if not vp.exists():
                continue
            v = json.loads(vp.read_text(encoding="utf-8"))
            over = (json.loads((D / "cgx" / model_dir(m) / t / f"{i}.json").read_text(encoding="utf-8"))
                    .get("primitives_over_cap"))
            for fa in v["facts"]:
                cnt["all"] += 1
                if fa.get("source") != "judges":
                    continue
                cnt["judged"] += 1
                if model_class.get((i, fa["fact_id"])):
                    cnt["judge_only_type"] += 1
                elif over:
                    cnt["over_cap"] += 1
                else:
                    cnt["unbound_object"] += 1
        res["routing"][f"{SHORT[m]}|{t}"] = {**cnt, "judged_share": cnt["judged"] / cnt["all"],
                                             **{k: cnt[k] / cnt["judged"] for k in ("judge_only_type", "over_cap", "unbound_object")}}

OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
print(json.dumps(res, indent=1, default=float))

# addendum: per-model B-A and validity change on items with no page text
no_text = [i for i in items if ctx[i] == 0]
res["paired_BA_notext"] = {SHORT[m]: gest({i: S("B", m)[i] - S("A", m)[i] for i in items}, no_text) for m in MODELS}
res["dvalid"] = {SHORT[m]: {k: (sum(cells["B"][m][i]["outcome_code"] == "OK" for i in its)
                                 - sum(cells["A"][m][i]["outcome_code"] == "OK" for i in its)) / len(its)
                            for k, its in (("text", with_text), ("notext", no_text))} for m in MODELS}
res["groups_text_notext_overlap"] = len({group[i] for i in with_text} & {group[i] for i in no_text})
OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
