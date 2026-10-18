"""R1 read-only: H with a model-independent denominator (EXEMPT counted as not recovered) and per-class
recall under the same rule; Track A. Writes JSON to the review folder only."""
import collections, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hmb import stats
D = Path(__import__("os").environ.get("HMB_ROOT", REPO / "hmb_data"))
SR = json.loads((D / "reports" / "scores_test_histmathbench-1.0.0.json").read_text(encoding="utf-8"))
MODELS = sorted(SR["tables"]["A"]["models"])
group, items = {}, []
for p in sorted((D / "data" / "test").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if r["quality"]["eligible"]:
        items.append(r["id"]); group[r["id"]] = r["split_group"]
def gest(vals):
    by = collections.defaultdict(list)
    for i in items: by[group[i]].append(vals[i])
    gm = [float(np.mean(v)) for v in by.values()]
    im = stats.im_group_t(gm)
    return float(np.mean(gm)), im.get("p_value")
Sfix, res = {}, {}
for m in MODELS:
    md = m.replace(":", "_"); s = {}; cls = collections.defaultdict(list)
    for i in items:
        sc = json.loads((D / "scores" / md / "A" / f"{i}.json").read_text(encoding="utf-8"))
        g = sc["outcome_code"] == "OK"
        n = sum(sc["per_class"][c]["n"] for c in "ERL")
        npres = sum(sc["per_class"][c]["n_present"] for c in "ERL") if g else 0
        s[i] = npres / n
        for c in "ERL":
            pc = sc["per_class"][c]
            if pc["n"] > 0:
                cls[c].append((pc["n_present"] / pc["n"]) if g else 0.0)
    Sfix[m] = s
    res[m] = {"H_fixed_den": gest(s)[0], "H_report": SR["tables"]["A"]["models"][m]["headline"]["estimate"],
              "class_fixed_den_item_mean": {c: float(np.mean(v)) for c, v in cls.items()}}
pairs = {}
keys = []
for x in range(len(MODELS)):
    for y in range(x + 1, len(MODELS)):
        a, b = MODELS[x], MODELS[y]
        est, p = gest({i: Sfix[a][i] - Sfix[b][i] for i in items})
        keys.append(f"{a}|{b}"); pairs[f"{a}|{b}"] = {"diff": est, "p": p}
bh = stats.benjamini_hochberg([pairs[k]["p"] for k in keys], 0.10)
for k, rej, adj in zip(keys, bh["rejected"], bh["adjusted"]):
    pairs[k].update(bh_adj=adj, sep=rej)
res["pairs"] = pairs
Path("r1_fixed_den.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
for m in MODELS: print(m[:28], {k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in res[m].items()})
for k, v in pairs.items(): print(k.replace('global.anthropic.','').replace('us.meta.','')[:70], round(v['diff'],4), '%.3g'%v['p'], round(v['bh_adj'],4), v['sep'])
