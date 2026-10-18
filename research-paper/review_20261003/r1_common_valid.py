"""R1 read-only: Haiku vs Llama (and all pairs) on items valid for both models, Track A, group-balanced."""
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
S, ok = {}, {}
for m in MODELS:
    md = m.replace(":", "_"); S[m] = {}; ok[m] = set()
    for i in items:
        sc = json.loads((D / "scores" / md / "A" / f"{i}.json").read_text(encoding="utf-8"))
        S[m][i] = float(sc["S"] or 0.0)
        if sc["outcome_code"] == "OK": ok[m].add(i)
def gest(vals, its):
    by = collections.defaultdict(list)
    for i in its: by[group[i]].append(vals[i])
    gm = [float(np.mean(v)) for v in by.values()]
    im = stats.im_group_t(gm)
    return {"est": float(np.mean(gm)), "ci": im.get("ci"), "p": im.get("p_value"), "G": len(gm), "n": len(its)}
out = {}
for x in range(len(MODELS)):
    for y in range(x + 1, len(MODELS)):
        a, b = MODELS[x], MODELS[y]
        cv = sorted(ok[a] & ok[b])
        out[f"{a}|{b}"] = {"H_valid_common_a": gest(S[a], cv)["est"], "H_valid_common_b": gest(S[b], cv)["est"],
                           "diff": gest({i: S[a][i] - S[b][i] for i in cv}, cv)}
Path("r1_common_valid.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k, v in out.items():
    d = v["diff"]; print(k.replace('global.anthropic.','').replace('us.meta.','')[:70], round(v["H_valid_common_a"],3), round(v["H_valid_common_b"],3), round(d["est"],3), [round(c,3) for c in d["ci"]], d["G"], d["n"])
