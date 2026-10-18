"""DA counter-hypothesis tests. Read-only on the repo; writes JSON to the review scratch folder only."""
import collections, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hmb import stats
D = Path(__import__("os").environ.get("HMB_ROOT", REPO / "hmb_data"))
OUT = Path(__file__).with_name("da_judge_out.json")
SR = json.loads((D / "reports" / "scores_test_histmathbench-1.0.0.json").read_text(encoding="utf-8"))
MODELS = sorted(SR["tables"]["A"]["models"])
SH = {"google.gemma-4-26b-a4b": "G26", "us.meta.llama4-maverick-17b-instruct-v1:0": "MAV",
      "mistral.ministral-3-14b-instruct": "MIN", "google.gemma-4-e2b": "E2B",
      "global.anthropic.claude-haiku-4-5-20251001-v1:0": "HAI", "nvidia.nemotron-nano-12b-v2": "NEM"}
md = lambda m: m.replace(":", "_")
SH.update({k.replace(":", "_"): v for k, v in list(SH.items())})
items, group, ctx, frame, cls = [], {}, {}, {}, {}
for p in sorted((D / "data" / "test").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if not r["quality"]["eligible"]:
        continue
    i = r["id"]; items.append(i); group[i] = r["split_group"]
    ex = r.get("extraction") or {}
    ctx[i] = int(ex.get("context_char_count", 0))
    frame[i] = (ex.get("frame") or (ex.get("source_crop") or {}).get("frame") or "?")
elig = {}
for i in items:
    c = json.loads((D / "semantic_cards" / f"{i}.json").read_text(encoding="utf-8"))
    for f in c["facts"]:
        if f["eligible"]:
            elig[(i, f["fact_id"])] = f["fact_id"][0].upper()  # e_/r_/l_ prefix guess
            cls[(i, f["fact_id"])] = f.get("klass") or f.get("class") or f["fact_id"].split("_")[0]

def gmean(vals):
    by = collections.defaultdict(list)
    for i, v in vals.items():
        by[group[i]].append(v)
    gm = [float(np.mean(v)) for v in by.values()]
    return float(np.mean(gm)), gm

res = {"frame_counts": collections.Counter(frame.values()),
       "frame_by_ctx": collections.Counter((frame[i], ctx[i] > 0) for i in items)}
res["frame_by_ctx"] = {f"{k[0]}|text={k[1]}": v for k, v in res["frame_by_ctx"].items()}
variants = ["stored", "rebuilt", "geo_only", "judged_absent", "forced_absent", "forced_excluded"]
H = {v: {} for v in variants}
src_stats = {}
recon_err = {}
for t in "AB":
    for m in MODELS:
        Sv = {v: {} for v in variants}
        cnt = collections.Counter(); errs = 0
        for i in items:
            sc = json.loads((D / "scores" / md(m) / t / f"{i}.json").read_text(encoding="utf-8"))
            Sv["stored"][i] = float(sc["S"] or 0.0)
            if sc["outcome_code"] != "OK":
                for v in variants[1:]:
                    Sv[v][i] = 0.0
                continue
            vf = json.loads((D / "verdicts" / md(m) / t / f"{i}.json").read_text(encoding="utf-8"))
            num = collections.Counter(); den = collections.Counter()
            for fa in vf["facts"]:
                if (i, fa["fact_id"]) not in elig:
                    continue
                if fa["verdict"] == "EXEMPT":
                    cnt["exempt"] += 1
                    continue
                pres = fa["verdict"] == "PRESENT"
                judged = fa.get("source") == "judges"
                forced = str(fa.get("evidence") or "").startswith("forced")
                key = "forced" if forced else ("judged" if judged else "geo")
                cnt[key + "_n"] += 1; cnt[key + "_present"] += pres
                den["rebuilt"] += 1; num["rebuilt"] += pres
                den["judged_absent"] += 1; num["judged_absent"] += pres and not judged
                den["forced_absent"] += 1; num["forced_absent"] += pres and not forced
                if not forced:
                    den["forced_excluded"] += 1; num["forced_excluded"] += pres
                if not judged:
                    den["geo_only"] += 1; num["geo_only"] += pres
            for v in variants[1:]:
                Sv[v][i] = num[v] / den[v] if den[v] else 0.0
            if abs(Sv["rebuilt"][i] - Sv["stored"][i]) > 1e-6:
                errs += 1
        recon_err[f"{SH[m]}|{t}"] = errs
        for v in variants:
            H[v][f"{SH[m]}|{t}"] = gmean(Sv[v])[0]
        src_stats[f"{SH[m]}|{t}"] = {k: cnt[k + "_present"] / cnt[k + "_n"] if cnt[k + "_n"] else None for k in ("geo", "judged", "forced")} | {
            k + "_n": cnt[k + "_n"] for k in ("geo", "judged", "forced")}
        if t == "A":
            res.setdefault("_S", {})[SH[m]] = Sv
res["recon_mismatch_items"] = recon_err
res["present_rate_by_source"] = src_stats
res["H"] = H
# rankings and tau vs stored
order = lambda d, t: sorted([k for k in d if k.endswith("|" + t)], key=lambda k: -d[k])
res["rank_A"] = {v: order(H[v], "A") for v in variants}
for v in variants[2:]:
    res.setdefault("tau_A", {})[v] = stats.kendall_tau_b([H["stored"][f"{SH[m]}|A"] for m in MODELS], [H[v][f"{SH[m]}|A"] for m in MODELS])
# paired group t for key pair diffs under variants
pairs = [("G26", "MAV"), ("HAI", "E2B"), ("MIN", "E2B"), ("HAI", "MAV")]
Sd = res.pop("_S")
res["pairs"] = {}
for a, b in pairs:
    for v in ["stored", "geo_only", "forced_absent", "forced_excluded"]:
        d = {i: Sd[a][v][i] - Sd[b][v][i] for i in items}
        est, gm = gmean(d)
        im = stats.im_group_t(gm, alpha=0.05)
        res["pairs"][f"{a}-{b}|{v}"] = {"d": est, "ci": im.get("ci"), "p": im.get("p_value")}
OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("frame_counts", "frame_by_ctx", "recon_mismatch_items", "present_rate_by_source", "rank_A", "tau_A", "pairs")}, indent=1, default=float))
for v in variants:
    print(v, {k: round(x, 3) for k, x in H[v].items() if k.endswith("|A")})
