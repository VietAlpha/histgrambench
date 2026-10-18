"""R1 read-only check: forced tie-break share, single-judge identity, forced-as-not-recovered H,
BH vs BY on Track A pairs. Reads hmb_data only; writes JSON to the review folder."""
import collections, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hmb import stats  # noqa: E402

D = Path(__import__("os").environ.get("HMB_ROOT", REPO / "hmb_data"))
OUT = Path(__file__).with_name("r1_forced_check.json")
SR = json.loads((D / "reports" / "scores_test_histmathbench-1.0.0.json").read_text(encoding="utf-8"))
MODELS = sorted(SR["tables"]["A"]["models"])
group = {}
items = []
for p in sorted((D / "data" / "test").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if r["quality"]["eligible"]:
        items.append(r["id"]); group[r["id"]] = r["split_group"]


def gest(vals, its):
    by = collections.defaultdict(list)
    for i in its:
        by[group[i]].append(vals[i])
    gm = [float(np.mean(v)) for v in by.values()]
    im = stats.im_group_t(gm)
    return {"est": float(np.mean(gm)), "ci": im.get("ci"), "p": im.get("p_value"), "G": len(gm)}


res = {"models": {}}
S_full, S_forced0, S_forced1 = {}, {}, {}
for m in MODELS:
    md = m.replace(":", "_")
    cnt = collections.Counter()
    fam_first = collections.Counter()
    sf, s0, s1 = {}, {}, {}
    for i in items:
        sc = json.loads((D / "scores" / md / "A" / f"{i}.json").read_text(encoding="utf-8"))
        S = float(sc["S"] or 0.0)
        sf[i] = S
        s0[i] = S
        s1[i] = S
        if sc["outcome_code"] != "OK":
            continue
        scorable = sum(sc["per_class"][c]["n_star"] for c in "ERL")
        vp = D / "verdicts" / md / "A" / f"{i}.json"
        if not vp.exists() or scorable == 0:
            continue
        v = json.loads(vp.read_text(encoding="utf-8"))
        forced_present = forced_notpresent = 0
        for fv in v["facts"]:
            cnt["all"] += 1
            if fv.get("source") != "judges":
                continue
            cnt["judged"] += 1
            votes = fv.get("votes") or []
            fams = [x["judge_family"] for x in votes]
            cnt["nvotes_%d" % len(votes)] += 1
            if votes:
                fam_first[fams[0]] += 1
            ev = fv.get("evidence") or ""
            if ev.startswith("forced"):
                cnt["forced"] += 1
                if fv["verdict"] == "PRESENT":
                    forced_present += 1
                else:
                    forced_notpresent += 1
            # does the final verdict equal the first listed judge's answer?
            if votes and fv["verdict"] in ("PRESENT", "ABSENT", "CONTRADICTED"):
                first_pos = votes[0]["positive"]
                cnt["eq_first"] += int((fv["verdict"] == "PRESENT") == bool(first_pos))
                cnt["decided_with_votes"] += 1
        # forced -> not recovered (lower bound); forced -> recovered (upper bound)
        s0[i] = S - forced_present / scorable
        s1[i] = S + forced_notpresent / scorable
    S_full[m], S_forced0[m], S_forced1[m] = sf, s0, s1
    res["models"][m] = {
        "judged": cnt["judged"], "forced": cnt["forced"],
        "forced_share_of_judged": cnt["forced"] / cnt["judged"] if cnt["judged"] else None,
        "verdict_equals_first_listed_judge_share": cnt["eq_first"] / cnt["decided_with_votes"] if cnt["decided_with_votes"] else None,
        "first_listed_family_counts": dict(fam_first),
        "vote_count_hist": {k: v for k, v in cnt.items() if k.startswith("nvotes_")},
        "H_full": gest(sf, items)["est"],
        "H_forced_as_not_recovered": gest(s0, items)["est"],
        "H_forced_as_recovered": gest(s1, items)["est"],
    }

# pairwise under forced-as-not-recovered, BH and BY
def pairs(Sd):
    out = {}
    for x in range(len(MODELS)):
        for y in range(x + 1, len(MODELS)):
            a, b = MODELS[x], MODELS[y]
            g = gest({i: Sd[a][i] - Sd[b][i] for i in items}, items)
            out[(a, b)] = {"p": g["p"], "diff": g["est"]}
    return out

def bh_by(pr):
    keys = list(pr)
    p = np.array([pr[k]["p"] for k in keys])
    m = len(p)
    bh = stats.benjamini_hochberg(p, 0.10)
    cm = sum(1.0 / k for k in range(1, m + 1))
    by = stats.benjamini_hochberg(np.minimum(p * cm, 1.0), 0.10)
    return {f"{a}|{b}": {"diff": round(pr[(a, b)]["diff"], 4), "p": pr[(a, b)]["p"],
                         "bh_adj": bh["adjusted"][j], "bh_rej": bh["rejected"][j],
                         "by_adj": by["adjusted"][j], "by_rej": by["rejected"][j]}
            for j, (a, b) in enumerate(keys)}, cm

res["pairs_full"], res["by_c_m"] = bh_by(pairs(S_full))
res["pairs_forced_as_not_recovered"], _ = bh_by(pairs(S_forced0))
res["pairs_forced_as_recovered"], _ = bh_by(pairs(S_forced1))
OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
print(json.dumps(res, indent=1, default=float))
