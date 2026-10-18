"""R1 read-only check: exact group-weighted loss decomposition, EXEMPT shares, PRESENT rate by routing class,
item-weighted vs group-weighted Q5. Reads hmb_data only; writes JSON to the review folder."""
import collections, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
D = Path(__import__("os").environ.get("HMB_ROOT", REPO / "hmb_data"))
OUT = Path(__file__).with_name("r1_decomp_check.json")
SR = json.loads((D / "reports" / "scores_test_histmathbench-1.0.0.json").read_text(encoding="utf-8"))
MODELS = sorted(SR["tables"]["A"]["models"])
group, items = {}, []
for p in sorted((D / "data" / "test").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if r["quality"]["eligible"]:
        items.append(r["id"]); group[r["id"]] = r["split_group"]
cards = {i: json.loads((D / "semantic_cards" / f"{i}.json").read_text(encoding="utf-8")) for i in items}
judge_type = {(i, f["fact_id"]): f.get("decidability") != "geo" for i in items for f in cards[i]["facts"]}
groups = sorted(set(group.values()))
res = {}
for m in MODELS:
    md = m.replace(":", "_")
    inval_g = collections.defaultdict(list)   # per group: 1 - g_i
    miss_g = collections.defaultdict(list)    # per group: g_i * (1 - S_i)
    S_g = collections.defaultdict(list)
    ex = collections.Counter()
    route = collections.Counter()
    for i in items:
        sc = json.loads((D / "scores" / md / "A" / f"{i}.json").read_text(encoding="utf-8"))
        g = 1 if sc["outcome_code"] == "OK" else 0
        S = float(sc["S"] or 0.0)
        inval_g[group[i]].append(1 - g); miss_g[group[i]].append(g * (1 - S)); S_g[group[i]].append(S)
        if not g:
            continue
        for c in "ERL":
            ex["n_" + c] += sc["per_class"][c]["n"]; ex["exempt_" + c] += sc["per_class"][c]["n_exempt"]
        vp = D / "verdicts" / md / "A" / f"{i}.json"
        cg = json.loads((D / "cgx" / md / "A" / f"{i}.json").read_text(encoding="utf-8"))
        over = cg.get("primitives_over_cap")
        v = json.loads(vp.read_text(encoding="utf-8"))
        for fv in v["facts"]:
            if fv.get("source") != "judges":
                cls = "geo_decided"
            elif judge_type.get((i, fv["fact_id"])):
                cls = "judge_only_type"
            elif over:
                cls = "over_cap"
            else:
                cls = "unbound"
            route[cls + "|all"] += 1
            route[cls + "|" + fv["verdict"]] += 1
            if fv["fact_id"].startswith("e_"):
                route[cls + "|entity_fact"] += 1
                route[cls + "|entity_fact|" + fv["verdict"]] += 1
    H = np.mean([np.mean(S_g[k]) for k in groups])
    inv = np.mean([np.mean(inval_g[k]) for k in groups])
    miss = np.mean([np.mean(miss_g[k]) for k in groups])
    res[m] = {"H": H, "loss": 1 - H, "invalid_part": inv, "valid_miss_part": miss, "check": 1 - H - inv - miss,
              "exempt_share": {c: ex["exempt_" + c] / ex["n_" + c] for c in "ERL"},
              "route": dict(route),
              "present_rate": {k: route[k + "|PRESENT"] / route[k + "|all"] for k in
                               ("geo_decided", "judge_only_type", "over_cap", "unbound") if route[k + "|all"]},
              "entity_present_rate_unbound": (route["unbound|entity_fact|PRESENT"] / route["unbound|entity_fact"]
                                              if route["unbound|entity_fact"] else None)}
OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
for m, v in res.items():
    print(m[:30], {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "route"})
