"""DA: does the unanimity key drop hard facts? Read-only on repo."""
import collections, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
D = Path(__import__("os").environ.get("HMB_ROOT", REPO / "hmb_data"))
OUT = Path(__file__).with_name("da_key_out.json")
items = []
for p in sorted((D / "data" / "test").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if r["quality"]["eligible"]:
        items.append(r["id"])
union = collections.Counter(); elig = collections.Counter(); reasons = collections.Counter()
supp = {}  # (i,fid) -> 'both_generated' | 'one_generated'
pred_union = collections.Counter(); pred_elig = collections.Counter()
for i in items:
    c = json.loads((D / "semantic_cards" / f"{i}.json").read_text(encoding="utf-8"))
    for f in c["facts"]:
        k = f["class"]
        union[k] += 1; pred_union[f["predicate"]] += 1
        if f["eligible"]:
            elig[k] += 1; pred_elig[f["predicate"]] += 1
            g = sum(v == "generated" for v in f["votes"].values())
            supp[(i, f["fact_id"])] = (k, "both_generated" if g >= 2 else "one_generated")
        else:
            for r in f["ineligible_reasons"]:
                reasons[(k, r)] += 1
res = {"union_by_class": union, "eligible_by_class": elig,
       "retention_by_class": {k: elig[k] / union[k] for k in union},
       "ineligible_reasons": {f"{a}|{b}": v for (a, b), v in reasons.most_common()},
       "support_mix": collections.Counter(f"{a}|{b}" for a, b in supp.values()),
       "relation_pred_retention": {p: [pred_elig[p], pred_union[p], round(pred_elig[p] / pred_union[p], 3)]
                                   for p in sorted(pred_union, key=lambda x: -pred_union[x])}}
models = [p.name for p in (D / "verdicts").iterdir()]
res["present_by_support"] = {}
for m in models:
    cnt = collections.Counter()
    for i in items:
        vp = D / "verdicts" / m / "A" / f"{i}.json"
        if not vp.exists():
            continue
        for fa in json.loads(vp.read_text(encoding="utf-8"))["facts"]:
            key = supp.get((i, fa["fact_id"]))
            if key is None or fa["verdict"] == "EXEMPT":
                continue
            cnt[key + ("n",)] += 1; cnt[key + ("p",)] += fa["verdict"] == "PRESENT"
    res["present_by_support"][m] = {f"{c}|{s}": round(cnt[(c, s, "p")] / cnt[(c, s, "n")], 3)
                                    for c in ("entity", "relation", "label") for s in ("both_generated", "one_generated") if cnt[(c, s, "n")]}
OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
print(json.dumps(res, indent=1, default=float, ensure_ascii=False))
