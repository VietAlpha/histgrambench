"""DA: run timing of Track A vs B; text-free group structure; weighting sensitivity. Read-only."""
import collections, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hmb import stats
D = Path(__import__("os").environ.get("HMB_ROOT", REPO / "hmb_data"))
OUT = Path(__file__).with_name("da_time_out.json")
items, group, ctx = [], {}, {}
for p in sorted((D / "data" / "test").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if r["quality"]["eligible"]:
        i = r["id"]; items.append(i); group[i] = r["split_group"]
        ctx[i] = int((r.get("extraction") or {}).get("context_char_count", 0))
models = sorted(p.name for p in (D / "submissions").iterdir() if p.is_dir())
res = {"timing": {}, "harness": {}, "same_day_split": {}}
notext = [i for i in items if ctx[i] == 0]; text = [i for i in items if ctx[i] > 0]
gs_nt = collections.Counter(group[i] for i in notext)
gs_all = collections.Counter(group[i] for i in items)
res["notext_groups"] = {"G": len(gs_nt), "singletons_in_notext": sum(1 for g, n in gs_nt.items() if n == 1),
                        "notext_group_sizes_sorted": sorted(gs_nt.values(), reverse=True)}
res["text_groups"] = sorted(collections.Counter(group[i] for i in text).values(), reverse=True)
for m in models:
    day = {}; hv = collections.Counter()
    for t in "AB":
        for i in items:
            s = json.loads((D / "submissions" / m / t / f"{i}.json").read_text(encoding="utf-8"))
            u = s["attempts"][-1]["utc"] if s.get("attempts") else None
            day[(t, i)] = (u or "")[:10]
            hv[(t, s.get("harness_version", "")[:8])] += 1
    res["timing"][m] = {t: dict(collections.Counter(day[(t, i)] for i in items)) for t in "AB"}
    res["harness"][m] = {f"{a}|{b}": v for (a, b), v in hv.items()}
    # paired B-A on no-text items, split by same-day vs different-day A/B
    Sa = {i: float(json.loads((D / "scores" / m / "A" / f"{i}.json").read_text(encoding="utf-8"))["S"] or 0) for i in items}
    Sb = {i: float(json.loads((D / "scores" / m / "B" / f"{i}.json").read_text(encoding="utf-8"))["S"] or 0) for i in items}
    same = [i for i in notext if day[("A", i)] == day[("B", i)]]
    diff = [i for i in notext if day[("A", i)] != day[("B", i)]]
    res["same_day_split"][m] = {"n_same": len(same), "n_diff": len(diff),
                                "item_mean_BA_same": float(np.mean([Sb[i] - Sa[i] for i in same])) if same else None,
                                "item_mean_BA_diff": float(np.mean([Sb[i] - Sa[i] for i in diff])) if diff else None,
                                "item_mean_BA_notext": float(np.mean([Sb[i] - Sa[i] for i in notext])),
                                "item_mean_BA_text": float(np.mean([Sb[i] - Sa[i] for i in text])),
                                "item_mean_BA_all": float(np.mean([Sb[i] - Sa[i] for i in items])),
                                "frac_items_changed_notext": float(np.mean([abs(Sb[i] - Sa[i]) > 1e-9 for i in notext])),
                                "valid_flip_notext": None}
    # singleton-group contribution on no-text
    by = collections.defaultdict(list)
    for i in notext:
        by[group[i]].append(Sb[i] - Sa[i])
    gm = {g: float(np.mean(v)) for g, v in by.items()}
    small = [v for g, v in gm.items() if gs_nt[g] <= 3]; big = [v for g, v in gm.items() if gs_nt[g] > 3]
    res["same_day_split"][m]["notext_groupmean_small_le3"] = [float(np.mean(small)), len(small)] if small else None
    res["same_day_split"][m]["notext_groupmean_big_gt3"] = [float(np.mean(big)), len(big)] if big else None
OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
print(json.dumps(res, indent=1, default=float))
