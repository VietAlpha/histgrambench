"""Read-only computations for outline revision r1 (review 2026-10-03 roadmap). No model call, no compile,
no write under hmb_data/. Sensitivity rows re-score stored verdicts with the project's own scorer
(hmb.grading.cli._rescore, score_cell) and order models with the project's own partial order.
Output: revision_numbers.json beside this script."""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hmb import stats  # noqa: E402
from hmb.adjudication.core import recompute_without_family  # noqa: E402
from hmb.config import load_thresholds  # noqa: E402
from hmb.grading.cli import _rescore, load_track_cells, load_verdicts  # noqa: E402
from hmb.grading.tables import headline, pairwise_partial_order  # noqa: E402
from hmb.schemas import Card  # noqa: E402
from hmb.store import Layout, read_json  # noqa: E402

L = Layout.from_env(__import__("os").environ.get("HMB_ROOT", REPO / "hmb_data"))
D = L.root
TH = load_thresholds(D / "thresholds.json")
OUT = Path(__file__).with_name("revision_numbers.json")
SR = json.loads((D / "reports" / "scores_test_histmathbench-1.0.0.json").read_text(encoding="utf-8"))
MODELS = sorted(SR["tables"]["A"]["models"])
SHORT = {"google.gemma-4-26b-a4b": "G26", "us.meta.llama4-maverick-17b-instruct-v1_0": "MAV",
         "mistral.ministral-3-14b-instruct": "MIN", "google.gemma-4-e2b": "E2B",
         "global.anthropic.claude-haiku-4-5-20251001-v1_0": "HAI", "nvidia.nemotron-nano-12b-v2": "NEM"}
res: dict = {}

# ------------------------------------------------------------------ items and metadata
rights = {b["book_id"]: b for b in yaml.safe_load((D / "sources" / "rights_manifest.yaml").read_text(encoding="utf-8"))}
rec = {}
for p in sorted((D / "data" / "test").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if r["quality"]["eligible"]:
        rec[r["id"]] = r
items = sorted(rec)
group = {i: rec[i]["split_group"] for i in items}
book = {i: rec[i]["source"]["book_id"] for i in items}
ctx = {i: int((rec[i].get("extraction") or {}).get("context_char_count", 0)) for i in items}
cards = {i: Card.model_validate(read_json(L.card(i))) for i in items}
G = len(set(group.values()))


def repo(b):
    return rights[b]["acquisition"]["source_repository"].split(" (")[0].strip().replace("owner file", "Archival")


def gest(values, its):
    by = collections.defaultdict(list)
    for i in its:
        by[group[i]].append(values[i])
    gm = [float(np.mean(v)) for v in by.values()]
    im = stats.im_group_t(gm, alpha=float(TH.value("alpha")))
    return {"est": round(float(np.mean(gm)), 4), "ci": [round(x, 4) for x in im["ci"]] if im.get("ci") else None,
            "p": im.get("p_value"), "G": len(gm), "n": len(its)}


cells = {}
for t in "AB":
    main_, _, _ = load_track_cells(L, MODELS, t, TH)
    cells[t] = {m: {i: c[i] for i in items} for m, c in main_.items()}
for m in MODELS:
    assert abs(headline(cells["A"][m], items, group, TH)["estimate"] - SR["tables"]["A"]["models"][m]["headline"]["estimate"]) < 1e-12


def summarize(cbm, its=None):
    its = its or sorted(set.intersection(*[set(c) for c in cbm.values()]))
    H = {SHORT[m]: round(headline(cbm[m], its, group, TH)["estimate"], 4) for m in MODELS}
    po = pairwise_partial_order(cbm, its, group, TH)
    bands = {SHORT[m]: po["rank_bands"][m] for m in MODELS}
    nb = len({tuple(b) for b in bands.values()})
    top = po["pairs"].get("google.gemma-4-26b-a4b|us.meta.llama4-maverick-17b-instruct-v1_0")
    full = [SR["tables"]["A"]["models"][m]["headline"]["estimate"] for m in MODELS]
    tau = stats.kendall_tau_b(full, [H[SHORT[m]] for m in MODELS])["tau_b"]
    return {"N": len(its), "H": H, "bands": bands, "n_distinct_bands": nb, "n_separated": len(po["separated"]),
            "G26_vs_MAV": {"diff": round(top["diff"], 4), "p": top["p_in_family"], "bh_adj": top["bh_adjusted"],
                           "separated": top["separated"]}, "tau_b_vs_headline": tau}


# ------------------------------------------------------------------ sensitivity rows (Track A), project scorer
def vtrans(f):
    def tr(card, vf):
        if vf is None:
            return card, None
        return card, vf.model_copy(update={"facts": [fv.model_copy(update={"verdict": f(fv, card, vf)}) for fv in vf.facts]})
    return tr


def keep(fv, *_):
    return fv.verdict


cA = {m: dict(cells["A"][m]) for m in MODELS}
res["headline_A"] = summarize(cA)
sens = {
    "geometry_confirmed": lambda fv, c, v: ("ABSENT" if fv.source == "judges" and str(fv.verdict) == "PRESENT" else fv.verdict),
    "forced_as_absent": lambda fv, c, v: ("ABSENT" if (fv.evidence or "").find("forced") >= 0 and str(fv.verdict) == "PRESENT" else fv.verdict),
    "forced_as_present": lambda fv, c, v: ("PRESENT" if (fv.evidence or "").find("forced") >= 0 and str(fv.verdict) in ("ABSENT", "INDETERMINATE") else fv.verdict),
    "fixed_denominator": lambda fv, c, v: ("ABSENT" if str(fv.verdict) == "EXEMPT" else fv.verdict),
}
res["sensitivity_A"] = {}
for name, f in sens.items():
    res["sensitivity_A"][name] = summarize(_rescore(L, TH, MODELS, "A", cA, cards, vtrans(f)))
fams = ["google", "mistral", "qwen"]
for fam in fams:
    def lo(fv, card, vf, fam=fam, memo={}):
        key = (vf.item_id, vf.model_id, fam)
        if key not in memo:
            memo.clear()
            memo[key] = recompute_without_family(vf, card, fam)
        return memo[key][fv.fact_id]
    res["sensitivity_A"][f"without_{fam}"] = summarize(_rescore(L, TH, MODELS, "A", cA, cards, vtrans(lo)))
# drop largest group
sizes = collections.Counter(group.values())
big = max(sizes, key=lambda k: sizes[k])
res["sensitivity_A"]["without_largest_group"] = summarize(cA, [i for i in items if group[i] != big])
# item-weighted point order
res["item_weighted_A"] = {SHORT[m]: round(float(np.mean([cA[m][i].S for i in items])), 4) for m in MODELS}

# BY adjustment of the 15 Track A pairs
po = SR["tables"]["A"]["partial_order"]["pairs"]
ps = np.array([po[k]["p_in_family"] for k in po])
order = np.argsort(ps)
m_ = len(ps)
cm = sum(1.0 / k for k in range(1, m_ + 1))
adj = np.empty(m_)
running = 1.0
for rank in range(m_ - 1, -1, -1):
    idx = order[rank]
    running = min(running, ps[idx] * m_ * cm / (rank + 1))
    adj[idx] = min(running, 1.0)
res["by_A"] = {"c_m": round(cm, 4), "separated": int((adj <= 0.10).sum()),
               "pairs": {k: {"p": po[k]["p_in_family"], "bh": po[k]["bh_adjusted"], "by": float(a)} for k, a in zip(po, adj)}}

# ------------------------------------------------------------------ fact-level tallies from verdict files
COEXACT = {"on", "endpoint_of", "vertex_of", "edge_of", "center_of", "arc_of", "intersect_at", "between", "contains", "face_of"}
DEPICTED_EXACT = {"parallel", "perpendicular", "tangent_to", "collinear", "concyclic"}
MARKED_EXACT = {"equal_length", "equal_angle"}
pred = {(i, f.fact_id): f.predicate for i in items for f in cards[i].facts if f.eligible}
by_item = collections.defaultdict(list)
for (ii, fid), pr in pred.items():
    by_item[ii].append((fid, pr))
cfact = {(i, f.fact_id): f for i in items for f in cards[i].facts}
over_cap = {}
tiers = {}
routing = {}
unbound_entity = {}
for t in "AB":
    for m in MODELS:
        rows = collections.defaultdict(list)
        rt = collections.Counter()
        ue = [0, 0]
        for i in items:
            ok = str(cells[t][m][i].outcome_code) == "OK"
            vf = load_verdicts(L, m, t, i, TH) if ok else None
            vmap = {fv.fact_id: fv for fv in vf.facts} if vf else {}
            cap = None
            if ok:
                cg = D / "cgx" / m / t / f"{i}.json"
                cap = json.loads(cg.read_text(encoding="utf-8")).get("primitives_over_cap") if cg.exists() else None
            for tier, P in (("coexact", COEXACT), ("depicted_exact", DEPICTED_EXACT), ("marked_exact", MARKED_EXACT)):
                fids = [fid for fid, pr in by_item[i] if pr in P]
                if not fids:
                    continue
                if not ok:
                    rows[tier].append(0.0)
                    continue
                sc = [vmap[f] for f in fids if f in vmap and str(vmap[f].verdict) != "EXEMPT"]
                if sc:
                    rows[tier].append(sum(str(v.verdict) == "PRESENT" for v in sc) / len(sc))
            for fv in vmap.values():
                rt["all"] += 1
                if fv.source != "judges":
                    continue
                c = cfact[(i, fv.fact_id)]
                kind = "judge_only_type" if c.decidability != "geo" else ("over_cap" if cap else "unbound")
                rt[kind] += 1
                if kind == "unbound" and c.predicate in {"point", "segment", "line", "ray", "circle", "arc", "ellipse",
                                                         "polygon", "axis", "arrow", "angle_marker", "right_angle_marked", "length_tick"}:
                    ue[1] += 1
                    ue[0] += str(fv.verdict) == "PRESENT"
        tiers[f"{SHORT[m]}|{t}"] = {k: {"R": round(float(np.mean(v)), 4), "n_items": len(v)} for k, v in rows.items()}
        judged = rt["judge_only_type"] + rt["over_cap"] + rt["unbound"]
        routing[f"{SHORT[m]}|{t}"] = {"judged_share": round(judged / rt["all"], 4),
                                      **{k: round(rt[k] / judged, 4) for k in ("judge_only_type", "over_cap", "unbound")}}
        unbound_entity[f"{SHORT[m]}|{t}"] = round(ue[0] / ue[1], 4) if ue[1] else None
res["relation_tiers"] = tiers
res["routing"] = routing
res["unbound_entity_present_rate"] = unbound_entity

# ------------------------------------------------------------------ loss decomposition, common-valid HAI vs MAV
dec = {}
for m in MODELS:
    by = collections.defaultdict(list)
    for i in items:
        c = cA[m][i]
        nu = 1.0 if str(c.outcome_code) == "OK" else 0.0
        by[group[i]].append((1 - nu, nu * (1 - (c.S or 0.0))))
    inv = float(np.mean([np.mean([a for a, _ in v]) for v in by.values()]))
    miss = float(np.mean([np.mean([b for _, b in v]) for v in by.values()]))
    dec[SHORT[m]] = {"loss": round(inv + miss, 4), "invalid_part": round(inv, 4), "valid_miss_part": round(miss, 4),
                     "invalid_share_of_loss": round(inv / (inv + miss), 4)}
res["loss_decomposition_A"] = dec
hai, mav = "global.anthropic.claude-haiku-4-5-20251001-v1_0", "us.meta.llama4-maverick-17b-instruct-v1_0"
both = [i for i in items if str(cA[hai][i].outcome_code) == "OK" and str(cA[mav][i].outcome_code) == "OK"]
res["common_valid_HAI_minus_MAV_A"] = gest({i: cA[hai][i].S - cA[mav][i].S for i in both}, both)

# ------------------------------------------------------------------ page text: subset tests with BH, group-size split, item-weighted
with_text = [i for i in items if ctx[i] > 0]
no_text = [i for i in items if ctx[i] == 0]
tests, keys = [], []
pt = {}
for m in MODELS:
    d = {i: cells["B"][m][i].S - cells["A"][m][i].S for i in items}
    for name, its in (("all", items), ("text", with_text), ("no_text", no_text)):
        e = gest(d, its)
        pt[f"{SHORT[m]}|{name}"] = e
        tests.append(e["p"])
        keys.append(f"{SHORT[m]}|{name}")
    gs = collections.Counter(group[i] for i in no_text)
    small = [i for i in no_text if gs[group[i]] <= 3]
    large = [i for i in no_text if gs[group[i]] > 3]
    pt[f"{SHORT[m]}|no_text_groups_le3"] = gest(d, small)
    pt[f"{SHORT[m]}|no_text_groups_gt3"] = gest(d, large)
    pt[f"{SHORT[m]}|item_weighted_all"] = round(float(np.mean([d[i] for i in items])), 4)
bh = stats.benjamini_hochberg(tests, float(TH.value("bh_q")))
res["page_text"] = {"tests": pt, "bh_adjusted": dict(zip(keys, [round(x, 4) for x in bh["adjusted"]])),
                    "bh_rejected": [k for k, r in zip(keys, bh["rejected"]) if r],
                    "groups": {"text": len({group[i] for i in with_text}), "no_text": len({group[i] for i in no_text}),
                               "both": len({group[i] for i in with_text} & {group[i] for i in no_text})}}

# ------------------------------------------------------------------ corpus composition, weight share, language, pathway
JOURNAL = re.compile(r"Nouvelles annales|Mathematische Annalen|Nachrichten der Gesellschaft|Extrait|Suite de", re.I)
LANG = [(r"數學", "zh"), (r"[А-Яа-я]", "ru"), (r"Goojratee", "gu"), (r"Elementos de geometria", "pt"),
        (r"Problemi e teoremi", "it"), (r"Nouvelles annales|Éléments de géométrie|La géométrie|Lemoine|Suite de", "fr"),
        (r"Elemente|Vorlesungen|Geschichte|Annalen|Logik|Ebene geometrie|Abhandlungen|Nachrichten|Einführung", "de")]


def lang(title):
    for pat, code in LANG:
        if re.search(pat, title):
            return code
    return "en"


def pathway(i):
    r = repo(book[i])
    if r == "Project Gutenberg":
        return "transcriber-prepared image"
    ex = rec[i].get("extraction") or {}
    frame = ex.get("frame") or (ex.get("source_crop") or {}).get("frame")
    return "repository plate image" if frame == "precropped_image" else "page-scan crop"


n_g = collections.Counter(group.values())
w = {i: 1.0 / (G * n_g[group[i]]) for i in items}
comp = collections.defaultdict(lambda: {"figures": 0, "works": set(), "groups": set(), "weight": 0.0})
for i in items:
    b = book[i]
    title = str(rights[b]["display_title"])
    for dim, val in (("repository", repo(b)), ("genre", "journal article" if JOURNAL.search(title) else "book"),
                     ("language", lang(title)), ("pathway", pathway(i))):
        c = comp[f"{dim}|{val}"]
        c["figures"] += 1
        c["works"].add(b)
        c["groups"].add(group[i])
        c["weight"] += w[i]
res["composition"] = {k: {"figures": v["figures"], "works": len(v["works"]), "groups": len(v["groups"]),
                          "headline_weight": round(v["weight"], 4)} for k, v in sorted(comp.items())}
res["singleton_groups"] = {"n": sum(1 for v in n_g.values() if v == 1), "weight": round(sum(1 for v in n_g.values() if v == 1) / G, 4)}

# label scripts in keys
scr = collections.Counter()
for i in items:
    s = set()
    for f in cards[i].facts:
        if f.eligible and f.predicate == "label":
            txt = str(f.args[0])
            if re.search(r"[Ͱ-Ͽ]", txt):
                s.add("greek")
            if re.search(r"[Ѐ-ӿ]", txt):
                s.add("cyrillic")
            if re.search(r"[一-鿿]", txt):
                s.add("cjk")
    for x in s:
        scr[x] += 1
res["label_script_items"] = dict(scr)

# descriptive strata on Track A: repository, pathway, Byrne
def strat(sel):
    its = [i for i in items if sel(i)]
    gcount = len({group[i] for i in its})
    nf = sum(sum(f.eligible for f in cards[i].facts) for i in its)
    ok = len(its) >= 30 and gcount >= 3 and nf >= 150
    return {"n": len(its), "groups": gcount, "facts": nf, "publishable": ok,
            "H": {SHORT[m]: (round(headline(cA[m], its, group, TH)["estimate"], 4) if ok else None) for m in MODELS}}


res["strata_A"] = {}
for r_ in sorted({repo(book[i]) for i in items}):
    res["strata_A"][f"repository|{r_}"] = strat(lambda i, r_=r_: repo(book[i]) == r_)
res["strata_A"]["without_Numdam"] = strat(lambda i: repo(book[i]) != "Numdam")
for pw in ("page-scan crop", "repository plate image", "transcriber-prepared image"):
    res["strata_A"][f"pathway|{pw}"] = strat(lambda i, pw=pw: pathway(i) == pw)
byrne = [i for i in items if rights[book[i]]["publication_year"] == 1847]
res["byrne"] = {"n": len(byrne), "groups": len({group[i] for i in byrne}),
                "mean_facts": round(float(np.mean([sum(f.eligible for f in cards[i].facts) for i in byrne])), 2),
                "items_with_label_facts": sum(any(f.eligible and f.predicate == "label" for f in cards[i].facts) for i in byrne),
                "item_mean_S_A": {SHORT[m]: round(float(np.mean([cA[m][i].S for i in byrne])), 4) for m in MODELS}}

# label errors share (score report, Track A)
le = {}
for m in MODELS:
    e = SR["tables"]["A"]["models"][m]["per_class"]["label_errors"]
    tot = sum(e.values())
    le[SHORT[m]] = {k: round(v / tot, 3) for k, v in e.items()}
res["label_error_shares_A"] = le

OUT.write_text(json.dumps(res, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)), encoding="utf-8")
print("written", OUT)
