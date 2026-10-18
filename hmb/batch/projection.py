"""Granular cost projection: on-demand vs Global CRIS + hybrid batch (2026-09-26). A projection, not an invoice.

Inputs (all committed, so `hmb batch-cost --out reports --date 20260926` regenerates the result):
  * hmb/batch/price_snapshot.json                 dated per-route on-demand and explicit batch rates
  * reports/prompt_repair_20260926_manifest.json  measured per-call tokens (live flow + two-writer cards)
  * reports/ms2_pilot_writer_calls_20260927_manifest.json  measured staged writer calls of the roster writers
                                                  (GPT-6 Sol at medium, Kimi K2.5) on 15 real MS2 pilot crops
  * reports/batch_probes_20260926/*.json          measured probe usage (one-time engineering spend)
  * hmb/batch/capability_matrix.json              which routes are batch in the implemented design

Formula per call: cost = input_tokens x in_rate / 1e6 + output_tokens x out_rate / 1e6 (image tokens are part
of the reported input tokens on every roster route). Per-call token means are the measured means for that
(model, purpose); the low/high usage cases use the 90% bootstrap interval of each mean together with the
structural ranges in STRUCT. Prices are never varied inside a usage case: price uncertainty is reported
separately (missing rates stay symbolic).
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = REPO / "reports" / "prompt_repair_20260926_manifest.json"
WRITER_MANIFEST = REPO / "reports" / "ms2_pilot_writer_calls_20260927_manifest.json"
PROBES = REPO / "reports" / "batch_probes_20260926"

# Implemented routes (Global CRIS migration 2026-09-26) and the pre-migration routes for the route baseline.
ROUTES = {
    "gatekeeper": {"now": "global.openai.gpt-6-luna", "before": "openai.gpt-6-luna", "tokens": ("openai.gpt-6-luna", "G8b")},
    # Writers (owner decisions 2026-09-26): GPT-6 Sol (Global CRIS, medium) and Kimi K2.5; Sonnet 5 held slot A in
    # between. "tokens" is the model whose measured calls price the slot: the roster writers' own staged calls
    # (WRITER_MANIFEST, MS2 pilot) for the implemented design, the measured single-call Opus 5.5 / GPT-6 Sol calls
    # (Mantle) for the previous one (S2).
    "writer_a": {"now": "global.openai.gpt-6-sol", "before": "us.anthropic.claude-opus-5-5",
                 "tokens": "global.openai.gpt-6-sol", "tokens_before": "us.anthropic.claude-opus-5-5"},
    "writer_b": {"now": "moonshotai.kimi-k2.5", "before": "openai.gpt-6-sol", "tokens": "moonshotai.kimi-k2.5",
                 "tokens_before": "openai.gpt-6-sol"},
}
JUDGES = ("qwen.qwen3-vl-235b-a22b-instruct", "google.gemma-4-31b", "mistral.mistral-large-3-675b-instruct")
IMAGE_CONTESTANTS = ("us.meta.llama4-maverick-17b-instruct-v1:0", "nvidia.nemotron-nano-12b-v2",
                     "google.gemma-4-26b-a4b", "mistral.ministral-3-14b-instruct", "google.gemma-4-e2b")
PREVIOUS_IMAGE_CONTESTANTS = ("us.meta.llama4-maverick-17b-instruct-v1:0", "moonshotai.kimi-k2.5",
                              "nvidia.nemotron-nano-12b-v2", "google.gemma-4-26b-a4b",
                              "mistral.ministral-3-14b-instruct", "google.gemma-4-e2b")
TEXT_FLOOR = "deepseek.v3.1"
TRACKS = 2

# Structural usage parameters: (low, base, high) and their source. "measured" = 2026-09-26 live flow.
STRUCT = {
    "crops_per_item": ((1.25, 2.0, 4.0), "ASSUMED: G8b candidates screened per final item (real-crop funnel not yet measured; "
                                         "CP2 smoke passed 7 of 8 crops to the gatekeeper)"),
    "cards_per_item": ((1.0, 1.25, 1.67), "ASSUMED: cards built per final item = 1 / card-gate retention r_hat (MS3 not yet "
                                          "run); 1.25 = r_hat 0.8"),
    "valid1": ((0.70, 0.82, 0.95), "measured 23/28 locked-contestant cells compiled OK (flow); range assumed"),
    "verify_calls_per_writer": ((2.5, 3.4, 8.0), "asc-2.0.0 verification: ceil(|union MGS| / card_verify_chunk_size) "
                                                  "per writer. Low = mean on the 6 synthetic items (1-4); base = MS2 "
                                                  "pilot mean (51 calls per writer / 15 real cards, 2026-09-27); "
                                                  "high = the largest measured on the CP2 crops (hard cap 14)"),
    "gen_calls_per_stage": ((1.0, 1.0, 1.25), "staged generation (2026-09-26 repair): per writer 1 entity + 1 fact "
                                              "call, each retried at most once when cut with no complete line "
                                              "(<= 4 calls). Measured 0 retries in 4 stages at the roster's low "
                                              "effort; high ASSUMES a retry on 1 stage in 4"),
    "q1_per_cell": ((1.5, 1.5, 1.5), "measured ~1.5 Q1 calls per judged image cell (all judges)"),
    "q3_per_cell": ((1.0, 1.0, 1.0), "measured ~1 Q3 call per judged image cell"),
}
# measured per text-only cell (report §8); Q2 and the extras channel were removed 2026-09-26
T3_CELL = {"q1": 2.5, "q3": 2.5}
# Phases (reports/prompt_repair_20260926.md §8 fixed development work; final N is the owner's sizing decision).
PHASES = {"PILOT": {"items": 30, "kind": "full"}, "DEV": {"crops": 80, "kind": "cards"}}   # MS2, MS3 (CAL/VAL deleted 2026-09-26)
FINAL_N = (100, 250, 500, 1000)
BASE_N = 250
S3_RECORD_KB = (20.0, 50.0, 300.0)     # ASSUMED batch record size (crop base64 + prompt); corpus crops average ~19 KB


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def token_stats(seed: int = 7) -> dict:
    """(model, purpose) -> {n, in_mean, out_mean, in_ci, out_ci, out_p10, out_p90} from the measured calls."""
    d = _load(MANIFEST)
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for c in d["calls"]:
        if c["root"] in ("root_flow", "root_cards2") and c.get("effort") in (None, "medium", "high"):
            if c["model_id"] == "qwen.qwen3-vl-235b-a22b-instruct" and c.get("effort") == "high":
                continue                                  # the adopted Qwen grader effort is medium
            groups.setdefault((c["model_id"], c["purpose"]), []).append((c["input_tokens"], c["output_tokens"]))
    for c in _load(WRITER_MANIFEST)["calls"]:           # the roster writers at their pinned efforts (staged)
        groups.setdefault((c["model_id"], c["purpose"]), []).append((c["input_tokens"], c["output_tokens"]))
    rng = random.Random(seed)
    out = {}
    for k, v in groups.items():
        ins, outs = [x for x, _ in v], [y for _, y in v]

        def ci(xs):
            if len(xs) < 2:
                return (xs[0], xs[0])
            means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(1000))
            return (means[50], means[949])
        so = sorted(outs)
        out[f"{k[0]}|{k[1]}"] = {"n": len(v), "in_mean": statistics.fmean(ins), "out_mean": statistics.fmean(outs),
                                 "in_ci": ci(ins), "out_ci": ci(outs), "out_p10": so[len(so) // 10],
                                 "out_p90": so[min(len(so) - 1, len(so) * 9 // 10)]}
    return out


def judge_shares(stats: dict) -> dict:
    """Share of each question's calls per judge (measured on the flow)."""
    sh = {}
    for q in ("Q1", "Q3"):
        n = {j: stats.get(f"{j}|{q}", {}).get("n", 0) for j in JUDGES}
        t = sum(n.values())
        sh[q] = {j: n[j] / t for j in JUDGES}
    return sh


class Prices:
    def __init__(self, snap: dict):
        self.m = {e["model_id"]: e for e in snap["models"]}
        self.s3 = snap["s3"]

    def rate(self, model_id: str, lane: str) -> dict:
        e = self.m[model_id]
        r = e["on_demand"] if lane == "on_demand" else e["batch"]
        if r is None:
            raise KeyError(f"{model_id}: no published {lane} rate (never invented)")
        return r

    def has_batch(self, model_id: str) -> bool:
        return self.m[model_id]["batch"] is not None


def _calls(case: int, stats: dict, shares: dict, n_items: float = 0, crops: float = 0, cards: float = 0,
           contestants: tuple = IMAGE_CONTESTANTS,
           cells: float = 0, judged_cells: float | None = None, t3_cells: float = 0,
           writers: str = "now") -> list[dict]:
    """Every modelled call group: role, token model, purpose, calls, per-call in/out tokens (case 0/1/2)."""
    s = {k: v[0][case] for k, v in STRUCT.items()}

    def tok(model, purpose):
        st = stats[f"{model}|{purpose}"]
        pick = {0: 0, 2: 1}
        if case == 1:
            return st["in_mean"], st["out_mean"]
        return st["in_ci"][pick[case]], st["out_ci"][pick[case]]
    rows = []
    g_calls = crops if crops else n_items * s["crops_per_item"]
    if g_calls:
        i, o = tok("openai.gpt-6-luna", "G8b")
        rows.append({"role": "gatekeeper", "slot": "gatekeeper", "stage": "G8b", "calls": g_calls, "in": i, "out": o})
    c_calls = cards if cards else n_items * s["cards_per_item"]
    if c_calls:
        for slot in ("writer_a", "writer_b"):
            if writers == "before":             # previous design: one generation + one unchunked verification call
                tm, calls = ROUTES[slot]["tokens_before"], (("card_generation", 1.0), ("card_verification", 1.0))
            else:                               # staged generation (each stage <= 1 retry), chunked verification
                tm = ROUTES[slot]["tokens"]
                calls = (("card_generation_entities", s["gen_calls_per_stage"]),
                         ("card_generation_facts", s["gen_calls_per_stage"]),
                         ("card_verification", s["verify_calls_per_writer"]))
            for p, per in calls:
                i, o = tok(tm, p)
                rows.append({"role": "answer-key builder", "slot": slot, "stage": p, "calls": c_calls * per,
                             "in": i, "out": o})
    n_cells = cells if cells else n_items * len(contestants) * TRACKS
    if n_items and not cells:
        for mid in contestants:
            i, o = tok(mid, "contestant")
            rows.append({"role": "contestant", "slot": mid, "stage": "contestant", "calls": n_items * TRACKS, "in": i, "out": o})
    jc = judged_cells if judged_cells is not None else n_cells * s["valid1"]
    if jc:
        for q, per in (("Q1", s["q1_per_cell"]), ("Q3", s["q3_per_cell"])):
            for j in JUDGES:
                i, o = tok(j, q)
                rows.append({"role": "judge", "slot": j, "stage": q, "calls": jc * per * shares[q][j], "in": i, "out": o})
    if t3_cells:
        i, o = tok(TEXT_FLOOR, "contestant")
        rows.append({"role": "contestant", "slot": TEXT_FLOOR, "stage": "T3 contestant", "calls": t3_cells, "in": i, "out": o})
        for q, per in (("Q1", T3_CELL["q1"]), ("Q3", T3_CELL["q3"])):
            for j in JUDGES:
                i, o = tok(j, q)
                rows.append({"role": "judge", "slot": j, "stage": f"T3 {q}", "calls": t3_cells * per * shares[q][j],
                             "in": i, "out": o})
    return rows


def _route(slot: str, scenario: str) -> str:
    if slot in ROUTES:
        return ROUTES[slot]["before" if scenario == "S2" else "now"]
    return slot


# Batch in the implemented design: the Maverick and Ministral contestant calls and the Kimi K2.5 writer calls.
BATCH_IMPLEMENTED = ("us.meta.llama4-maverick-17b-instruct-v1:0", "moonshotai.kimi-k2.5", "mistral.ministral-3-14b-instruct")


def _lane(model: str, scenario: str, phase_records: float, prices: Prices, min_records: int = 100) -> str:
    if scenario == "S3":
        return "batch" if model in BATCH_IMPLEMENTED and phase_records >= min_records else "on_demand"
    if scenario == "S4":
        return "batch" if prices.has_batch(model) else "on_demand"
    return "on_demand"


def price_rows(rows: list[dict], scenario: str, prices: Prices, *, phase: str) -> list[dict]:
    out = []
    for r in rows:
        model = _route(r["slot"], scenario)
        lane = _lane(model, scenario, r["calls"], prices)
        rt = prices.rate(model, lane)
        cost = r["calls"] * (r["in"] * rt["in"] + r["out"] * rt["out"]) / 1e6
        out.append({**r, "phase": phase, "scenario": scenario, "model": model, "lane": lane, "rate_in": rt["in"],
                    "rate_out": rt["out"], "input_tokens": r["calls"] * r["in"], "output_tokens": r["calls"] * r["out"],
                    "usd": cost})
    return out


def phase_rows(phase: str, case: int, stats, shares, n_final: int, scenario: str = "S3") -> list[dict]:
    cont = PREVIOUS_IMAGE_CONTESTANTS if scenario == "S2" else IMAGE_CONTESTANTS
    w = "before" if scenario == "S2" else "now"
    if phase == "FINAL":
        return _calls(case, stats, shares, n_items=n_final, contestants=cont, writers=w)
    p = PHASES[phase]
    if p["kind"] == "full":
        return _calls(case, stats, shares, n_items=p["items"], contestants=cont, writers=w)
    return _calls(case, stats, shares, crops=p["crops"], cards=p["crops"], writers=w)


def s3_cost(records: float, prices: Prices, case: int, jobs: float) -> float:
    kb = S3_RECORD_KB[case]
    gb = records * kb * 2 / 1024 / 1024                     # input + output kept one month
    return jobs * (4 * prices.s3["put_per_1k"] + 4 * prices.s3["get_per_1k"]) / 1000 + gb * prices.s3["storage_gb_month"]


def probe_spend(prices: Prices) -> dict:
    """Measured one-time engineering spend: batch probe jobs (batch lane) + recorded parity ledgers."""
    rows = []
    for p in sorted(PROBES.glob("*__plain.json")) + sorted(PROBES.glob("*__schema.json")):
        ev = _load(p)
        r = ev.get("result")
        if not r:
            rows.append({"probe": p.stem, "usd": 0.0, "note": "rejected at submission (no charge)"})
            continue
        rt = prices.rate(ev["model_id"], "batch")
        rows.append({"probe": p.stem, "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                     "usd": (r["input_tokens"] * rt["in"] + r["output_tokens"] * rt["out"]) / 1e6, "lane": "batch"})
    for name in ("gpt6_global_cris_parity.json", "gpt6_mantle_vs_global_parity.json"):
        f = PROBES / name
        if f.exists():
            led = _load(f).get("ledger") or {}
            rows.append({"probe": name, "usd": float(led.get("usd", 0.0)), "lane": "on_demand",
                         "note": f"{led.get('settled')} settled calls (scratch ledger, dated rates)"})
    rows.append({"probe": "raw sanity calls (IAM check, token accounting)", "usd": 0.002, "lane": "on_demand",
                 "note": "8 small calls, upper bound from their reported usage (<= 2k tokens each at <= $0.55/M)"})
    return {"rows": rows, "usd": sum(r["usd"] for r in rows)}


def project(n_final: int = BASE_N, pricing: Path | None = None) -> dict:
    prices = Prices(_load(pricing or HERE / "price_snapshot.json"))
    stats = token_stats()
    shares = judge_shares(stats)
    out: dict = {"n_final": n_final, "scenarios": {}, "struct": {k: {"low_base_high": v[0], "source": v[1]}
                                                                  for k, v in STRUCT.items()},
                 "token_stats": stats, "judge_shares": shares}
    for sc in ("S1", "S2", "S3", "S4"):
        cases = {}
        for case, cname in ((0, "low"), (1, "base"), (2, "high")):
            rows = []
            for ph in ("PILOT", "DEV", "FINAL"):
                rows += price_rows(phase_rows(ph, case, stats, shares, n_final, sc), sc, prices, phase=ph)
            batch_recs = sum(r["calls"] for r in rows if r["lane"] == "batch")
            s3 = s3_cost(batch_recs, prices, case, jobs=max(1, len({r["model"] for r in rows if r["lane"] == "batch"})) * 3) \
                if batch_recs else 0.0
            cases[cname] = {"rows": rows, "s3_usd": s3, "total_usd": sum(r["usd"] for r in rows) + s3,
                            "final_usd": sum(r["usd"] for r in rows if r["phase"] == "FINAL"),
                            "batch_calls": batch_recs, "calls": sum(r["calls"] for r in rows),
                            "batch_usd": sum(r["usd"] for r in rows if r["lane"] == "batch")}
        out["scenarios"][sc] = cases
    out["probes"] = probe_spend(prices)
    out["t3_per_item"] = {sc: sum(r["usd"] for r in price_rows(_calls(1, stats, shares, t3_cells=1), sc, prices, phase="T3"))
                          for sc in ("S1", "S2", "S3", "S4")}
    # Unknown rates stay symbolic: the on-demand dollars of every call whose model publishes no batch rate.
    base = out["scenarios"]["S3"]["base"]["rows"]
    unknown: dict = {}
    for r in base:
        if not prices.has_batch(r["model"]):
            u = unknown.setdefault(r["model"], {"on_demand_usd": 0.0, "input_tokens": 0.0, "output_tokens": 0.0})
            u["on_demand_usd"] += r["usd"]
            u["input_tokens"] += r["input_tokens"]
            u["output_tokens"] += r["output_tokens"]
    out["unknown_batch_rates"] = unknown
    # Price-side uncertainty that is not usage: Converse implicit prompt-cache reads (billed at 10% of the input
    # rate) inside the GPT-6 input tokens; measured cached share on identical requests (token-accounting evidence).
    ta = PROBES / "gpt6_token_accounting.json"
    cached_share = None
    if ta.exists():
        t = _load(ta)["converse_global"]
        cached_share = t["cacheReadInputTokens"] / (t["inputTokens"] + t["cacheReadInputTokens"])
    gpt6_in = sum(r["input_tokens"] * r["rate_in"] / 1e6 for r in base if r["model"].startswith("global.openai.gpt-6"))
    out["price_uncertainty"] = {"gpt6_cached_input_share_measured": cached_share, "gpt6_input_usd_base": gpt6_in,
                                "gpt6_cache_read_saving_usd_if_share_holds": (gpt6_in * cached_share * 0.9)
                                if cached_share else None}
    out["per_n"] = {}
    for n in FINAL_N:
        out["per_n"][n] = {sc: sum(r["usd"] for r in price_rows(phase_rows("FINAL", 1, stats, shares, n, sc), sc, prices,
                                                                phase="FINAL")) for sc in ("S1", "S2", "S3", "S4")}
    return out


def _fmt(x: float) -> str:
    return f"${x:,.2f}" if x >= 1 else f"${x:.4f}"


def render_md(p: dict, date: str) -> str:
    sc = p["scenarios"]
    names = {"S1": "Implemented roster (writers GPT-6 Sol + Kimi K2.5; GPT-6 on Global CRIS), all on-demand",
             "S2": "Previous configuration (writers Opus 5.5 + GPT-6 Sol, GPT-6 on Mantle, Kimi a contestant), all on-demand",
             "S3": "Implemented: new roster + Global CRIS + hybrid batch",
             "S4": "Sensitivity: maximum plausible batch (NOT implemented)"}
    L = [f"# Batch inference cost projection, {date[:4]}-{date[4:6]}-{date[6:]}", "",
         "Projection, not an invoice (the AWS bill is authoritative). Regenerate with "
         f"`.venv/Scripts/python.exe -m hmb batch-cost --out reports --date {date}`; machine-readable companion "
         f"`reports/batch_inference_cost_projection_{date}.json`. Final benchmark size N = {p['n_final']} items "
         "(the owner's sizing decision; other N below).", "",
         "## Scenarios (full program: pilot + development set + final; engineering probes separate; the MS3 variance probe is not modelled)", "",
         "| Scenario | Low | Base | High | Final run only (base) | Calls on batch (base) | $ on batch (base) |",
         "|---|---|---|---|---|---|---|"]
    for k in ("S1", "S2", "S3", "S4"):
        c = sc[k]
        b = c["base"]
        L.append(f"| {k} {names[k]} | {_fmt(c['low']['total_usd'])} | {_fmt(b['total_usd'])} | {_fmt(c['high']['total_usd'])} "
                 f"| {_fmt(b['final_usd'])} | {b['batch_calls']:,.0f} of {b['calls']:,.0f} ({100 * b['batch_calls'] / b['calls']:.1f}%) "
                 f"| {_fmt(b['batch_usd'])} |")
    s1, s2, s3, s4 = (sc[k]["base"]["total_usd"] for k in ("S1", "S2", "S3", "S4"))
    gk = {k: sum(r["usd"] for r in sc[k]["base"]["rows"] if r["slot"] == "gatekeeper") for k in ("S1", "S2")}
    L += ["", "**Savings (base usage):**",
          f"- Roster and route change, isolated from batch (S2 - S1): {_fmt(s2 - s1)} ({100 * (s2 - s1) / s2:.1f}% of S2): "
          "the writers GPT-6 Sol (Global CRIS) + Kimi K2.5 (staged, measured) instead of Opus 5.5 + GPT-6 Sol (Mantle, single "
          "call, measured), one contestant fewer, and GPT-6 Luna on Global CRIS. Of this, Global CRIS for Luna alone is "
          f"{_fmt(gk['S2'] - gk['S1'])}.",
          f"- Hybrid batch on top (S1 - S3): {_fmt(s1 - s3)} ({100 * (s1 - s3) / s1:.2f}% of S1): Kimi K2.5 writer calls "
          "(the final run; pilot/development lanes are below the 100-record minimum) and the Maverick and Ministral contestant "
          "calls.",
          f"- Implemented design vs the previous on-demand configuration (S2 - S3): {_fmt(s2 - s3)} "
          f"({100 * (s2 - s3) / s2:.1f}%).",
          f"- Sensitivity only, not implemented (S1 - S4): {_fmt(s1 - s4)} ({100 * (s1 - s4) / s1:.1f}%). It prices every "
          "model with an explicit batch SKU at that SKU, including the three Mantle graders and Mantle contestants whose "
          "route has no batch API, and Nemotron (15% record errors in its probe).", ""]
    # breakdowns (base, S3)
    rows = sc["S3"]["base"]["rows"]

    def agg(key):
        d: dict = {}
        for r in rows:
            d[r[key]] = d.get(r[key], 0.0) + r["usd"]
        return d
    L += ["## Implemented design (S3, base) by phase", "", "| Phase | USD | Share |", "|---|---|---|"]
    tot = sum(r["usd"] for r in rows)
    for ph, v in agg("phase").items():
        L.append(f"| {ph} | {_fmt(v)} | {100 * v / tot:.1f}% |")
    L += ["", "## By role (S3, base, full program)", "", "| Role | USD | Share |", "|---|---|---|"]
    for k, v in sorted(agg("role").items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {_fmt(v)} | {100 * v / tot:.1f}% |")
    L += ["", "## By exact model and route, stage and lane (S3 vs S1, base, final run)", "",
          "| Model (route id) | Role | Stage | Lane (S3) | Calls | Input tok | Output tok | In/out $/M (S3) | S3 USD | S1 USD |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    s1rows = {(r["slot"], r["stage"], r["phase"]): r for r in sc["S1"]["base"]["rows"]}
    for r in [x for x in rows if x["phase"] == "FINAL"]:
        o = s1rows[(r["slot"], r["stage"], r["phase"])]
        L.append(f"| `{r['model']}` | {r['role']} | {r['stage']} | {r['lane']} | {r['calls']:,.0f} | {r['input_tokens']:,.0f} "
                 f"| {r['output_tokens']:,.0f} | {r['rate_in']}/{r['rate_out']} | {_fmt(r['usd'])} | {_fmt(o['usd'])} |")
    L += ["", f"Per final item (S3, base): {_fmt(sc['S3']['base']['final_usd'] / p['n_final'])}; per image cell "
          f"(contestant + judging): {_fmt(sum(r['usd'] for r in rows if r['phase'] == 'FINAL' and r['role'] in ('contestant', 'judge')) / (p['n_final'] * 12))}.",
          "", "## Final run size", "", "| N items | S1 | S2 | S3 | S4 | S3 per item |", "|---|---|---|---|---|---|"]
    for n, v in p["per_n"].items():
        L.append(f"| {n} | {_fmt(v['S1'])} | {_fmt(v['S2'])} | {_fmt(v['S3'])} | {_fmt(v['S4'])} | {_fmt(v['S3'] / int(n))} |")
    L += ["", "## Usage assumptions (low / base / high)", "", "| Parameter | Low | Base | High | Source |", "|---|---|---|---|---|"]
    for k, v in p["struct"].items():
        lo, ba, hi = v["low_base_high"]
        L.append(f"| {k} | {lo} | {ba} | {hi} | {v['source']} |")
    L += [f"| per-call tokens | 5th pct of bootstrap mean | measured mean | 95th pct of bootstrap mean | "
          f"{MANIFEST.relative_to(REPO).as_posix()} (root_flow, root_cards2; adopted efforts) |",
          f"| batch record size (S3) | {S3_RECORD_KB[0]} KB | {S3_RECORD_KB[1]} KB | {S3_RECORD_KB[2]} KB | ASSUMED (corpus crops average ~19 KB) |",
          f"| writer tokens (GPT-6 Sol medium, Kimi K2.5) | 5th pct of bootstrap mean | measured mean | 95th pct | "
          f"{WRITER_MANIFEST.relative_to(REPO).as_posix()}: staged calls on 15 real MS2 pilot crops (15 per stage per "
          "writer, 51 verification calls each); S2 keeps the measured single-call Opus 5.5 / GPT-6 Sol tokens |",
          "", "Judging models Q1 and Q3 (Q2 and the extras channel were removed 2026-09-26). Third-judge escalations "
          "are inside the measured per-cell counts. Transport "
          "failures are released by the ledger (not billed). Batch record failures are resubmitted within the retry budget; "
          "the probes of the three batched routes had 0 record errors, so no batch rerun is modelled.", "",
          "## Engineering probes (one-time, measured, separate from benchmark spend)", "", "| Probe | USD | Note |", "|---|---|---|"]
    for r in p["probes"]["rows"]:
        L.append(f"| {r['probe']} | {_fmt(r['usd'])} | {r.get('note', r.get('lane', ''))} |")
    L += [f"| **total** | **{_fmt(p['probes']['usd'])}** | |", ""]
    L += ["## Price uncertainty (kept separate from usage)", "",
          "Every rate used is published and dated (hmb/batch/price_snapshot.json: AWS Price List API publication "
          "2026-09-25 retrieved 2026-09-26, and the GPT-6 model cards). Missing rates are never filled in:", "",
          "| Model | Missing price | Why | Live billing evidence | Symbolic contribution (S3 base) |", "|---|---|---|---|---|"]
    for m, u in p["unknown_batch_rates"].items():
        L.append(f"| `{m}` | batch input/output per 1M tokens | no batch rate published; batch rejected at submission "
                 f"(probe) | none (rejected jobs are not billed) | {u['input_tokens']:,.0f} x r_in_batch + "
                 f"{u['output_tokens']:,.0f} x r_out_batch (on-demand today: {_fmt(u['on_demand_usd'])}) |")
    pu = p["price_uncertainty"]
    if pu["gpt6_cached_input_share_measured"] is not None:
        L += ["", f"GPT-6 on Converse reports implicit prompt-cache reads separately: "
              f"{100 * pu['gpt6_cached_input_share_measured']:.0f}% of the input tokens of an identical G8b request were "
              "cache reads, which AWS bills at 10% of the input rate. The projection prices all input tokens at the full "
              f"rate (conservative); if that share held for every GPT-6 call, the GPT-6 input cost "
              f"({_fmt(pu['gpt6_input_usd_base'])}) would be about {_fmt(pu['gpt6_cache_read_saving_usd_if_share_holds'])} "
              "lower. The ledger does not price cache reads (pre-existing)."]
    b1 = p["scenarios"]["S3"]["base"]
    fin = [r for r in b1["rows"] if r["phase"] == "FINAL"]
    n = p["n_final"]
    by = {k: sum(r["usd"] for r in fin if r["role"] == k) / n
          for k in ("answer-key builder", "judge", "contestant", "gatekeeper")}
    L += ["", "## Reconciliation with the earlier ~$0.55 per item (reports/prompt_repair_20260926.md §8)", "",
          f"This projection's final-run cost per item (S3, base) is {_fmt(b1['final_usd'] / n)}: answer key "
          f"{_fmt(by['answer-key builder'])}, judging {_fmt(by['judge'])}, contestants {_fmt(by['contestant'])}, "
          f"gatekeeper {_fmt(by['gatekeeper'])}. The earlier figure was built as answer key $0.18 + 12 image cells x "
          "$0.022 judging + 1 text-only cell $0.086 + contestants. The material differences, not forced to agree:",
          "- **Text-only cell.** The earlier figure included one DeepSeek text-only cell per item. `hmb run` has no "
          "scored text-input track (the T3 probe is a diagnostic, recommended on a subset), so it is excluded here and "
          f"reported separately: {_fmt(p['t3_per_item']['S3'])} per item if T3 ran on every item.",
          "- **Cards per item.** The earlier figure assumed one card per final item; here 1.25 (card-gate retention 0.8, "
          f"assumed until MS3). At 1.0 the answer key is {_fmt(by['answer-key builder'] / 1.25)} per item.",
          "- **Judged cells.** Only cells that compile are judged (measured 0.82); the earlier figure judged every "
          "image cell.",
          "- **Writers.** The earlier figure's answer key was Opus 5.5 + GPT-6 Sol (Mantle); the writers are now GPT-6 "
          "Sol on Global CRIS ($2.00/$10) and Kimi K2.5 ($0.60/$3.00, batch $0.30/$1.50), priced on their own "
          "measured staged calls in the MS2 pilot (2 generation calls each, chunked verification of real-crop unions).",
          "- **GPT-6 rates.** The earlier rate card used $0.10/$0.50 (Luna) and $2/$10 (Sol), which are the Global CRIS "
          "rates; the Mantle route actually used then costs 10% more ($0.11/$0.55, $2.20/$11). S2 prices the "
          "Mantle route correctly, so the earlier Mantle-era figures were ~10% low for these two models.",
          "- **Measured ledger.** The representative live flow's recorded cost ($4.32 for 6 synthetic items with 3 "
          "writers and 15 contestants) is not comparable per item; the per-call token means used here come from those "
          "same settled calls.", ""]
    return "\n".join(L) + "\n"


def write_projection(out_dir: Path, *, date: str | None = None, pricing: Path | None = None) -> list[Path]:
    from datetime import datetime, timezone
    date = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    p = project(pricing=pricing)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"batch_inference_cost_projection_{date}.md"
    js = out_dir / f"batch_inference_cost_projection_{date}.json"
    md.write_text(render_md(p, date), encoding="utf-8")
    js.write_text(json.dumps(p, indent=1, default=str) + "\n", encoding="utf-8")
    return [md, js]
