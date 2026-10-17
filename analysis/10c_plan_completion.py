"""Step 10c: pre-registered analyses that the first pipeline omitted (found by the M8 review; deviations.md #18).
Estimation sample only (the holdout is never used here).

a) M4 statement novelty (PROJECT_PLAN.md sections 4-6). M4 is unsigned (1 - cosine similarity with the previous
   statement), so it cannot enter the signed H1 specification. Following Ehrmann & Talmi (2020) it is tested on the
   SIZE of the response: |30-min 2y change| on standardised novelty + |MP1| (two-sided).
b) Placebo on random non-FOMC days (plan section 6). Each meeting's standardised tone surprise is assigned to a
   random trading day that is at least 5 trading days away from every FOMC statement (estimation period); the daily
   2-year yield change on that day is regressed on it. 2,000 draws give the placebo distribution of beta, compared
   with the same regression on the actual meeting days (daily window; no MP1, which exists on meeting days only).
c) "Use the 1994-1999 statements" (plan section 7). Scheduled statements before 1999-05-18 with high-frequency data
   are added. M3 for them uses the 1999 model (labels 1996-1998), which has partial label look-ahead for
   1994-1998 statements; M1 (no learning) is reported alongside.
d) H5 information-effect check (plan section 1, H5 rule): S&P 500 on z, pc1, an indicator of JK "central-bank
   information" meetings (CBI_pm != 0) and z x indicator (two-sided).

Output: reports/results_plan_completion.json
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stats  # noqa: E402
from common.paths import MARKET, PROCESSED, REPORTS  # noqa: E402

es = importlib.import_module("06_event_study")
P = es.PRIMARY


def ols_wild(d, y, regs, j=1, alt="two-sided", seed=1):
    d = d[[y] + regs].dropna()
    Y = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in regs])
    r = stats.ols(Y, X, ["const"] + regs)
    name = regs[j - 1]
    return {"n": r["n"], "beta": r["coef"][name], "se_hc3": r["se"][name],
            "p_wild": stats.wild_boot(Y, X, j, B=9999, seed=seed, alternative=alt)}


def main() -> None:
    est = es.load_estimation()
    z = f"z_{P}"
    out = {"primary_measure": P}

    # a) novelty -------------------------------------------------------------------------------
    est["z_novelty"] = est.novelty_M4 / est.novelty_M4.std(ddof=1)
    est["abs_tfut02"], est["abs_mp1"] = est.hf_TFUT02.abs(), est.hf_MP1.abs()
    out["a_novelty_abs_response"] = ols_wild(est, "abs_tfut02", ["z_novelty", "abs_mp1"])

    # b) placebo on random non-FOMC days ----------------------------------------------------------
    f = pd.read_csv(MARKET / "fred_DGS2.csv")
    f.columns = ["date", "v"]
    f["date"] = pd.to_datetime(f["date"])
    y2 = (pd.to_numeric(f.set_index("date")["v"], errors="coerce").dropna() * 100).sort_index()
    dy = y2.diff().dropna()
    allev = pd.read_csv(PROCESSED / "events.csv", parse_dates=["date"])
    fomc_pos = set()
    for d in allev.date:
        if d in dy.index:
            p = dy.index.get_loc(d)
            fomc_pos.update(range(p - 5, p + 6))
    lo, hi = dy.index.get_loc(dy.index[dy.index >= est.date.min()][0]), dy.index.get_loc(dy.index[dy.index <= est.date.max()][-1])
    pool = np.array([p for p in range(lo, hi + 1) if p not in fomc_pos])
    zz = est[z].to_numpy(float)
    actual = ols_wild(est.assign(d_dgs2=est.d_DGS2), "d_dgs2", [z])
    rng = np.random.default_rng(77)
    betas = []
    for _ in range(2000):
        yv = dy.iloc[rng.choice(pool, size=len(zz), replace=False)].to_numpy(float)
        X = np.column_stack([np.ones(len(zz)), zz])
        betas.append(np.linalg.lstsq(X, yv, rcond=None)[0][1])
    betas = np.array(betas)
    out["b_placebo_non_fomc_days"] = {"actual_daily_beta_no_mp1": actual, "n_placebo_draws": len(betas),
                                      "placebo_pool_days": int(len(pool)),
                                      "placebo_beta_mean": float(betas.mean()), "placebo_beta_sd": float(betas.std(ddof=1)),
                                      "share_placebo_abs_ge_actual": float(np.mean(np.abs(betas) >= abs(actual["beta"])))}

    # c) 1994-1999 statements ---------------------------------------------------------------------
    tone = pd.read_csv(PROCESSED / "tone_long.csv", dtype={"id": str}).set_index(["measure", "cutoff", "id"])
    allev = pd.read_csv(PROCESSED / "events.csv", dtype={"id": str}, parse_dates=["date"])
    sched = allev[allev.event_type == "scheduled"].sort_values("date")
    prev = dict(zip(sched.id, sched.id.shift(1)))
    early = sched[(sched.date < "1999-05-18") & sched.id.map(prev).notna()].copy()
    for m, cut in (("M3", 1999), ("M1", 0)):
        early[f"S_{m}"] = [tone.loc[(m, cut, i), "tau_count"] - tone.loc[(m, cut, prev[i]), "tau_count"] for i in early.id]
        early[f"z_{m}"] = early[f"S_{m}"] / est[f"S_{m}"].std(ddof=1)
    ext = pd.concat([early, est], ignore_index=True)
    out["c_extended_1994_2022"] = {"n_added_pre_1999": int(len(early)),
                                   "M3": ols_wild(ext, "hf_TFUT02", ["z_M3", "hf_MP1"], alt="greater"),
                                   "M1": ols_wild(ext, "hf_TFUT02", ["z_M1", "hf_MP1"], alt="greater"),
                                   "note": "M3 for 1994-1998 statements uses the 1999 model (labels 1996-1998): partial look-ahead"}

    # d) H5 information-effect interaction --------------------------------------------------------
    est["cbi_meeting"] = (est.jk_CBI_pm.abs() > 0).astype(float)
    est["z_x_cbi"] = est[z] * est.cbi_meeting
    out["d_H5_cbi_interaction"] = {"n_cbi_meetings": int(est.cbi_meeting.sum()),
                                   "interaction": ols_wild(est, "hf_SP500", ["z_x_cbi", z, "jk_pc1", "cbi_meeting"]),
                                   "z_in_monetary_meetings": ols_wild(est, "hf_SP500", [z, "z_x_cbi", "jk_pc1", "cbi_meeting"])}

    (REPORTS / "results_plan_completion.json").write_text(json.dumps(out, indent=2, default=float))
    a = out["a_novelty_abs_response"]
    print(f"a) novelty -> |2y|: beta {a['beta']:+.3f} (se {a['se_hc3']:.3f}) p {a['p_wild']:.4f} n {a['n']}")
    b = out["b_placebo_non_fomc_days"]
    print(f"b) actual daily beta {b['actual_daily_beta_no_mp1']['beta']:+.3f} (p {b['actual_daily_beta_no_mp1']['p_wild']:.4f}); "
          f"placebo mean {b['placebo_beta_mean']:+.3f} sd {b['placebo_beta_sd']:.3f}; share |placebo| >= |actual| = {b['share_placebo_abs_ge_actual']:.4f}")
    c = out["c_extended_1994_2022"]
    print(f"c) +{c['n_added_pre_1999']} pre-1999 statements: M3 beta {c['M3']['beta']:+.3f} p {c['M3']['p_wild']:.4f} n {c['M3']['n']}; "
          f"M1 {c['M1']['beta']:+.3f} p {c['M1']['p_wild']:.4f}")
    dd = out["d_H5_cbi_interaction"]
    print(f"d) CBI meetings {dd['n_cbi_meetings']}; interaction {dd['interaction']['beta']:+.4f} p {dd['interaction']['p_wild']:.4f}; "
          f"z in MP meetings {dd['z_in_monetary_meetings']['beta']:+.4f} p {dd['z_in_monetary_meetings']['p_wild']:.4f}")


if __name__ == "__main__":
    main()
