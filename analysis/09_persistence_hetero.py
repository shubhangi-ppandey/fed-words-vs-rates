"""Step 9: persistence (local projections), ZLB heterogeneity (H6) and industries (H7); estimation sample only.

- Local projections (Jorda 2005): cumulative change from the close before the meeting to h trading days after
  (h = 0..20) in the 2- and 10-year Treasury yields (FRED, bp) and the cumulative market return (Ken French, %),
  on the standardised primary tone surprise + MP1. Newey-West SE (1 lag: meetings are ~6 weeks apart).
- H6: 30-min 2-year yield on z, ZLB dummy and z x ZLB (+ MP1); two-sided wild bootstrap on the interaction (plan).
- H7: daily return of each of the 49 Ken French industries on z, MP1 and the market return; Holm across 49.
  Pre-specified rate-sensitive group (Util, RlEst, Banks) vs the other 46: difference of mean coefficients with an
  event-bootstrap CI.

Outputs: reports/results_persistence_hetero.json
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
PRIMARY = es.PRIMARY
H_MAX = 20
RATE_SENSITIVE = ["Util", "RlEst", "Banks"]


def daily_series():
    out = {}
    for sid in ("DGS2", "DGS10"):
        f = pd.read_csv(MARKET / f"fred_{sid}.csv")
        f.columns = ["date", "v"]
        f["date"] = pd.to_datetime(f["date"])
        out[sid] = pd.to_numeric(f.set_index("date")["v"], errors="coerce").dropna() * 100.0  # bp
    ff3 = pd.read_csv(MARKET / "kf_ff3_daily.csv", parse_dates=["date"]).set_index("date")
    out["mkt"] = (ff3["Mkt-RF"] + ff3["RF"]).dropna()
    return out


def cum_change(s: pd.Series, date: pd.Timestamp, h: int, is_return: bool):
    if date not in s.index:
        return np.nan
    pos = s.index.get_loc(date)
    if pos < 1 or pos + h >= len(s):
        return np.nan
    if is_return:
        return float(s.iloc[pos:pos + h + 1].sum())
    return float(s.iloc[pos + h] - s.iloc[pos - 1])


def main() -> None:
    est = es.load_estimation()
    z = f"z_{PRIMARY}"
    out = {"primary_measure": PRIMARY, "local_projections": {}, "H6": {}, "H7": {}}

    # --- local projections ---------------------------------------------------------------------
    series = daily_series()
    for name, s in series.items():
        irf = []
        for h in range(H_MAX + 1):
            y = est.date.map(lambda d: cum_change(s, d, h, name == "mkt"))
            d = pd.DataFrame({"y": y, "z": est[z], "mp1": est.hf_MP1}).dropna()
            X = np.column_stack([np.ones(len(d)), d.z, d.mp1])
            r = stats.newey_west(d.y.to_numpy(float), X, lags=1, names=["const", "z", "mp1"])
            irf.append({"h": h, "n": r["n"], "beta": r["coef"]["z"], "se": r["se"]["z"], "p": r["p"]["z"]})
        out["local_projections"][name] = irf

    # --- pre-event placebo horizons and pre-trend-controlled LPs (added after the M8 review; deviations.md #19)
    #     pre-window change over the k days BEFORE the meeting (close t-1 minus close t-1-k; returns t-k..t-1)
    out["pre_event_placebo"], out["local_projections_pretrend_controlled"] = {}, {}
    for name, s in series.items():
        is_ret = name == "mkt"

        def pre(d, k, s=s, is_ret=is_ret):
            if d not in s.index:
                return np.nan
            pos = s.index.get_loc(d)
            if pos - 1 - k < 0:
                return np.nan
            return float(s.iloc[pos - k:pos].sum()) if is_ret else float(s.iloc[pos - 1] - s.iloc[pos - 1 - k])
        rows_pre = []
        for k in range(1, H_MAX + 1):
            y = est.date.map(lambda d, k=k: pre(d, k))
            d = pd.DataFrame({"y": y, "z": est[z], "mp1": est.hf_MP1}).dropna()
            X = np.column_stack([np.ones(len(d)), d.z, d.mp1])
            r = stats.newey_west(d.y.to_numpy(float), X, lags=1, names=["const", "z", "mp1"])
            rows_pre.append({"k": k, "n": r["n"], "beta": r["coef"]["z"], "se": r["se"]["z"], "p": r["p"]["z"]})
        out["pre_event_placebo"][name] = rows_pre
        pretrend = est.date.map(lambda d: pre(d, H_MAX))
        rows_c = []
        for h in range(H_MAX + 1):
            y = est.date.map(lambda d, h=h: cum_change(s, d, h, is_ret))
            d = pd.DataFrame({"y": y, "z": est[z], "mp1": est.hf_MP1, "pre": pretrend}).dropna()
            X = np.column_stack([np.ones(len(d)), d.z, d.mp1, d.pre])
            r = stats.newey_west(d.y.to_numpy(float), X, lags=1, names=["const", "z", "mp1", "pre"])
            rows_c.append({"h": h, "n": r["n"], "beta": r["coef"]["z"], "se": r["se"]["z"], "p": r["p"]["z"]})
        out["local_projections_pretrend_controlled"][name] = rows_c

    # --- H6: ZLB interaction -------------------------------------------------------------------
    d = est[["hf_TFUT02", z, "zlb", "hf_MP1"]].dropna()
    X = np.column_stack([np.ones(len(d)), d[z] * d.zlb, d[z], d.zlb, d.hf_MP1])
    r = stats.ols(d.hf_TFUT02.to_numpy(float), X, ["const", "z_x_zlb", "z", "zlb", "mp1"])
    out["H6"] = {"n": r["n"], "n_zlb": int(d.zlb.sum()), "beta_interaction": r["coef"]["z_x_zlb"],
                 "se_hc3": r["se"]["z_x_zlb"], "beta_z_outside_zlb": r["coef"]["z"],
                 "p_wild": stats.wild_boot(d.hf_TFUT02.to_numpy(float), X, 1, B=9999, seed=21),  # two-sided (plan)
                 "p_wild_greater": stats.wild_boot(d.hf_TFUT02.to_numpy(float), X, 1, B=9999, seed=21, alternative="greater")}

    # --- H7: industries ------------------------------------------------------------------------
    ind = pd.read_csv(PROCESSED / "events_industries_daily.csv", dtype={"id": str})
    names = [c for c in ind.columns if c not in ("id", "date")]
    m = est[["id", z, "hf_MP1", "d_mkt"]].merge(ind, on="id").dropna(subset=[z, "hf_MP1", "d_mkt"])
    X = np.column_stack([np.ones(len(m)), m[z], m.hf_MP1, m.d_mkt])
    rows = []
    for c in names:
        dd = m[[c]].notna().to_numpy().ravel()
        Y, Xc = m[c].to_numpy(float)[dd], X[dd]
        r = stats.ols(Y, Xc, ["const", "z", "mp1", "mkt"])
        rows.append({"industry": c.strip(), "n": r["n"], "beta": r["coef"]["z"], "se_hc3": r["se"]["z"],
                     "p_wild": stats.wild_boot(Y, Xc, 1, B=1999, seed=31)})
    for r_, ph in zip(rows, stats.holm([r["p_wild"] for r in rows])):
        r_["p_holm"] = ph
    betas = np.array([r["beta"] for r in rows])
    grp = np.array([r["industry"] in RATE_SENSITIVE for r in rows])
    rng = np.random.default_rng(41)
    Ymat = m[[c for c in names]].to_numpy(float)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(m), len(m))
        bb = np.linalg.lstsq(X[i], np.nan_to_num(Ymat[i]), rcond=None)[0][1]
        boots.append(bb[grp].mean() - bb[~grp].mean())
    out["H7"] = {"n_events": int(len(m)), "industries": rows,
                 "n_holm_significant": int(sum(r["p_holm"] < 0.05 for r in rows)),
                 "rate_sensitive_group": RATE_SENSITIVE,
                 "group_minus_rest": float(betas[grp].mean() - betas[~grp].mean()),
                 "group_minus_rest_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    (REPORTS / "results_persistence_hetero.json").write_text(json.dumps(out, indent=2, default=float))

    for k, v in out["local_projections"].items():
        print(k, " ".join(f"h{x['h']}:{x['beta']:+.2f}" for x in v if x["h"] in (0, 1, 5, 10, 20)))
        pre_ = {x["k"]: x for x in out["pre_event_placebo"][k]}
        ctl = {x["h"]: x for x in out["local_projections_pretrend_controlled"][k]}
        print("   pre-event k5/k10/k20:", " ".join(f"{pre_[q]['beta']:+.2f}(p{pre_[q]['p']:.3f})" for q in (5, 10, 20)),
              "| pre-trend-controlled h0/h5/h20:", " ".join(f"{ctl[q]['beta']:+.2f}(p{ctl[q]['p']:.3f})" for q in (0, 5, 20)))
    print("H6:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["H6"].items()})
    print("H7: Holm-significant industries:", out["H7"]["n_holm_significant"],
          "| group - rest:", round(out["H7"]["group_minus_rest"], 4), [round(x, 4) for x in out["H7"]["group_minus_rest_ci95"]])


if __name__ == "__main__":
    main()
