"""Step 10: pre-specified robustness checks of H1 (30-min 2-year yield on the primary tone surprise + MP1),
estimation sample unless stated (PROJECT_PLAN.md section 7).

Checks
 1 baseline                         as in step 6
 2 excl. 2008-09                    drop meetings in the financial-crisis years
 3 probability-weighted tone        S from mean(p_hawkish - p_dovish)
 4 no boilerplate filter            tone recomputed over ALL sentences (primary model and cutoff rule)
 5 AR(1)-residual surprise          tau_t minus its expanding-window AR(1) forecast (>= 20 past meetings)
 6 daily window                     daily change in the 2-year yield (FRED) instead of the 30-min change
 7 daily window, no press conference  as 6, dropping meetings followed by a press conference
 8 JK decomposition controls        MP_median and CBI_median instead of MP1
 9 Bauer-Swanson control            MPS_ORTH instead of MP1 (sample ends 2022)
10 GSS target factor control        gss_target instead of MP1
11 + unscheduled meetings           adds unscheduled statements with high-frequency data (1999+)
12 leave-one-year-out               range of beta when each calendar year is dropped in turn

Output: reports/results_robustness.json
"""
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stats  # noqa: E402
from common.paths import PROCESSED, RAW, REPORTS, USER_AGENT  # noqa: E402

es = importlib.import_module("06_event_study")
PRIMARY = es.PRIMARY
CUT_MIN, CUT_MAX = 1999, 2023
B = 4999


def presser_dates() -> set:
    """FOMC press-conference dates scraped from the Fed's calendar pages (cached in data/raw)."""
    cache = RAW / "press_conference_dates.txt"
    if cache.exists():
        return set(cache.read_text().split())
    dates, s = set(), requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    pages = [f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm" for y in range(2011, 2021)]
    pages.append("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    for u in pages:
        dates |= set(re.findall(r"fomcpresconf(\d{8})\.htm", s.get(u, timeout=40).text))
        time.sleep(0.6)
    cache.write_text("\n".join(sorted(dates)))
    return dates


def run(d: pd.DataFrame, y: str, x: str, controls: list, label: str) -> dict:
    d = d[[y, x] + controls].dropna()
    Y = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in [x] + controls])
    r = stats.ols(Y, X, ["const", x] + controls)
    return {"check": label, "n": r["n"], "beta": r["coef"][x], "se_hc3": r["se"][x],
            "p_wild_greater": stats.wild_boot(Y, X, 1, B=B, seed=7, alternative="greater")}


def tone_no_filter(sent: pd.DataFrame, ids: set) -> pd.DataFrame:
    """Primary-measure tone over ALL sentences, one value per (cutoff, statement)."""
    tp = PROCESSED / "transformer_preds"
    rows = []
    for Y in range(CUT_MIN, CUT_MAX + 1):
        p = pd.read_csv(tp / f"chrono_{Y}.csv", dtype={"id": str})
        p = p[p.id.isin(ids)]
        lab = p[["pd", "ph", "pn"]].to_numpy().argmax(1)
        p = p.assign(h=(lab == 1).astype(float), dv=(lab == 0).astype(float))
        g = p.groupby("id")
        rows.append(pd.DataFrame({"cutoff": Y, "tau": g.h.mean() - g.dv.mean()}).reset_index())
    return pd.concat(rows).set_index(["cutoff", "id"])["tau"]


def main() -> None:
    est = es.load_estimation()
    z = f"z_{PRIMARY}"
    sd = est[f"S_{PRIMARY}"].std(ddof=1)
    res = [run(est, "hf_TFUT02", z, ["hf_MP1"], "1 baseline")]
    res.append(run(est[~est.date.dt.year.isin([2008, 2009])], "hf_TFUT02", z, ["hf_MP1"], "2 excl. 2008-09"))
    res.append(run(est, "hf_TFUT02", f"z_{PRIMARY}_prob", ["hf_MP1"], "3 probability-weighted tone"))

    # 4 no boilerplate filter
    ev_all = pd.read_csv(PROCESSED / "events.csv", dtype={"id": str}, parse_dates=["date"])
    sched = ev_all[ev_all.event_type == "scheduled"].sort_values("date")
    prev = dict(zip(sched.id, sched.id.shift(1)))
    sent = pd.read_csv(PROCESSED / "sentences.csv", dtype={"id": str})
    tnf = tone_no_filter(sent, set(sent.id))
    cut = est.date.dt.year.clip(CUT_MIN, CUT_MAX)
    s_nf = [tnf.loc[(c, i)] - tnf.loc[(c, prev[i])] for c, i in zip(cut, est.id)]
    est["z_nofilter"] = np.array(s_nf) / np.std(s_nf, ddof=1)
    res.append(run(est, "hf_TFUT02", "z_nofilter", ["hf_MP1"], "4 no boilerplate filter"))

    # 5 AR(1)-residual surprise on the primary tone level (expanding window, past meetings only)
    tau = est[f"tau_{PRIMARY}"].to_numpy()
    ar = np.full(len(est), np.nan)
    for t in range(21, len(est)):
        yv, xv = tau[1:t], tau[:t - 1]
        b = np.linalg.lstsq(np.column_stack([np.ones(len(xv)), xv]), yv, rcond=None)[0]
        ar[t] = tau[t] - (b[0] + b[1] * tau[t - 1])
    est["z_ar"] = ar / np.nanstd(ar, ddof=1)
    res.append(run(est, "hf_TFUT02", "z_ar", ["hf_MP1"], "5 AR(1)-residual surprise"))

    # 6-7 daily windows
    res.append(run(est, "d_DGS2", z, ["hf_MP1"], "6 daily window (FRED 2y)"))
    pc = presser_dates()
    no_pc = est[~est.date.dt.strftime("%Y%m%d").isin(pc)]
    res.append(run(no_pc, "d_DGS2", z, ["hf_MP1"], "7 daily window, no press conference"))

    # 8-10 alternative policy-surprise controls
    res.append(run(est, "hf_TFUT02", z, ["jk_MP_median", "jk_CBI_median"], "8 JK MP/CBI controls"))
    res.append(run(est, "hf_TFUT02", z, ["bs_MPS_ORTH"], "9 Bauer-Swanson MPS_ORTH control"))
    res.append(run(est, "hf_TFUT02", z, ["gss_target"], "10 GSS target-factor control"))

    # 11 add unscheduled statements (with HF data, 1999 to 2022)
    tone = pd.read_csv(PROCESSED / "tone_long.csv", dtype={"id": str}).set_index(["measure", "cutoff", "id"])
    uns = ev_all[(ev_all.event_type == "unscheduled_or_other") & (ev_all.date >= "1999-05-18")
                 & (ev_all.date < "2023-01-01")].copy()
    all_st = ev_all[ev_all.event_type.isin(["scheduled", "unscheduled_or_other"])].sort_values("date")
    prev_any = dict(zip(all_st.id, all_st.id.shift(1)))
    ucut = uns.date.dt.year.clip(CUT_MIN, CUT_MAX)
    uns["S"] = [tone.loc[(PRIMARY, int(c), i), "tau_count"] - tone.loc[(PRIMARY, int(c), prev_any[i]), "tau_count"]
                for c, i in zip(ucut, uns.id)]
    uns[z] = uns["S"] / sd
    both = pd.concat([est[["hf_TFUT02", z, "hf_MP1"]], uns[["hf_TFUT02", z, "hf_MP1"]]])
    res.append(run(both, "hf_TFUT02", z, ["hf_MP1"], f"11 + {len(uns)} unscheduled statements"))

    # 13 drop the first meeting (its previous statement is 6 months earlier; added after the M8 review)
    res.append(run(est.iloc[1:], "hf_TFUT02", z, ["hf_MP1"], f"13 drop first meeting ({est.date.iloc[0].date()}, 6-month gap)"))

    # Freedman-Lane permutation p for the baseline (alternative to the pre-specified Manly-style shuffle)
    d0 = est[["hf_TFUT02", z, "hf_MP1"]].dropna()
    X0 = np.column_stack([np.ones(len(d0)), d0[z], d0.hf_MP1])
    fl_p = stats.freedman_lane(d0.hf_TFUT02.to_numpy(float), X0, 1, B=9999, seed=9, alternative="greater")

    # 12 leave-one-year-out
    loo = []
    for yr in sorted(est.date.dt.year.unique()):
        d = est[est.date.dt.year != yr][["hf_TFUT02", z, "hf_MP1"]].dropna()
        X = np.column_stack([np.ones(len(d)), d[z], d.hf_MP1])
        loo.append({"dropped_year": int(yr), "beta": float(np.linalg.lstsq(X, d.hf_TFUT02, rcond=None)[0][1])})
    out = {"primary_measure": PRIMARY, "checks": res,
           "leave_one_year_out": {"min": min(x["beta"] for x in loo), "max": max(x["beta"] for x in loo), "by_year": loo},
           "n_press_conference_dates": len(pc), "baseline_freedman_lane_p_greater": fl_p}
    (REPORTS / "results_robustness.json").write_text(json.dumps(out, indent=2, default=float))
    print(pd.DataFrame(res).round(4).to_string(index=False))
    print("leave-one-year-out beta range:", round(out["leave_one_year_out"]["min"], 4), "..",
          round(out["leave_one_year_out"]["max"], 4))


if __name__ == "__main__":
    main()
