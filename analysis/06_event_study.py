"""Step 6: core event-study results on the ESTIMATION sample (1999-05 .. 2022-12); the holdout is never loaded.

Pre-registered tests (PROJECT_PLAN.md section 1; deviations.md #10 fixes M3 as the primary measure):
- H1  30-min 2-year Treasury yield change on the tone surprise, controlling for the target surprise MP1
      (one-sided, beta > 0: the only hypothesis stated one-sided in the plan)
- H2  GSS path factor on the tone surprise, controlling for the GSS target factor (two-sided p; supported if beta > 0)
- H5  30-min S&P 500 return on the tone surprise, controlling for the JK policy factor pc1 (two-sided p; beta < 0)
- RQ3 channel table: 2/5/10/30-year Treasury yields, S&P 500 and EUR on the tone surprise + MP1
      (two-sided, Holm-adjusted within the family of 6 outcomes)

Tone surprises are standardised by their estimation-sample SD, so beta = response (bp or %) per 1-SD surprise.
Inference: HC3, restricted wild bootstrap (9,999 Rademacher draws) and a permutation test (9,999 shuffles).

Outputs: reports/results_core.json, reports/tables/core_results.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stats  # noqa: E402
from common.paths import PROCESSED, REPORTS  # noqa: E402

PRIMARY = "M3"
MEASURES = ["M1", "M2b", "M3b", "M2", "M3"]
B = 9999
CHANNELS = ["hf_TFUT02", "hf_TFUT05", "hf_TFUT10", "hf_TFUT30", "hf_SP500", "hf_EUR"]


def load_estimation() -> pd.DataFrame:
    ev = pd.read_csv(PROCESSED / "events_model.csv", dtype={"id": str}, parse_dates=["date"])
    est = ev[ev.holdout == 0].copy()
    assert est.date.max() < pd.Timestamp("2023-01-01"), "holdout leaked into the estimation sample"
    for m in MEASURES:
        for suf in ("", "_prob"):
            c = f"S_{m}{suf}"
            if c in est:
                est[f"z{c[1:]}"] = est[c] / est[c].std(ddof=1)
    return est


def fit(est, y, x, controls, alternative, seed=1):
    d = est[[y, x] + controls].dropna()
    Y = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in [x] + controls])
    r = stats.ols(Y, X, ["const", x] + controls)
    return {"outcome": y, "regressor": x, "controls": controls, "n": r["n"], "r2": round(r["r2"], 4),
            "beta": r["coef"][x], "se_hc3": r["se"][x], "t_hc3": r["t"][x], "p_hc3_two_sided": r["p"][x],
            "p_wild": stats.wild_boot(Y, X, 1, B=B, seed=seed, alternative=alternative),
            "p_perm": stats.permutation(Y, X, 1, B=B, seed=seed + 1, alternative=alternative),
            "alternative": alternative}


def main() -> None:
    est = load_estimation()
    rows = []
    for m in MEASURES:
        zx = f"z_{m}"
        rows.append({"test": "H1", "measure": m, **fit(est, "hf_TFUT02", zx, ["hf_MP1"], "greater")})
        rows.append({"test": "H2", "measure": m, **fit(est, "gss_path", zx, ["gss_target"], "two-sided")})
        rows.append({"test": "H5", "measure": m, **fit(est, "hf_SP500", zx, ["jk_pc1"], "two-sided")})
        fam = [fit(est, y, zx, ["hf_MP1"], "two-sided") for y in CHANNELS]
        for r, ph in zip(fam, stats.holm([f["p_wild"] for f in fam])):
            rows.append({"test": "RQ3", "measure": m, **r, "p_wild_holm": ph})
    res = pd.DataFrame(rows)

    # incremental R2 of the tone surprise for H2 (path factor), primary measure, with bootstrap CI
    d = est[["gss_path", "gss_target", f"z_{PRIMARY}"]].dropna()
    rng = np.random.default_rng(11)

    def r2(dd, with_s):
        X = np.column_stack([np.ones(len(dd)), dd.gss_target] + ([dd[f"z_{PRIMARY}"]] if with_s else []))
        e = dd.gss_path - X @ np.linalg.lstsq(X, dd.gss_path, rcond=None)[0]
        return 1 - (e ** 2).sum() / ((dd.gss_path - dd.gss_path.mean()) ** 2).sum()
    inc = r2(d, True) - r2(d, False)
    boots = [r2(d.iloc[i], True) - r2(d.iloc[i], False) for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]

    tables = REPORTS / "tables"
    tables.mkdir(exist_ok=True)
    res.to_csv(tables / "core_results.csv", index=False)
    out = {"primary_measure": PRIMARY, "sample": {"n_events": int(len(est)), "start": str(est.date.min().date()),
                                                  "end": str(est.date.max().date())},
           "surprise_sd": {m: float(est[f"S_{m}"].std(ddof=1)) for m in MEASURES},
           "H2_incremental_r2": {"value": float(inc), "ci95": [float(np.percentile(boots, 2.5)),
                                                                float(np.percentile(boots, 97.5))]},
           "results": res.to_dict(orient="records")}
    (REPORTS / "results_core.json").write_text(json.dumps(out, indent=2, default=float))

    show = res[["test", "measure", "outcome", "n", "beta", "se_hc3", "p_wild", "p_perm"]].copy()
    if "p_wild_holm" in res:
        show["p_holm"] = res["p_wild_holm"]
    print(show.round(4).to_string(index=False))
    print(f"H2 incremental R2 ({PRIMARY}): {inc:.4f}  CI95 {np.percentile(boots, 2.5):.4f}..{np.percentile(boots, 97.5):.4f}")


if __name__ == "__main__":
    main()
