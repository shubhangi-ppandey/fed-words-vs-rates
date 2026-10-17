"""Step 7b (pre-registered extension E3, deviations.md #12): separating pretraining from label look-ahead.

Common estimation sample 2000-01 .. 2022-12 (ChronoBERT's first vintage is 1999-12-31). H1 specification
(30-min 2-year yield on the standardised tone surprise + MP1); surprises are re-standardised on this sample.
Contrasts (paired event bootstrap, 9,999 draws; hypothesised direction = larger positive coefficient):
- M5f - M5 : ChronoBERT, full-information (2024 vintage + all labels) minus fully chronological
             -> total look-ahead (pretraining + labels) within one architecture
- M2  - M3 : RoBERTa-large, label look-ahead only (as in step 7, on the common sample)
- M3  - M5 : label-safe RoBERTa (pretraining to 2019) minus fully chronological ChronoBERT
             -> pretraining look-ahead, confounded with architecture (stated as such)
Output: reports/results_lookahead_chronobert.json
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stats  # noqa: E402
from common.paths import PROCESSED, REPORTS  # noqa: E402

PAIRS = [("ChronoBERT: full - chronological", "M5f", "M5"),
         ("RoBERTa-large: full - chronological", "M2", "M3"),
         ("label-safe RoBERTa - fully chronological ChronoBERT", "M3", "M5")]
MEASURES = ["M1", "M2b", "M3b", "M2", "M3", "M5f", "M5"]


def main() -> None:
    ev = pd.read_csv(PROCESSED / "events_model.csv", dtype={"id": str}, parse_dates=["date"])
    d = ev[(ev.holdout == 0) & (ev.date >= "2000-01-01")].copy()
    assert d.date.max() < pd.Timestamp("2023-01-01")
    for m in MEASURES:
        d[f"z_{m}"] = d[f"S_{m}"] / d[f"S_{m}"].std(ddof=1)
    d = d.dropna(subset=[f"z_{m}" for m in MEASURES] + ["hf_TFUT02", "hf_MP1"])
    Y = d.hf_TFUT02.to_numpy(float)
    X = {m: np.column_stack([np.ones(len(d)), d[f"z_{m}"], d.hf_MP1]) for m in MEASURES}
    betas = {}
    for m in MEASURES:
        r = stats.ols(Y, X[m], ["const", "z", "mp1"])
        betas[m] = {"beta": r["coef"]["z"], "se_hc3": r["se"]["z"],
                    "p_wild_greater": stats.wild_boot(Y, X[m], 1, B=9999, seed=13, alternative="greater")}
    contrasts = []
    for label, a, b in PAIRS:
        pb = stats.paired_boot(Y, X[a], X[b], 1, B=9999, seed=17)
        contrasts.append({"contrast": label, "a": a, "b": b, **pb})
    out = {"sample": {"n": int(len(d)), "start": str(d.date.min().date()), "end": str(d.date.max().date())},
           "betas": betas, "contrasts": contrasts,
           "surprise_correlations": d[[f"S_{m}" for m in MEASURES]].corr().round(4).to_dict()}
    (REPORTS / "results_lookahead_chronobert.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"common sample n = {len(d)}")
    for m, v in betas.items():
        print(f"  {m:4} beta {v['beta']:+.3f} (se {v['se_hc3']:.3f}) p {v['p_wild_greater']:.4f}")
    for c in contrasts:
        print(f"  {c['contrast']:52} diff {c['diff']:+.3f} CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
