"""Step 10b (added after the M8 review; NOT pre-registered; deviations.md #16): fine-tuning-seed robustness of H1.

For RoBERTa-large M2 (full sample) and M3 (chronological), tone surprises are rebuilt from the seed-42 models
(pre-registered), seeds 43 and 44 (04g), and from seed-averaged probabilities, with exactly the aggregation and
surprise rules of 04d/05 (count-based tau; one model scores both statements). H1 is re-estimated on the estimation
sample for each. Output: reports/results_seed_robustness.json
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

es = importlib.import_module("06_event_study")
SEEDS = {42: PROCESSED / "transformer_preds", 43: PROCESSED / "transformer_preds_seed43",
         44: PROCESSED / "transformer_preds_seed44"}
CUT_MIN, CUT_MAX = 1999, 2022


def tau_from(pred: pd.DataFrame, kept: pd.DataFrame) -> pd.Series:
    p = kept.merge(pred, on=["id", "sent_idx"])
    lab = p[["pd", "ph", "pn"]].to_numpy().argmax(1)
    p = p.assign(h=(lab == 1).astype(float), d=(lab == 0).astype(float))
    g = p.groupby("id")
    return g.h.mean() - g.d.mean()


def main() -> None:
    est = es.load_estimation()
    sent = pd.read_csv(PROCESSED / "sentences.csv", dtype={"id": str})
    kept = sent[sent.boilerplate == 0][["id", "sent_idx"]]
    ev = pd.read_csv(PROCESSED / "events.csv", dtype={"id": str}, parse_dates=["date"])
    sched = ev[ev.event_type == "scheduled"].sort_values("date")
    prev = dict(zip(sched.id, sched.id.shift(1)))
    rows, preds = [], {}
    for seed, folder in SEEDS.items():
        preds[seed] = {"full": pd.read_csv(folder / "full.csv", dtype={"id": str})}
        for Y in range(CUT_MIN, CUT_MAX + 1):
            preds[seed][Y] = pd.read_csv(folder / f"chrono_{Y}.csv", dtype={"id": str})
    avg = {}
    for key in preds[42]:
        base = preds[42][key][["id", "sent_idx"]].copy()
        base[["pd", "ph", "pn"]] = np.mean([preds[s][key][["pd", "ph", "pn"]].to_numpy() for s in SEEDS], axis=0)
        avg[key] = base
    variants = {**{f"seed {s}": preds[s] for s in SEEDS}, "seed-averaged": avg}

    for name, pr in variants.items():
        taus = {k: tau_from(v, kept) for k, v in pr.items()}
        for measure in ("M2", "M3"):
            S = []
            for _, r in est.iterrows():
                key = "full" if measure == "M2" else int(min(max(r.date.year, CUT_MIN), CUT_MAX))
                S.append(taus[key][r.id] - taus[key][prev[r.id]])
            S = np.array(S)
            d = est.assign(zs=S / S.std(ddof=1))
            Y = d.hf_TFUT02.to_numpy(float)
            X = np.column_stack([np.ones(len(d)), d.zs, d.hf_MP1])
            r = stats.ols(Y, X, ["const", "z", "mp1"])
            rows.append({"variant": name, "measure": measure, "n": r["n"], "beta": r["coef"]["z"], "se_hc3": r["se"]["z"],
                         "p_wild_greater": stats.wild_boot(Y, X, 1, B=9999, seed=3, alternative="greater"),
                         "corr_with_seed42_S": float(np.corrcoef(S, est[f"S_{measure}"])[0, 1])})
    out = {"rows": rows}
    for measure in ("M2", "M3"):
        b = [r["beta"] for r in rows if r["measure"] == measure and r["variant"].startswith("seed ")]
        out[f"{measure}_beta_range_across_seeds"] = [min(b), max(b)]
    (REPORTS / "results_seed_robustness.json").write_text(json.dumps(out, indent=2, default=float))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
