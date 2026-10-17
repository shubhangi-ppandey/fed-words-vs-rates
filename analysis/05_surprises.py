"""Step 5: tone surprises, statement novelty and Gurkaynak-Sack-Swanson (GSS) target/path factors.

Tone surprise (primary, pre-registered): S_t = tau_t - tau_prev, where prev is the previous SCHEDULED statement.
- Fixed-scorer measures (M1, M2, M2b): one scorer for every statement.
- Chronological measures (M3, M3b): for a meeting in year Y the SAME model (trained on labels with year < Y;
  Y capped at 2023) scores both the current and the previous statement, so S_t uses no information after Y-1.

Novelty (M4): 1 - cosine similarity of word-count vectors between consecutive scheduled statements
(no learned weights, hence no look-ahead).

GSS factors: principal components of the 30-min surprises in MP1, FF4, ED2, ED3 (complete for all main-sample
meetings; ED4 has gaps in 1999-2003, see deviations.md #8), rotated so the second factor does not move MP1, and
scaled so the target factor moves MP1 and the path factor moves ED3 one-for-one (bp). Loadings are estimated on
the ESTIMATION sample only and applied unchanged to the holdout.

Output: data/processed/events_model.csv  (main-sample meetings, one row each)
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED  # noqa: E402

MEASURES_FIXED = ["M1", "M2", "M2b"]
MEASURES_CHRONO = ["M3", "M3b"]
CUT_MIN, CUT_MAX = 1999, 2023
CB_MIN = 2000  # ChronoBERT M5: first vintage 1999-12-31, so meetings from 2000
GSS_COLS = ["hf_MP1", "hf_FF4", "hf_ED2", "hf_ED3"]


def cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], float)
    vb = np.array([b.get(k, 0) for k in keys], float)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def gss_factors(X_est: np.ndarray, X_all: np.ndarray, mp1_col: int, path_col: int):
    """Return (target, path) factor scores for X_all, with loadings from X_est."""
    mu, sd = X_est.mean(0), X_est.std(0, ddof=1)
    Z_est, Z_all = (X_est - mu) / sd, (X_all - mu) / sd
    _, _, vt = np.linalg.svd(Z_est, full_matrices=False)
    W = vt[:2].T                                    # first two principal directions
    F_est = Z_est @ W
    F_est_sd = F_est.std(0, ddof=1)
    F_est, F_all = F_est / F_est_sd, (Z_all @ W) / F_est_sd
    # loadings of MP1 on the two unit-variance factors
    gamma = np.linalg.lstsq(F_est, Z_est[:, mp1_col], rcond=None)[0]
    b = np.array([gamma[1], -gamma[0]]) / np.linalg.norm(gamma)   # path direction: orthogonal to MP1's loading
    a = gamma / np.linalg.norm(gamma)                               # target direction
    U = np.column_stack([a, b])
    T_est, T_all = F_est @ U, F_all @ U
    # scale: target moves MP1 1:1, path moves the path instrument 1:1 (in bp)
    s_t = np.linalg.lstsq(T_est[:, [0]], X_est[:, mp1_col] - X_est[:, mp1_col].mean(), rcond=None)[0][0]
    s_p = np.linalg.lstsq(T_est[:, [1]], X_est[:, path_col] - X_est[:, path_col].mean(), rcond=None)[0][0]
    return T_all[:, 0] * s_t, T_all[:, 1] * s_p, U, (s_t, s_p)


def main() -> None:
    ev = pd.read_csv(PROCESSED / "events.csv", dtype={"id": str}, parse_dates=["date"])
    tone = pd.read_csv(PROCESSED / "tone_long.csv", dtype={"id": str})
    sent = pd.read_csv(PROCESSED / "sentences.csv", dtype={"id": str})

    sched = ev[ev.event_type == "scheduled"].sort_values("date").reset_index(drop=True)
    sched["prev_id"] = sched["id"].shift(1)
    main_ = sched[sched.main_sample == 1].copy()

    tl = tone.set_index(["measure", "cutoff", "id"])
    fixed, chrono = list(MEASURES_FIXED), list(MEASURES_CHRONO)
    if "M5" in set(tone.measure):  # ChronoBERT extension E3 (defined for meetings from 2000)
        fixed.append("M5f")
        chrono.append("M5")
    for m in fixed + chrono:
        for var in ("tau_count", "tau_prob"):
            if m == "M1" and var == "tau_prob":
                continue
            cur, prev = [], []
            for _, r in main_.iterrows():
                if m == "M5" and r.date.year < CB_MIN:
                    cur.append(np.nan)
                    prev.append(np.nan)
                    continue
                cut = 0 if m in fixed else int(min(max(r.date.year, CUT_MIN), CUT_MAX))
                cur.append(tl.loc[(m, cut, r.id), var])
                prev.append(tl.loc[(m, cut, r.prev_id), var])
            suffix = "" if var == "tau_count" else "_prob"
            main_[f"tau_{m}{suffix}"] = cur
            main_[f"S_{m}{suffix}"] = np.array(cur) - np.array(prev)

    # M4 novelty
    kept = sent[sent.boilerplate == 0]
    bags = {i: Counter(re.findall(r"[a-z]+", " ".join(g.text).lower())) for i, g in kept.groupby("id")}
    main_["novelty_M4"] = [1 - cosine(bags[a], bags[b]) for a, b in zip(main_.id, main_.prev_id)]

    # GSS target / path factors
    est = (main_.holdout == 0).to_numpy()
    X = main_[GSS_COLS].to_numpy(float)
    tgt, path, U, scales = gss_factors(X[est], X, GSS_COLS.index("hf_MP1"), GSS_COLS.index("hf_ED3"))
    main_["gss_target"], main_["gss_path"] = tgt, path

    main_.to_csv(PROCESSED / "events_model.csv", index=False)

    e = main_[est]
    print(f"main-sample meetings: {len(main_)} (estimation {int(est.sum())}, holdout {int((~est).sum())})")
    print("first meeting and its previous statement:", main_.iloc[0][["id", "prev_id"]].tolist())
    s_cols = [c for c in main_.columns if c.startswith("S_")]
    print("tone-surprise SD (estimation):", e[s_cols].std().round(3).to_dict())
    print("Spearman correlation of count-based surprises (estimation):")
    print(e[[f"S_{m}" for m in ["M1", "M2b", "M3b", "M2", "M3"]]].corr("spearman").round(2).to_string())
    print("GSS check (estimation): corr(target, MP1) =", round(np.corrcoef(e.gss_target, e.hf_MP1)[0, 1], 3),
          "| corr(path, MP1) =", round(np.corrcoef(e.gss_path, e.hf_MP1)[0, 1], 3),
          "| corr(target, path) =", round(np.corrcoef(e.gss_target, e.gss_path)[0, 1], 3))


if __name__ == "__main__":
    main()
