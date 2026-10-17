"""Step 8 (H4): the untouched holdout, 2023-01 .. 2026-09 (after the tone classifiers' label data, 1996-2022).

Run ONCE, after every estimation-sample analysis is final (milestone M4). The first run writes
reports/holdout_lock.json with a timestamp and a hash of the estimation results it was run against; later runs
refuse to overwrite it unless --force is given (any forced re-run is to be logged in deviations.md).

Specification = H1 (and H2, H5) exactly as in step 6. Tone surprises are standardised by the ESTIMATION-sample SD,
and the GSS loadings were estimated without the holdout (step 5). The minimum detectable effect (80% power,
5% two-sided) is reported because about 30 meetings give little power.

Output: reports/results_holdout.json
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stats  # noqa: E402
from common.paths import PROCESSED, REPORTS  # noqa: E402

MEASURES = ["M1", "M2b", "M3b", "M2", "M3"]
SPECS = [("H1", "hf_TFUT02", ["hf_MP1"], "greater"), ("H2", "gss_path", ["gss_target"], "two-sided"),
         ("H5", "hf_SP500", ["jk_pc1"], "two-sided")]
LOCK = REPORTS / "holdout_lock.json"


def main() -> None:
    if LOCK.exists() and "--force" not in sys.argv:
        print(f"holdout already evaluated ({json.loads(LOCK.read_text())['run_at']}); refusing to re-run")
        return
    core = (REPORTS / "results_core.json").read_bytes()
    ev = pd.read_csv(PROCESSED / "events_model.csv", dtype={"id": str}, parse_dates=["date"])
    est, hold = ev[ev.holdout == 0], ev[ev.holdout == 1].copy()
    for m in MEASURES:
        hold[f"z_{m}"] = hold[f"S_{m}"] / est[f"S_{m}"].std(ddof=1)
    core_json = json.loads(core)
    est_beta = {(r["test"], r["measure"]): r["beta"] for r in core_json["results"] if r["test"] in ("H1", "H2", "H5")}
    # extension E3: ChronoBERT measures, with estimation betas from the same H1/H2/H5 specifications
    measures = list(MEASURES)
    if "S_M5" in ev:
        for m in ("M5f", "M5"):
            measures.append(m)
            hold[f"z_{m}"] = hold[f"S_{m}"] / est[f"S_{m}"].std(ddof=1)
            e = est.assign(z=est[f"S_{m}"] / est[f"S_{m}"].std(ddof=1))
            for test, y, ctrl, _ in SPECS:
                d = e[[y, "z"] + ctrl].dropna()
                Xe = np.column_stack([np.ones(len(d)), d.z] + [d[c].to_numpy(float) for c in ctrl])
                est_beta[(test, m)] = float(np.linalg.lstsq(Xe, d[y].to_numpy(float), rcond=None)[0][1])

    rows = []
    for test, y, ctrl, alt in SPECS:
        for m in measures:
            d = hold[[y, f"z_{m}"] + ctrl].dropna()
            Y = d[y].to_numpy(float)
            X = np.column_stack([np.ones(len(d)), d[f"z_{m}"]] + [d[c].to_numpy(float) for c in ctrl])
            r = stats.ols(Y, X, ["const", "z"] + ctrl)
            se = r["se"]["z"]
            rows.append({"test": test, "measure": m, "outcome": y, "n": r["n"], "beta": r["coef"]["z"], "se_hc3": se,
                         "ci95": [r["coef"]["z"] - 1.96 * se, r["coef"]["z"] + 1.96 * se],
                         "p_wild": stats.wild_boot(Y, X, 1, B=9999, seed=3, alternative=alt),
                         "mde_80pct_power": 2.80 * se,
                         "estimation_beta": est_beta[(test, m)],
                         "same_sign_as_estimation": bool(np.sign(r["coef"]["z"]) == np.sign(est_beta[(test, m)]))})
    # In the holdout the full-sample and chronological models coincide (both use all 1996-2022 labels):
    # record which surprise series are identical so that reports do not present them as independent evidence.
    identical = [[a, b] for a, b in (("M2", "M3"), ("M2b", "M3b"))
                 if np.allclose(hold[f"S_{a}"].to_numpy(float), hold[f"S_{b}"].to_numpy(float), equal_nan=True)]
    out = {"sample": {"n_events": int(len(hold)), "start": str(hold.date.min().date()), "end": str(hold.date.max().date())},
           "identical_surprise_pairs": identical,
           "note": "identical pairs differ only by the estimation-sample SD used to standardise",
           "results": rows}
    (REPORTS / "results_holdout.json").write_text(json.dumps(out, indent=2, default=float))
    LOCK.write_text(json.dumps({"run_at": datetime.now().isoformat(timespec="seconds"),
                                "results_core_sha256": hashlib.sha256(core).hexdigest()}, indent=2))
    print(pd.DataFrame(rows)[["test", "measure", "n", "beta", "se_hc3", "p_wild", "mde_80pct_power",
                              "estimation_beta", "same_sign_as_estimation"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
