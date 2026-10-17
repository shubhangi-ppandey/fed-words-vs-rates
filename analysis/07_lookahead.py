"""Step 7 (H3): does look-ahead in the tone classifier inflate the measured effect of words on markets?

Within each architecture the two training windows are compared on the SAME estimation-sample events
(H1 specification: 30-min 2-year yield on the standardised tone surprise + MP1):
- RoBERTa-large: M2 (full sample, look-ahead) minus M3 (chronological, look-ahead-safe)
- TF-IDF:        M2b (full sample)           minus M3b (chronological)
Paired event bootstrap (9,999 resamples) of the coefficient difference; H3 is supported if the difference is
> 0 and its 95% CI excludes 0. The dictionary M1 (no learning at all) is reported as a reference point.
Also reported: how similar the look-ahead and safe surprises are (correlation, sign agreement).

Output: reports/results_lookahead.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stats  # noqa: E402
from common.paths import REPORTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib  # noqa: E402

es = importlib.import_module("06_event_study")
PAIRS = [("RoBERTa-large", "M2", "M3"), ("TF-IDF", "M2b", "M3b")]
OUTCOMES = [("hf_TFUT02", ["hf_MP1"]), ("gss_path", ["gss_target"]), ("hf_SP500", ["jk_pc1"])]


def main() -> None:
    est = es.load_estimation()
    out = {"pairs": [], "measure_similarity": {}}
    for arch, look, safe in PAIRS:
        a, b = f"S_{look}", f"S_{safe}"
        d0 = est[[a, b]].dropna()
        out["measure_similarity"][arch] = {
            "pearson": float(np.corrcoef(d0[a], d0[b])[0, 1]),
            "sign_agreement": float((np.sign(d0[a]) == np.sign(d0[b])).mean()),
        }
        for y, ctrl in OUTCOMES:
            d = est[[y, f"z_{look}", f"z_{safe}"] + ctrl].dropna()
            Y = d[y].to_numpy(float)
            C = [d[c].to_numpy(float) for c in ctrl]
            XA = np.column_stack([np.ones(len(d)), d[f"z_{look}"]] + C)
            XB = np.column_stack([np.ones(len(d)), d[f"z_{safe}"]] + C)
            pb = stats.paired_boot(Y, XA, XB, 1, B=9999, seed=5)
            # compare magnitudes in the hypothesised direction (H1/H2 positive, H5 negative)
            sign = -1.0 if y == "hf_SP500" else 1.0
            out["pairs"].append({"architecture": arch, "outcome": y, "look_ahead": look, "safe": safe, "n": int(len(d)),
                                 "beta_look_ahead": float(np.linalg.lstsq(XA, Y, rcond=None)[0][1]),
                                 "beta_safe": float(np.linalg.lstsq(XB, Y, rcond=None)[0][1]),
                                 "diff_in_hypothesised_direction": sign * pb["diff"],
                                 "ci95": sorted([sign * pb["ci95"][0], sign * pb["ci95"][1]]),
                                 "p_two_sided": pb["p_two_sided"]})
    # The plan's ORIGINAL H3 contrast (M2 - M3b, before deviation #6 redefined H3 within architecture) and paired
    # measurement-quality contrasts (added after the M8 review; not pre-registered). H1 specification throughout.
    def contrast(a, b, label):
        d = est[["hf_TFUT02", f"z_{a}", f"z_{b}", "hf_MP1"]].dropna()
        Y = d.hf_TFUT02.to_numpy(float)
        XA = np.column_stack([np.ones(len(d)), d[f"z_{a}"], d.hf_MP1])
        XB = np.column_stack([np.ones(len(d)), d[f"z_{b}"], d.hf_MP1])
        return {"contrast": label, "a": a, "b": b, **stats.paired_boot(Y, XA, XB, 1, B=9999, seed=5)}

    out["original_plan_H3"] = contrast("M2", "M3b", "plan H3: RoBERTa full sample (M2) - TF-IDF chronological (M3b)")
    out["measurement_contrasts"] = [contrast(a, b, f"{a} - {b}") for a, b in
                                    [("M3", "M3b"), ("M3", "M1"), ("M3", "M2b"), ("M2", "M1"), ("M2", "M2b")]]
    (REPORTS / "results_lookahead.json").write_text(json.dumps(out, indent=2))
    print("original plan H3 (M2 - M3b):", round(out["original_plan_H3"]["diff"], 3),
          [round(x, 3) for x in out["original_plan_H3"]["ci95"]])
    for c in out["measurement_contrasts"]:
        print(f"  {c['contrast']:10} {c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
    for p in out["pairs"]:
        print(f"{p['architecture']:13} {p['outcome']:10} look {p['beta_look_ahead']:+.3f}  safe {p['beta_safe']:+.3f}  "
              f"diff {p['diff_in_hypothesised_direction']:+.3f}  CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  p {p['p_two_sided']:.3f}")
    print("similarity:", json.dumps(out["measure_similarity"], indent=1))


if __name__ == "__main__":
    main()
