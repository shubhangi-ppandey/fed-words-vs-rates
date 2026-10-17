"""EXPLORATORY (not pre-registered; added after seeing robustness check 8): why does the tone effect on the
2-year yield vanish when the Jarocinski-Karadi MP and CBI shocks are controlled for?

Idea: MP + CBI = pc1, the first principal component of 1-month..1-year rate-futures surprises, i.e. the market's
expected-PATH surprise. If tone moves yields THROUGH the expected path (H2), conditioning on pc1 absorbs the channel.
Checks (estimation sample, primary measure M3, HC3 + wild bootstrap):
  a) tone on pc1 itself (does tone predict the path-type surprise?)
  b) each Treasury maturity on tone + pc1 (is there any effect beyond the path, e.g. on long yields?)
  c) each Treasury maturity on tone + GSS target + GSS path
Output: reports/exploratory_x1_path_absorption.json
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import stats  # noqa: E402
from common.paths import REPORTS  # noqa: E402

es = importlib.import_module("06_event_study")


def fit(d, y, regs):
    d = d[[y] + regs].dropna()
    Y = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in regs])
    r = stats.ols(Y, X, ["const"] + regs)
    return {"y": y, "controls": regs[1:], "n": r["n"], "beta": r["coef"][regs[0]], "se": r["se"][regs[0]],
            "p_wild": stats.wild_boot(Y, X, 1, B=4999, seed=3), "r2": r["r2"]}


est = es.load_estimation()
z = f"z_{es.PRIMARY}"
out = {"a_tone_predicts_pc1": fit(est, "jk_pc1", [z, "hf_MP1"]),
       "b_beyond_pc1": [fit(est, y, [z, "jk_pc1"]) for y in ["hf_TFUT02", "hf_TFUT05", "hf_TFUT10", "hf_TFUT30"]],
       "c_beyond_gss_both": [fit(est, y, [z, "gss_target", "gss_path"]) for y in ["hf_TFUT02", "hf_TFUT05", "hf_TFUT10", "hf_TFUT30"]],
       "corr_pc1_gss_path": float(np.corrcoef(est.jk_pc1, est.gss_path)[0, 1])}
(REPORTS / "exploratory_x1_path_absorption.json").write_text(json.dumps(out, indent=2, default=float))
print("a) tone -> pc1 (| MP1):", {k: round(v, 4) if isinstance(v, float) else v for k, v in out["a_tone_predicts_pc1"].items()})
print("corr(pc1, GSS path) =", round(out["corr_pc1_gss_path"], 3))
for k in ("b_beyond_pc1", "c_beyond_gss_both"):
    print(k)
    for r in out[k]:
        print(f"   {r['y']:10} beta {r['beta']:+.3f} se {r['se']:.3f} p {r['p_wild']:.4f} r2 {r['r2']:.3f}")
