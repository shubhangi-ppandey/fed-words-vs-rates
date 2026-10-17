"""Step 12: compile every number used by the paper, README and dashboard into reports/results.json, and apply
the pre-registered decision rules (PROJECT_PLAN.md section 1) mechanically, so no verdict is written by hand.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED, REPORTS  # noqa: E402

PRIMARY = "M3"
ALPHA = 0.05


def load(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pick(rows, **kw):
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


def main() -> None:
    val = load("validation_summary.json")
    core = load("results_core.json")
    la = load("results_lookahead.json")
    ho = load("results_holdout.json")
    ph = load("results_persistence_hetero.json")
    rb = load("results_robustness.json")
    e3 = load("results_lookahead_chronobert.json")        # pre-registered extension E3
    x1 = load("exploratory_x1_path_absorption.json")      # exploratory, NOT pre-registered
    pc = load("results_plan_completion.json")             # pre-registered analyses run after the M8 review
    sr = load("results_seed_robustness.json")             # seed robustness (after review; may be absent)
    lock = load("holdout_lock.json")
    if ho:  # the holdout must have been run against exactly these estimation results
        import hashlib
        core_hash = hashlib.sha256((REPORTS / "results_core.json").read_bytes()).hexdigest()
        assert lock and lock["results_core_sha256"] == core_hash,             "holdout_lock.json does not match results_core.json: re-run 08 (with --force) and log it in deviations.md"
    ev = pd.read_csv(PROCESSED / "events_model.csv", parse_dates=["date"])
    sent = pd.read_csv(PROCESSED / "sentences.csv")

    R = core["results"]
    h1 = pick(R, test="H1", measure=PRIMARY)
    h2 = pick(R, test="H2", measure=PRIMARY)
    h5 = pick(R, test="H5", measure=PRIMARY)
    verdicts = {
        "H1": {"rule": "beta > 0 and one-sided wild-bootstrap p < 0.05", "beta_bp_per_sd": h1["beta"],
               "se": h1["se_hc3"], "p_wild": h1["p_wild"], "p_perm": h1["p_perm"],
               "supported": bool(h1["beta"] > 0 and h1["p_wild"] < ALPHA)},
        "H2": {"rule": "beta > 0 and two-sided wild-bootstrap p < 0.05", "beta_bp_per_sd": h2["beta"],
               "se": h2["se_hc3"], "p_wild": h2["p_wild"], "p_perm": h2["p_perm"],
               "incremental_r2": core["H2_incremental_r2"],
               "supported": bool(h2["beta"] > 0 and h2["p_wild"] < ALPHA)},
        "H5": {"rule": "beta < 0 and two-sided wild-bootstrap p < 0.05", "beta_pct_per_sd": h5["beta"],
               "se": h5["se_hc3"], "p_wild": h5["p_wild"], "p_perm": h5["p_perm"],
               "supported": bool(h5["beta"] < 0 and h5["p_wild"] < ALPHA)},
    }
    h3 = {}
    for arch in ("RoBERTa-large", "TF-IDF"):
        p = pick(la["pairs"], architecture=arch, outcome="hf_TFUT02")
        h3[arch] = {"diff": p["diff_in_hypothesised_direction"], "ci95": p["ci95"],
                    "beta_look_ahead": p["beta_look_ahead"], "beta_safe": p["beta_safe"],
                    "supported": bool(p["diff_in_hypothesised_direction"] > 0 and p["ci95"][0] > 0)}
    verdicts["H3"] = {"rule": "look-ahead minus safe > 0 with 95% CI excluding 0 (H1 specification)", **h3}
    if ho:
        hh = pick(ho["results"], test="H1", measure=PRIMARY)
        verdicts["H4"] = {"rule": "holdout beta has the same sign as in estimation (MDE reported)",
                          "beta": hh["beta"], "ci95": hh["ci95"], "mde_80pct_power": hh["mde_80pct_power"],
                          "p_wild": hh["p_wild"], "n": hh["n"], "consistent": hh["same_sign_as_estimation"]}
    else:
        verdicts["H4"] = {"rule": "holdout beta has the same sign as in estimation", "status": "not yet run (locked)"}
    verdicts["H6"] = {"rule": "interaction > 0 and two-sided wild-bootstrap p < 0.05", **ph["H6"],
                      "supported": bool(ph["H6"]["beta_interaction"] > 0 and ph["H6"]["p_wild"] < ALPHA)}
    verdicts["H7"] = {"rule": "Holm-significant industries reported; rate-sensitive group minus rest < 0 (they fall more "
                              "after hawkish surprises) with 95% CI excluding 0",
                      "n_holm_significant": ph["H7"]["n_holm_significant"],
                      "holm_significant": [r["industry"] for r in ph["H7"]["industries"] if r["p_holm"] < ALPHA],
                      "group_minus_rest": ph["H7"]["group_minus_rest"], "ci95": ph["H7"]["group_minus_rest_ci95"],
                      "supported": bool(ph["H7"]["group_minus_rest_ci95"][1] < 0)}

    main_ = ev
    out = {
        "generated": date.today().isoformat(),
        "primary_measure": PRIMARY,
        "data": {"n_statements_downloaded": int(sent.id.nunique()),
                 "n_sentences": int(len(sent)), "n_sentences_kept": int((sent.boilerplate == 0).sum()),
                 "main_sample": int(len(main_)), "estimation": int((main_.holdout == 0).sum()),
                 "holdout": int(main_.holdout.sum()), "zlb_meetings": int(main_.zlb.sum()),
                 "first_meeting": str(main_.date.min().date()), "last_meeting": str(main_.date.max().date())},
        "validation": val,
        "verdicts": verdicts,
        "core": core, "lookahead": la, "holdout": ho, "persistence_hetero": ph, "robustness": rb,
        "extension_E3_chronobert": e3, "exploratory_x1_path_absorption": x1,
        "plan_completion": pc, "seed_robustness": sr, "holdout_lock": lock,
    }

    def clean(o):  # strict JSON: NaN/inf -> null
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    (REPORTS / "results.json").write_text(json.dumps(clean(out), indent=2, default=float, allow_nan=False))
    for k, v in verdicts.items():
        flag = v.get("supported", v.get("consistent", v.get("status")))
        print(f"{k}: {flag}")


if __name__ == "__main__":
    main()
