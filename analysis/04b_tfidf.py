"""Step 4b: TF-IDF + logistic-regression tone classifiers, and validation of the dictionary (M1).

Two training windows (the look-ahead design, PROJECT_PLAN.md section 4 and deviations.md #6):
- M2b "full":   trained on ALL labelled sentences 1996-2022 (look-ahead: labels from the sample years)
- M3b "chrono": for statements of year Y, trained only on labelled sentences with year < Y (look-ahead-safe);
                statements from 2023 on use all labels (all of them pre-date 2023)

Hyperparameters are fixed a priori (no tuning): word 1-2 grams, min_df=2, sublinear tf, logistic C=1.0.
Classes follow the TDW coding: 0 = dovish, 1 = hawkish, 2 = neutral.

Every chronological model scores ALL statement sentences, so that a tone surprise at a meeting in year Y can be
computed with one model (trained on labels < Y) applied to both the current and the previous statement.

Outputs
- data/processed/preds_tfidf_full.csv     statement-sentence probabilities, M2b (full-sample model)
- data/processed/preds_tfidf_chrono.csv   long format: cutoff Y, id, sent_idx, probabilities from the year<Y model
- reports/validation_tfidf_m1.json        test-split and chronological out-of-time accuracy / macro-F1 (with CIs)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dictionary  # noqa: E402
from common.paths import PROCESSED, REPORTS  # noqa: E402

FIRST_CUTOFF, LAST_CUTOFF = 1999, 2023  # chronological models: train on year < Y
MIN_TRAIN = 200
N_BOOT = 2000
rng = np.random.default_rng(42)


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        LogisticRegression(C=1.0, max_iter=5000),
    )


def scores(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        boots.append(f1_score(y[i], p[i], average="macro"))
    return {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
            "macro_f1_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "f1_by_class": dict(zip(["dovish", "hawkish", "neutral"],
                                    np.round(f1_score(y, p, average=None, labels=[0, 1, 2]), 4).tolist()))}


def main() -> None:
    tdw = pd.read_csv(PROCESSED / "tdw_labels.csv")
    sent = pd.read_csv(PROCESSED / "sentences.csv", dtype={"id": str})
    out = {}

    # --- M1 dictionary on labelled data (no training involved) -----------------------------------
    m1_all = tdw.text.map(dictionary.label)
    out["M1_dictionary"] = {"test_split": scores(tdw.label[tdw.split == "test"], m1_all[tdw.split == "test"]),
                            "all_labels": scores(tdw.label, m1_all)}

    # --- Standard split validation (train -> test) -------------------------------------------------
    tr, te = tdw[tdw.split == "train"], tdw[tdw.split == "test"]
    model = make_model().fit(tr.text, tr.label)
    out["TFIDF_test_split"] = scores(te.label, model.predict(te.text))

    # --- M2b: full-sample model -------------------------------------------------------------------
    full = make_model().fit(tdw.text, tdw.label)
    preds = sent[["id", "sent_idx"]].copy()
    preds[["pd", "ph", "pn"]] = full.predict_proba(sent.text)
    preds.to_csv(PROCESSED / "preds_tfidf_full.csv", index=False)

    # --- M3b: chronological models + out-of-time validation ---------------------------------------
    chrono = []
    oot_y, oot_p, per_year = [], [], []
    for Y in range(FIRST_CUTOFF, LAST_CUTOFF + 1):
        train = tdw[tdw.year < Y]
        assert train.year.max() < Y  # look-ahead guard
        if len(train) < MIN_TRAIN:
            continue
        m = make_model().fit(train.text, train.label)
        c = sent[["id", "sent_idx"]].copy()
        c[["pd", "ph", "pn"]] = m.predict_proba(sent.text)
        c.insert(0, "cutoff", Y)
        chrono.append(c)
        test = tdw[tdw.year == Y]
        if len(test):
            p = m.predict(test.text)
            oot_y.extend(test.label.tolist()); oot_p.extend(p.tolist())
            per_year.append({"year": Y, "n_train": int(len(train)), "n_test": int(len(test)),
                             "macro_f1": round(float(f1_score(test.label, p, average="macro")), 4)})
    out["M3b_chrono_out_of_time"] = {"pooled": scores(oot_y, oot_p), "by_year": per_year}

    pd.concat(chrono).to_csv(PROCESSED / "preds_tfidf_chrono.csv", index=False)
    (REPORTS / "validation_tfidf_m1.json").write_text(json.dumps(out, indent=2))

    print("M1 dictionary  test split:", out["M1_dictionary"]["test_split"]["macro_f1"],
          out["M1_dictionary"]["test_split"]["macro_f1_ci95"], "| all labels:", out["M1_dictionary"]["all_labels"]["macro_f1"])
    print("TF-IDF         test split:", out["TFIDF_test_split"]["macro_f1"], out["TFIDF_test_split"]["macro_f1_ci95"])
    print("M3b chrono     out-of-time pooled:", out["M3b_chrono_out_of_time"]["pooled"]["macro_f1"],
          out["M3b_chrono_out_of_time"]["pooled"]["macro_f1_ci95"])
    print(pd.DataFrame(per_year).to_string(index=False))
    print(f"chronological cutoffs saved: {[c.cutoff.iloc[0] for c in chrono][0]}..{[c.cutoff.iloc[0] for c in chrono][-1]}")


if __name__ == "__main__":
    main()
