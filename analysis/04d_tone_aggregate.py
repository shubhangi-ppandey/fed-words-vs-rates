"""Step 4d: aggregate sentence-level tone to statement-level tone for every measure and model window,
and collect the validation of all classifiers in one file.

Statement tone (pre-registered, PROJECT_PLAN.md section 5), over non-boilerplate sentences:
- tau_count = (#hawkish - #dovish) / #sentences   (primary; labels = argmax class)
- tau_prob  = mean(p_hawkish - p_dovish)           (alternative; not defined for the dictionary)

Measures: M1 dictionary; M2b TF-IDF full sample; M3b TF-IDF chronological (one row per cutoff);
M2 RoBERTa-large full sample; M3 RoBERTa-large chronological (one row per cutoff).

Outputs
- data/processed/tone_long.csv          measure, cutoff (0 = not chronological), id, n_sent, tau_count, tau_prob
- data/processed/sentence_scores.csv    per kept sentence: labels of every non-chronological measure and of the
                                        chronological measure applied with the statement's own-year cutoff (for the app)
- reports/validation_summary.json       accuracy / macro-F1 of every measure (test split and out-of-time)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED, REPORTS  # noqa: E402

TP = PROCESSED / "transformer_preds"
CUTOFFS = list(range(1999, 2024))
CB_CUTOFFS = list(range(2000, 2024))  # ChronoBERT: first vintage is 1999-12-31
rng = np.random.default_rng(7)


def aggregate(df: pd.DataFrame, has_prob: bool) -> pd.DataFrame:
    """df has columns id, label (0 dov, 1 hawk, 2 neu) and, if has_prob, ph and pd."""
    g = df.groupby("id")
    out = pd.DataFrame({"n_sent": g.size(),
                        "tau_count": g["label"].apply(lambda s: ((s == 1).sum() - (s == 0).sum()) / len(s))})
    out["tau_prob"] = g.apply(lambda x: (x["ph"] - x["pd"]).mean()) if has_prob else np.nan
    return out.reset_index()


def with_labels(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["label"] = pred[["pd", "ph", "pn"]].to_numpy().argmax(1)  # column order = classes 0, 1, 2
    return pred


def boot_f1(y, p, n=2000):
    y, p = np.asarray(y), np.asarray(p)
    b = [f1_score(y[i], p[i], average="macro") for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def main() -> None:
    sent = pd.read_csv(PROCESSED / "sentences.csv", dtype={"id": str})
    kept = sent[sent.boilerplate == 0][["id", "sent_idx", "year", "text", "m1_label"]]
    key = ["id", "sent_idx"]
    rows = []

    # M1 dictionary
    t = aggregate(kept.rename(columns={"m1_label": "label"}), has_prob=False)
    rows.append(t.assign(measure="M1", cutoff=0))

    # TF-IDF
    full = with_labels(pd.read_csv(PROCESSED / "preds_tfidf_full.csv", dtype={"id": str}))
    rows.append(aggregate(kept[key].merge(full, on=key), True).assign(measure="M2b", cutoff=0))
    chrono = pd.read_csv(PROCESSED / "preds_tfidf_chrono.csv", dtype={"id": str})
    for Y, c in chrono.groupby("cutoff"):
        rows.append(aggregate(kept[key].merge(with_labels(c.drop(columns="cutoff")), on=key), True)
                    .assign(measure="M3b", cutoff=int(Y)))

    # RoBERTa-large
    m2 = with_labels(pd.read_csv(TP / "full.csv", dtype={"id": str}))
    rows.append(aggregate(kept[key].merge(m2, on=key), True).assign(measure="M2", cutoff=0))
    m3_by_cut = {}
    for Y in CUTOFFS:
        c = with_labels(pd.read_csv(TP / f"chrono_{Y}.csv", dtype={"id": str}))
        m3_by_cut[Y] = c
        rows.append(aggregate(kept[key].merge(c, on=key), True).assign(measure="M3", cutoff=Y))

    # ChronoBERT (extension E3), included once 04f has produced its outputs
    cb = PROCESSED / "transformer_preds_chronobert"
    m5_by_cut = {}
    if (cb / "cb_full.csv").exists():
        m5f = with_labels(pd.read_csv(cb / "cb_full.csv", dtype={"id": str}))
        rows.append(aggregate(kept[key].merge(m5f, on=key), True).assign(measure="M5f", cutoff=0))
        for Y in CB_CUTOFFS:
            c = with_labels(pd.read_csv(cb / f"cb_chrono_{Y}.csv", dtype={"id": str}))
            m5_by_cut[Y] = c
            rows.append(aggregate(kept[key].merge(c, on=key), True).assign(measure="M5", cutoff=Y))

    tone = pd.concat(rows, ignore_index=True)[["measure", "cutoff", "id", "n_sent", "tau_count", "tau_prob"]]
    tone.to_csv(PROCESSED / "tone_long.csv", index=False)

    # Sentence-level scores for the dashboard (own-year cutoff for the chronological measures)
    ss = kept.rename(columns={"m1_label": "M1"}).copy()
    ss = ss.merge(full[key + ["label"]].rename(columns={"label": "M2b"}), on=key)
    ss = ss.merge(m2[key + ["label", "ph", "pd"]].rename(columns={"label": "M2", "ph": "M2_ph", "pd": "M2_pd"}), on=key)
    own = ss.year.clip(lower=CUTOFFS[0], upper=CUTOFFS[-1])
    ss["M3"] = [int(m3_by_cut[y].set_index(key).loc[(i, k), "label"]) for i, k, y in zip(ss.id, ss.sent_idx, own)]
    ch = chrono.assign(label=chrono[["pd", "ph", "pn"]].to_numpy().argmax(1)).set_index(["cutoff"] + key)["label"]
    ss["M3b"] = [int(ch.loc[(y, i, k)]) for i, k, y in zip(ss.id, ss.sent_idx, own)]
    ss.loc[ss.year < CUTOFFS[0], ["M3", "M3b"]] = -1  # no look-ahead-safe model before 1999
    if m5_by_cut:
        own5 = ss.year.clip(lower=CB_CUTOFFS[0], upper=CB_CUTOFFS[-1])
        ss["M5"] = [int(m5_by_cut[y].set_index(key).loc[(i, k), "label"]) for i, k, y in zip(ss.id, ss.sent_idx, own5)]
        ss.loc[ss.year < CB_CUTOFFS[0], "M5"] = -1
    ss.to_csv(PROCESSED / "sentence_scores.csv", index=False)

    # Validation summary
    v = json.loads((REPORTS / "validation_tfidf_m1.json").read_text())
    val_m = json.loads((TP / "val_metrics.json").read_text())
    val_e = pd.read_csv(TP / "val_eval.csv")
    oot = []
    per_year = []
    for Y in CUTOFFS:
        f = TP / f"chrono_{Y}_eval.csv"
        if f.exists():
            e = pd.read_csv(f)
            if len(e):
                oot.append(e)
                mj = json.loads((TP / f"chrono_{Y}_metrics.json").read_text())
                per_year.append({"year": Y, "n_train": mj["n_train"], "n_test": len(e), "seed": mj["seed"],
                                 "macro_f1": round(float(f1_score(e.y, e.p, average="macro")), 4)})
    oot = pd.concat(oot)
    summary = {
        "M1_dictionary_test_split": v["M1_dictionary"]["test_split"],
        "TFIDF_test_split": v["TFIDF_test_split"],
        "M3b_tfidf_chrono_out_of_time": v["M3b_chrono_out_of_time"],
        "RoBERTa_large_test_split": {"n": int(len(val_e)), "accuracy": val_m["eval_accuracy"],
                                     "macro_f1": val_m["eval_macro_f1"], "macro_f1_ci95": boot_f1(val_e.y, val_e.p),
                                     "seed": val_m["seed"]},
        "M3_roberta_chrono_out_of_time": {"pooled": {"n": int(len(oot)),
                                                     "macro_f1": round(float(f1_score(oot.y, oot.p, average="macro")), 4),
                                                     "macro_f1_ci95": boot_f1(oot.y, oot.p)},
                                          "by_year": per_year},
    }
    if (cb / "cb_val_metrics.json").exists():
        cv_m = json.loads((cb / "cb_val_metrics.json").read_text())
        cv_e = pd.read_csv(cb / "cb_val_eval.csv")
        summary["ChronoBERT_test_split"] = {"n": int(len(cv_e)), "accuracy": cv_m["eval_accuracy"],
                                            "macro_f1": cv_m["eval_macro_f1"], "macro_f1_ci95": boot_f1(cv_e.y, cv_e.p),
                                            "seed": cv_m["seed"]}
        oot5, py5 = [], []
        for Y in CB_CUTOFFS:
            f = cb / f"cb_chrono_{Y}_eval.csv"
            if f.exists() and len(e := pd.read_csv(f)):
                oot5.append(e)
                mj = json.loads((cb / f"cb_chrono_{Y}_metrics.json").read_text())
                py5.append({"year": Y, "n_train": mj["n_train"], "n_test": len(e), "seed": mj["seed"],
                            "collapsed": mj.get("collapsed", False),
                            "macro_f1": round(float(f1_score(e.y, e.p, average="macro")), 4)})
        oot5 = pd.concat(oot5)
        summary["M5_chronobert_chrono_out_of_time"] = {
            "pooled": {"n": int(len(oot5)), "macro_f1": round(float(f1_score(oot5.y, oot5.p, average="macro")), 4),
                       "macro_f1_ci95": boot_f1(oot5.y, oot5.p)}, "by_year": py5}
    (REPORTS / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: (x.get("macro_f1") if "macro_f1" in x else x["pooled"]["macro_f1"]) for k, x in summary.items()}, indent=1))
    print(f"tone_long rows: {len(tone)}; sentence_scores rows: {len(ss)}")


if __name__ == "__main__":
    main()
