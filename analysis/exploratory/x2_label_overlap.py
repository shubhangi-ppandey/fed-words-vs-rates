"""EXPLORATORY (added after the M8 review): verbatim overlap between scored statement sentences and the labelled corpus.

A concrete look-ahead channel: if a statement sentence appears word-for-word in the labelled corpus, a model trained
on that label has "seen the answer". For each kept statement sentence we check exact matches (after whitespace and
case normalisation) and record the year of the matching label relative to the statement's year.
Output: reports/exploratory_x2_label_overlap.json
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROCESSED, REPORTS  # noqa: E402

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()  # noqa: E731
sent = pd.read_csv(PROCESSED / "sentences.csv", dtype={"id": str})
kept = sent[sent.boilerplate == 0].copy()
tdw = pd.read_csv(PROCESSED / "tdw_labels.csv")
years = tdw.assign(k=tdw.text.map(norm)).groupby("k").year.apply(list).to_dict()
kept["k"] = kept.text.map(norm)
kept["match_years"] = kept.k.map(lambda k: years.get(k, []))
m = kept[kept.match_years.map(len) > 0]
same_or_later = m[[any(y >= sy for y in ys) for ys, sy in zip(m.match_years, m.year)]]
ev = pd.read_csv(PROCESSED / "events_model.csv", dtype={"id": str})
est_ids = set(ev[ev.holdout == 0].id)
out = {"kept_sentences": int(len(kept)), "verbatim_in_labels": int(len(m)),
       "share": float(len(m) / len(kept)),
       "label_year_same_or_later_than_statement": int(len(same_or_later)),
       "label_year_earlier_only": int(len(m) - len(same_or_later)),
       "in_estimation_sample_statements": int(m.id.isin(est_ids).sum()),
       "same_or_later_in_estimation_sample": int(same_or_later.id.isin(est_ids).sum())}
(REPORTS / "exploratory_x2_label_overlap.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
