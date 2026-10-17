"""Step 4a: sentence table for all statements + cleaned labelled corpus + dictionary measure (M1).

Outputs
- data/processed/sentences.csv   one row per statement sentence: id, date, year, para, sent_idx, text,
                                 boilerplate flag, M1 dictionary label and hit counts
- data/processed/tdw_labels.csv  Trillion Dollar Words sentences (mojibake repaired): text, year, label, split
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dictionary  # noqa: E402
from common.paths import PROCESSED, RAW, STATEMENTS_TXT  # noqa: E402
from common.text import is_boilerplate, is_boilerplate_paragraph, repair, split_sentences  # noqa: E402


def main() -> None:
    idx = pd.read_csv(PROCESSED / "statements_index.csv", dtype={"id": str})
    rows = []
    for _, st in idx.iterrows():
        paras = (STATEMENTS_TXT / f"{st.id}.txt").read_text(encoding="utf-8").splitlines()
        k = 0
        for p_i, para in enumerate(paras):
            para_boiler = is_boilerplate_paragraph(para)
            for s in split_sentences(repair(para)):
                h, d = dictionary.score(s)
                rows.append({"id": st.id, "date": st.date, "year": int(st.date[:4]), "para": p_i, "sent_idx": k,
                             "text": s, "boilerplate": int(para_boiler or is_boilerplate(s)),
                             "m1_hits_h": h, "m1_hits_d": d,
                             "m1_label": dictionary.label(s)})
                k += 1
    sent = pd.DataFrame(rows)
    sent.to_csv(PROCESSED / "sentences.csv", index=False)

    tdw = pd.concat([pd.read_csv(RAW / "tdw" / f"{sp}.csv").assign(split=sp) for sp in ("train", "test")])
    tdw["text"] = tdw["sentence"].astype(str).map(repair)
    tdw = tdw[["text", "year", "label", "split", "orig_index"]].reset_index(drop=True)
    tdw.to_csv(PROCESSED / "tdw_labels.csv", index=False)

    keep = sent[sent.boilerplate == 0]
    print(f"statement sentences: {len(sent)} (boilerplate {int(sent.boilerplate.sum())}, kept {len(keep)})")
    print("sentences per statement (kept): median", int(keep.groupby('id').size().median()),
          "min", int(keep.groupby('id').size().min()), "max", int(keep.groupby('id').size().max()))
    print("M1 label shares (kept):", keep.m1_label.value_counts(normalize=True).round(3).to_dict())
    print(f"TDW: {len(tdw)} labelled sentences; repaired mojibake in "
          f"{int((tdw.text != pd.concat([pd.read_csv(RAW / 'tdw' / f'{sp}.csv') for sp in ('train', 'test')]).sentence.astype(str).values).sum())}")


if __name__ == "__main__":
    main()
