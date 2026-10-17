"""Step 4f (GPU, pre-registered extension E3; deviations.md #12): ChronoBERT tone classifiers.

ChronoBERT (He, Lv, Manela & Wu 2026) = ModernBERT-base pretrained ONLY on text dated up to each vintage date.
- cb_val           vintage 2024-12-31, TDW train split -> test split (validation, comparable to 04c 'val')
- cb_full   (M5f)  vintage 2024-12-31, ALL labels 1996-2022 (look-ahead in pretraining AND labels)
- cb_chrono_Y (M5) vintage (Y-1)-12-31, labels with year < Y, for Y = 2000..2023 (no look-ahead from either;
                   Y = 2023 also scores the 2023-2026 holdout, exactly as M3's chrono_2023 does)
Identical hyperparameters, minimum-steps rule and collapse rule as 04c (the training code is reused, not copied).
Outputs: data/processed/transformer_preds_chronobert/<job>.csv (+ _metrics.json, _eval.csv); resumable.
"""
import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED  # noqa: E402

base = importlib.import_module("04c_transformers")
dl = importlib.import_module("04e_download_chronobert")
base.OUT = PROCESSED / "transformer_preds_chronobert"
base.OUT.mkdir(exist_ok=True)
CUTOFFS = list(range(2000, 2024))


def use_vintage(v: str) -> None:
    base.MODEL_NAME = str(dl.local_path(v))


def main() -> None:
    tdw = pd.read_csv(PROCESSED / "tdw_labels.csv")
    sent = pd.read_csv(PROCESSED / "sentences.csv", dtype={"id": str})
    use_vintage("20241231")
    base.run_job("cb_val", tdw[tdw.split == "train"], tdw[tdw.split == "test"], sent)
    base.run_job("cb_full", tdw, None, sent)
    for Y in CUTOFFS:
        use_vintage(f"{Y - 1}1231")
        train_df = tdw[tdw.year < Y]
        assert train_df.year.max() < Y and int(Path(base.MODEL_NAME).name[:4]) < Y  # labels AND pretraining before Y
        base.run_job(f"cb_chrono_{Y}", train_df, tdw[tdw.year == Y], sent)
    print("all ChronoBERT jobs finished")


if __name__ == "__main__":
    main()
