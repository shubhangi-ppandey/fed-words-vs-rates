"""Step 4g (GPU; added after the M8 review, NOT pre-registered; deviations.md #16): fine-tuning-seed robustness.

The pre-registered RoBERTa-large models (04c) use a single seed (42). Fine-tuning randomness can be large
(Mosbach et al. 2021), so M2 (full sample) and M3 (chronological, cutoffs 1999-2022, i.e. the estimation sample)
are re-trained with seeds 43 and 44 using the identical code and hyperparameters. The collapse rule falls back to
seed + 100, + 200. Analysis: analysis/10b_seed_robustness.py.
Outputs: data/processed/transformer_preds_seed<S>/<job>.csv (+ metrics); resumable.
"""
import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED  # noqa: E402

base = importlib.import_module("04c_transformers")
EXTRA_SEEDS = [43, 44]
CUTOFFS = list(range(1999, 2023))  # estimation sample only


def main() -> None:
    tdw = pd.read_csv(PROCESSED / "tdw_labels.csv")
    sent = pd.read_csv(PROCESSED / "sentences.csv", dtype={"id": str})
    for s in EXTRA_SEEDS:
        base.SEEDS = [s, s + 100, s + 200]
        base.OUT = PROCESSED / f"transformer_preds_seed{s}"
        base.OUT.mkdir(exist_ok=True)
        base.run_job("full", tdw, None, sent)
        for Y in CUTOFFS:
            train_df = tdw[tdw.year < Y]
            assert train_df.year.max() < Y
            base.run_job(f"chrono_{Y}", train_df, tdw[tdw.year == Y], sent)
    print("seed robustness jobs finished")


if __name__ == "__main__":
    main()
