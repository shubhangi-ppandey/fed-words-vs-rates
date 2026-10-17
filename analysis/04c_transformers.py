"""Step 4c (GPU): RoBERTa-large hawkish/dovish classifiers with two training windows.

Jobs (each fine-tunes roberta-large from the same public checkpoint with identical, a-priori hyperparameters):
- val          train on the TDW train split, evaluate on the TDW test split (comparability with Shah et al. 2023)
- full   (M2)  train on ALL 2,480 labelled sentences 1996-2022  -> maximal look-ahead (as FOMC-RoBERTa)
- chrono_Y (M3) train on labelled sentences with year < Y, Y = 1999..2023 -> look-ahead-safe labels
               (the RoBERTa pretraining corpus itself dates to 2019; this residual exposure is documented)

Every job scores ALL statement sentences, so later steps can choose which model applies to which year.
Outputs go to data/processed/transformer_preds/<job>.csv (+ <job>_metrics.json); finished jobs are skipped,
so the script can be re-run after an interruption.

Hyperparameters (fixed, not tuned): lr 1e-5, 5 epochs but at least 600 optimizer steps (small label sets get more
epochs; deviations.md #11), batch 16, max_len 128, warmup 10%, weight decay 0.01, grad-clip 1.0, bf16 autocast,
seed 42. Collapse rule (pre-specified): if training-set macro-F1 < 0.40 the job is re-run with seed 43, then 44;
the seed used is recorded, and a job whose three seeds all collapse is kept but flagged "collapsed": true.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED  # noqa: E402

MODEL_NAME = "roberta-large"
LR, EPOCHS, BATCH, MAX_LEN, WARMUP, WD, CLIP = 1e-5, 5, 16, 128, 0.10, 0.01, 1.0
MIN_STEPS = 600  # small chronological label sets need more updates to leave the majority-class solution
SEEDS = [42, 43, 44]
COLLAPSE_F1 = 0.40
CUTOFFS = list(range(1999, 2024))
OUT = PROCESSED / "transformer_preds"
OUT.mkdir(exist_ok=True)
DEV = "cuda"


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def batches(texts, labels, tok, shuffle, seed):
    idx = list(range(len(texts)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), BATCH):
        j = idx[i:i + BATCH]
        enc = tok([texts[k] for k in j], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        yield {k: v.to(DEV) for k, v in enc.items()}, (torch.tensor([labels[k] for k in j], device=DEV) if labels is not None else None)


@torch.no_grad()
def predict(model, tok, texts) -> np.ndarray:
    model.eval()
    probs = []
    for enc, _ in batches(texts, None, tok, False, 0):
        with torch.autocast(DEV, dtype=torch.bfloat16):
            logits = model(**enc).logits
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(probs)


def train(texts, labels, seed):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEV)
    decay = [p for n, p in model.named_parameters() if not any(x in n for x in ("bias", "LayerNorm", "layer_norm"))]
    no_decay = [p for n, p in model.named_parameters() if any(x in n for x in ("bias", "LayerNorm", "layer_norm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}], lr=LR)
    per_epoch = math.ceil(len(texts) / BATCH)
    epochs = max(EPOCHS, math.ceil(MIN_STEPS / per_epoch))
    steps = epochs * per_epoch
    warm = int(WARMUP * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    model.train()
    for ep in range(epochs):
        for enc, y in batches(texts, labels, tok, True, seed * 1000 + ep):
            with torch.autocast(DEV, dtype=torch.bfloat16):
                loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    return model, tok, steps


def run_job(name, train_df, eval_df, sent):
    if (OUT / f"{name}.csv").exists():
        print(f"[skip] {name}")
        return
    t0 = time.time()
    collapsed = True
    for seed in SEEDS:
        model, tok, steps = train(train_df.text.tolist(), train_df.label.tolist(), seed)
        tr_pred = predict(model, tok, train_df.text.tolist()).argmax(1)
        tr_f1 = f1_score(train_df.label, tr_pred, average="macro")
        if tr_f1 >= COLLAPSE_F1:
            collapsed = False
            break
        print(f"  {name}: seed {seed} collapsed (train macro-F1 {tr_f1:.3f})", flush=True)
        if seed != SEEDS[-1]:
            del model; torch.cuda.empty_cache()
    metrics = {"job": name, "seed": seed, "n_train": int(len(train_df)), "steps": steps, "collapsed": collapsed,
               "train_years": [int(train_df.year.min()), int(train_df.year.max())],
               "train_macro_f1": round(float(tr_f1), 4)}
    if eval_df is not None and len(eval_df):
        p = predict(model, tok, eval_df.text.tolist()).argmax(1)
        metrics.update({"eval_n": int(len(eval_df)), "eval_accuracy": round(float(accuracy_score(eval_df.label, p)), 4),
                        "eval_macro_f1": round(float(f1_score(eval_df.label, p, average="macro")), 4)})
        pd.DataFrame({"y": eval_df.label.values, "p": p}).to_csv(OUT / f"{name}_eval.csv", index=False)
    probs = predict(model, tok, sent.text.tolist())
    df = sent[["id", "sent_idx"]].copy()
    df[["pd", "ph", "pn"]] = probs
    df.to_csv(OUT / f"{name}.csv", index=False)
    metrics["minutes"] = round((time.time() - t0) / 60, 2)
    (OUT / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] {json.dumps(metrics)}", flush=True)
    del model; torch.cuda.empty_cache()


def main() -> None:
    assert torch.cuda.is_available(), "GPU required"
    tdw = pd.read_csv(PROCESSED / "tdw_labels.csv")
    sent = pd.read_csv(PROCESSED / "sentences.csv", dtype={"id": str})
    run_job("val", tdw[tdw.split == "train"], tdw[tdw.split == "test"], sent)
    run_job("full", tdw, None, sent)
    for Y in CUTOFFS:
        train_df = tdw[tdw.year < Y]
        assert train_df.year.max() < Y  # look-ahead guard
        run_job(f"chrono_{Y}", train_df, tdw[tdw.year == Y], sent)
    print("all jobs finished")


if __name__ == "__main__":
    main()
