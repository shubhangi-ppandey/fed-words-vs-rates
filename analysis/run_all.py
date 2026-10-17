"""Run the whole pipeline in order. Usage: py -3.14 analysis/run_all.py [--from STEP]

Steps 04c, 04f and 04g need an NVIDIA GPU (several hours in total on an RTX 5070 Ti); they skip jobs whose outputs
exist. 04e downloads about 15 GB of ChronoBERT checkpoints into models_cache/.
Step 08 (holdout) refuses to re-run once reports/holdout_lock.json exists; step 12 checks that the lock matches
the current estimation results and stops otherwise (re-run 08 with --force and log it in reports/deviations.md).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_fetch_statements", "02_fetch_market", "02b_raw_manifest", "03_build_events", "04a_sentences",
         "04b_tfidf", "04c_transformers", "04e_download_chronobert", "04f_chronobert", "04g_seed_robustness",
         "04d_tone_aggregate", "05_surprises", "06_event_study", "07_lookahead", "07b_lookahead_chronobert",
         "09_persistence_hetero", "10_robustness", "10b_seed_robustness", "10c_plan_completion",
         "exploratory/x1_path_absorption", "exploratory/x2_label_overlap", "08_holdout", "12_compile_results",
         "11_figures", "13_write_reports"]

start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else STEPS[0]
for s in STEPS[STEPS.index(start):]:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, "-B", str(HERE / f"{s}.py")], check=True)
