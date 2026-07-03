#!/usr/bin/env python3
"""
02_run_multi_seeds.py
=====================
Launch multiple STBS estimations SEQUENTIALLY with different random seeds.
Each run saves results to stbs_cavi_results/seed_<SEED>_K<NUM_TOPICS>/.

Runs one at a time to avoid GPU contention (parallel was ~9x slower per epoch).

Usage (from STBS_CAVI/ directory):
    cd /path/to/STBS_CAVI
    TF_USE_LEGACY_KERAS=1 PYTHONUNBUFFERED=1 nohup ./venv_gpu/bin/python3 -u \
        ../Revision_code_CAVI/02_run_multi_seeds.py >> ../Revision_code_CAVI/stbs_cavi_results/multi_seed.log 2>&1 &
"""

import os
import sys
import subprocess
import time

# ================================================================== #
# CONFIGURATION
# ================================================================== #
SEEDS = [314159, 42, 123456]
NUM_EPOCHS = 300
NUM_TOPICS = 25  # Change this for different K runs

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STBS_CAVI_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'STBS_CAVI'))
RESULTS_BASE = os.path.join(SCRIPT_DIR, 'stbs_cavi_results')
ESTIMATE_SCRIPT = os.path.join(SCRIPT_DIR, '01_estimate_STBS.py')
PYTHON = os.path.join(STBS_CAVI_DIR, 'venv_gpu', 'bin', 'python3')

# R script for plots
R_PLOTS_SCRIPT = os.path.join(SCRIPT_DIR, '05_run_R_plots.R')

# Python word cloud script
WORDCLOUD_SCRIPT = os.path.join(SCRIPT_DIR, '03_generate_wordclouds.py')

# ================================================================== #

def main():
    print("=" * 60)
    print("STBS Multi-Seed Estimation (SEQUENTIAL)")
    print("=" * 60)
    print(f"Seeds:      {SEEDS}")
    print(f"Topics:     K={NUM_TOPICS}")
    print(f"Epochs:     {NUM_EPOCHS}")
    print(f"Mode:       Sequential (one at a time)")
    print(f"Results in: {RESULTS_BASE}")
    print(f"Est. time:  ~{len(SEEDS) * 100} min ({len(SEEDS) * 100 / 60:.1f} hours)")
    print("=" * 60)

    start_time = time.time()
    succeeded = []

    for i, seed in enumerate(SEEDS):
        out_dir = os.path.join(RESULTS_BASE, f"seed_{seed}_K{NUM_TOPICS}")
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run.log")

        cmd = [
            PYTHON, "-u", ESTIMATE_SCRIPT,
            "--seed", str(seed),
            "--num-epochs", str(NUM_EPOCHS),
            "--num-topics", str(NUM_TOPICS),
            "--output-dir", out_dir,
        ]

        env = os.environ.copy()
        env["TF_USE_LEGACY_KERAS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(SEEDS)}] Running seed={seed}, K={NUM_TOPICS}")
        print(f"  Output: {out_dir}")
        print(f"  Log:    {log_path}")
        print(f"{'=' * 60}")

        run_start = time.time()

        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=STBS_CAVI_DIR,
            )

        run_time = time.time() - run_start
        total_elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"  DONE in {run_time/60:.1f} min (total elapsed: {total_elapsed/60:.1f} min)")

            # Run R plots immediately after each successful run
            print(f"  Generating R plots...")
            r_result = subprocess.run(
                ["Rscript", R_PLOTS_SCRIPT, out_dir],
                capture_output=True, text=True, cwd=STBS_CAVI_DIR,
            )
            if r_result.returncode == 0:
                print(f"  R plots OK")
            else:
                print(f"  R plots FAILED: {r_result.stderr[:300]}")

            # Generate word clouds
            print(f"  Generating word clouds...")
            wc_result = subprocess.run(
                [PYTHON, WORDCLOUD_SCRIPT, "--results-dir", out_dir],
                capture_output=True, text=True, cwd=STBS_CAVI_DIR,
            )
            if wc_result.returncode == 0:
                print(f"  Word clouds OK")
            else:
                print(f"  Word clouds FAILED: {wc_result.stderr[:300]}")

            succeeded.append(seed)
        else:
            print(f"  FAILED (code {result.returncode}) after {run_time/60:.1f} min")
            print(f"  Check log: {log_path}")

        remaining = len(SEEDS) - (i + 1)
        if remaining > 0:
            print(f"\n  Cooling down 5 min before next run...")
            time.sleep(300)

        if remaining > 0 and len(succeeded) > 0:
            avg_time = total_elapsed / (i + 1)
            est_remaining = avg_time * remaining
            print(f"\n  Remaining: {remaining} runs, est. {est_remaining/60:.0f} min")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"ALL DONE in {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"{'=' * 60}")
    print(f"Succeeded: {len(succeeded)}/{len(SEEDS)}")
    for seed in succeeded:
        print(f"  {RESULTS_BASE}/seed_{seed}_K{NUM_TOPICS}/")
    failed = [s for s in SEEDS if s not in succeeded]
    if failed:
        print(f"Failed: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
