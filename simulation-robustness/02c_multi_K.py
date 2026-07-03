#!/usr/bin/env python3
"""
02b_multi_K.py
==============
Launch STBS estimations with different numbers of topics (K).
All runs use the same seed (123456) to isolate the effect of K.

Results saved to stbs_cavi_results/seed_123456_K{K}/.

Usage (from STBS_CAVI/ directory):
    cd /path/to/STBS_CAVI
    TF_USE_LEGACY_KERAS=1 PYTHONUNBUFFERED=1 nohup ./venv_gpu/bin/python3 -u \
        ../Revision_code_CAVI/02b_multi_K.py >> ../Revision_code_CAVI/stbs_cavi_results/multi_K.log 2>&1 &
"""

import os
import subprocess
import time

# ================================================================== #
# CONFIGURATION
# ================================================================== #
SEED = 123456
NUM_EPOCHS = 300
K_VALUES = [15, 20, 30]

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STBS_CAVI_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'STBS_CAVI'))
RESULTS_BASE = os.path.join(SCRIPT_DIR, 'stbs_cavi_results')
ESTIMATE_SCRIPT = os.path.join(SCRIPT_DIR, '01_estimate_STBS.py')
PYTHON = os.path.join(STBS_CAVI_DIR, 'venv_gpu', 'bin', 'python3')

# Post-processing scripts
R_PLOTS_SCRIPT = os.path.join(SCRIPT_DIR, '05_run_R_plots.R')
WORDCLOUD_SCRIPT = os.path.join(SCRIPT_DIR, '03_generate_wordclouds.py')

# ================================================================== #


def main():
    n_runs = len(K_VALUES)
    print("=" * 60)
    print("STBS Multi-K Estimation (SEQUENTIAL)")
    print("=" * 60)
    print(f"Seed:       {SEED}")
    print(f"K values:   {K_VALUES}")
    print(f"Epochs:     {NUM_EPOCHS}")
    print(f"Mode:       Sequential (one at a time)")
    print(f"Results in: {RESULTS_BASE}")
    print(f"Est. time:  ~{n_runs * 60} min ({n_runs * 60 / 60:.1f} hours)")
    print("=" * 60)

    start_time = time.time()
    succeeded = []

    for i, K in enumerate(K_VALUES):
        out_dir = os.path.join(RESULTS_BASE, f"seed_{SEED}_K{K}")
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run.log")

        cmd = [
            PYTHON, "-u", ESTIMATE_SCRIPT,
            "--seed", str(SEED),
            "--num-epochs", str(NUM_EPOCHS),
            "--num-topics", str(K),
            "--output-dir", out_dir,
        ]

        env = os.environ.copy()
        env["TF_USE_LEGACY_KERAS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{n_runs}] Running K={K}, seed={SEED}")
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

            # R plots
            print(f"  Generating R plots...")
            r_result = subprocess.run(
                ["Rscript", R_PLOTS_SCRIPT, out_dir],
                capture_output=True, text=True, cwd=STBS_CAVI_DIR,
            )
            if r_result.returncode == 0:
                print(f"  R plots OK")
            else:
                print(f"  R plots FAILED: {r_result.stderr[:300]}")

            # Word clouds
            print(f"  Generating word clouds...")
            wc_result = subprocess.run(
                [PYTHON, WORDCLOUD_SCRIPT, "--results-dir", out_dir],
                capture_output=True, text=True, cwd=STBS_CAVI_DIR,
            )
            if wc_result.returncode == 0:
                print(f"  Word clouds OK")
            else:
                print(f"  Word clouds FAILED: {wc_result.stderr[:300]}")

            succeeded.append(K)
        else:
            print(f"  FAILED (code {result.returncode}) after {run_time/60:.1f} min")
            print(f"  Check log: {log_path}")

        remaining = n_runs - (i + 1)
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
    print(f"Succeeded: {len(succeeded)}/{n_runs}")
    for K in succeeded:
        print(f"  {RESULTS_BASE}/seed_{SEED}_K{K}/")
    failed = [K for K in K_VALUES if K not in succeeded]
    if failed:
        print(f"Failed K values: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
