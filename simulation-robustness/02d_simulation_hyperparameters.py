#!/usr/bin/env python3
"""
02c_simulation_hyperparameters.py
=================================
Hyperparameter sensitivity analysis for the STBS-CAVI model.

Re-estimates the model with different hyperparameter settings and compares
results to the baseline (all shape/rate = 0.3).

Hyperparameters are varied ONE GROUP AT A TIME while keeping others at baseline.

------------------------------------------------------------------------
Hyperparameter groups and their roles (from the paper Section 3.1.1):
------------------------------------------------------------------------

GROUP 1: omega (regression / iota shrinkage)
  Controls iota_prec and iota_prec_rate.
  - SMALL (0.1): Stronger shrinkage of regression coefficients.
  - BASELINE (0.3): Moderate sparsity-inducing prior.
  - LARGE (1.0): Weaker shrinkage, more diffuse.

GROUP 2: beta (topic-word sparsity)
  Controls beta shape/rate and beta_rate shape/rate.
  - SMALL (0.1): Very sparse topics (peaked, distinctive).
  - BASELINE (0.3): Moderately sparse.
  - LARGE (1.0): Diffuse topics (more overlap).

GROUP 3: rho (polarity / eta shrinkage)
  Controls eta_prec and eta_prec_rate.
  - SMALL (0.1): Allows strong ideological word differentiation.
  - BASELINE (0.3): Moderate polarity shrinkage.
  - LARGE (1.0): More restrained polarity loadings.

GROUP 4: theta (document-topic sparsity)
  Controls theta shape/rate and theta_rate shape/rate.
  - SMALL (0.1): Each document dominated by 1-2 topics.
  - BASELINE (0.3): Moderate sparsity.
  - LARGE (1.0): Documents use many topics evenly.

GROUP 5: ideal (ideal point precision)
  Controls ideal_prec shape/rate.
  - SMALL (0.1): Most speakers have low precision (wide IP spread).
  - BASELINE (0.3): Moderate spread.
  - LARGE (1.0): More uniform precision across speakers.

------------------------------------------------------------------------
Test values: 0.1, 0.3 (baseline), 1.0
Total: 1 baseline + 5 groups x 2 variants = 11 estimations
------------------------------------------------------------------------

Usage:
    cd /path/to/STBS_CAVI

    # Step 1: baseline (already done as seed_123456_K25)
    # Step 2: run groups sequentially
    TF_USE_LEGACY_KERAS=1 PYTHONUNBUFFERED=1 nohup ./venv_gpu/bin/python3 -u \
        ../Revision_code_CAVI/02c_simulation_hyperparameters.py \
        >> ../Revision_code_CAVI/stbs_cavi_results/hyperparam_sim.log 2>&1 &

    # Or run a single group:
    python3 ../Revision_code_CAVI/02c_simulation_hyperparameters.py omega
    python3 ../Revision_code_CAVI/02c_simulation_hyperparameters.py analyze
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd

# ================================================================== #
# CONFIGURATION
# ================================================================== #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STBS_CAVI_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'STBS_CAVI'))
RESULTS_BASE = os.path.join(SCRIPT_DIR, 'stbs_cavi_results')
ESTIMATE_SCRIPT = os.path.join(SCRIPT_DIR, '01_estimate_STBS.py')
PYTHON = os.path.join(STBS_CAVI_DIR, 'venv_gpu', 'bin', 'python3')
CLEAN_DIR = os.path.join(STBS_CAVI_DIR, 'data', 'hein-daily', 'clean')

SEED = 123456
NUM_EPOCHS = 300
NUM_TOPICS = 25

HP_VALUES = [0.1, 0.3, 1.0]
BASELINE_DIR = os.path.join(RESULTS_BASE, f"seed_{SEED}_K{NUM_TOPICS}")

# The kappa scaling factors used in the paper (Section 3.1.1)
ETA_KAPPA = 10.0
IOTA_KAPPA = 10.0

# ================================================================== #
# Hyperparameter groups: name -> (description, overrides_fn)
# Each overrides_fn(value) returns a dict for --hp-overrides JSON
# ================================================================== #

def _omega_overrides(v):
    """Iota shrinkage: modify iota_prec and iota_prec_rate."""
    # In the paper: iota_prec ~ Gamma(a, a*2/kappa), iota_prec_rate ~ Gamma(a, a/a * kappa/2)
    return {
        "iota_prec": {"shape": v, "rate": v * 2.0 / IOTA_KAPPA},
        "iota_prec_rate": {"shape": v, "rate": v / v * IOTA_KAPPA / 2.0},
    }

def _beta_overrides(v):
    """Topic-word sparsity: modify beta and beta_rate."""
    return {
        "beta": {"shape": v, "rate": v},
        "beta_rate": {"shape": v, "rate": v / v},  # = 1.0
    }

def _rho_overrides(v):
    """Polarity/eta shrinkage: modify eta_prec and eta_prec_rate."""
    return {
        "eta_prec": {"shape": v, "rate": v * 2.0 / ETA_KAPPA},
        "eta_prec_rate": {"shape": v, "rate": v / v * ETA_KAPPA / 2.0},
    }

def _theta_overrides(v):
    """Document-topic sparsity: modify theta and theta_rate."""
    return {
        "theta": {"shape": v, "rate": v},
        "theta_rate": {"shape": v, "rate": v / v},  # = 1.0
    }

def _ideal_overrides(v):
    """Ideal point precision: modify ideal_prec."""
    return {
        "ideal_prec": {"shape": v, "rate": v},
    }


HP_GROUPS = {
    "omega": ("Regression coefficient shrinkage (iota)", _omega_overrides),
    "beta":  ("Topic-word sparsity", _beta_overrides),
    "rho":   ("Polarity loading shrinkage (eta)", _rho_overrides),
    "theta": ("Document-topic sparsity", _theta_overrides),
    "ideal": ("Ideal point precision", _ideal_overrides),
}


# ================================================================== #
# Run a single estimation variant
# ================================================================== #

def run_variant(group, value, description):
    """Run one estimation with specific hyperparameter overrides."""
    label = f"{group}_{value}"
    out_dir = os.path.join(RESULTS_BASE, f"hp_{label}_K{NUM_TOPICS}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    _, overrides_fn = HP_GROUPS[group]
    hp_overrides = overrides_fn(value)
    hp_json = json.dumps(hp_overrides)

    cmd = [
        PYTHON, "-u", ESTIMATE_SCRIPT,
        "--seed", str(SEED),
        "--num-epochs", str(NUM_EPOCHS),
        "--num-topics", str(NUM_TOPICS),
        "--output-dir", out_dir,
        "--hp-overrides", hp_json,
    ]

    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n{'=' * 60}")
    print(f"  VARIANT: {group}={value}  ({description})")
    print(f"  Overrides: {hp_json}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    run_start = time.time()
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            env=env, cwd=STBS_CAVI_DIR,
        )

    run_time = time.time() - run_start
    if result.returncode == 0:
        print(f"  DONE in {run_time/60:.1f} min")
        return out_dir
    else:
        print(f"  FAILED (code {result.returncode}) after {run_time/60:.1f} min")
        print(f"  Check log: {log_path}")
        return None


# ================================================================== #
# Compare two result sets
# ================================================================== #

def load_results(results_dir):
    """Load ideal points, theta, beta, eta, iota from a results directory."""
    param_dir = os.path.join(results_dir, "params")
    res = {}

    # Ideal points (N_authors x K)
    res["ideal_loc"] = np.load(os.path.join(param_dir, "ideal_point_location_final.npy"))

    # Theta (D x K) -> aggregate per author
    theta_shape = np.load(os.path.join(param_dir, "theta_shape_final.npy"))
    theta_rate = np.load(os.path.join(param_dir, "theta_rate_final.npy"))
    res["E_theta"] = theta_shape / theta_rate

    # Beta (K x V)
    beta_shape = np.load(os.path.join(param_dir, "beta_shape_final.npy"))
    beta_rate = np.load(os.path.join(param_dir, "beta_rate_final.npy"))
    res["E_beta"] = beta_shape / beta_rate

    # Eta (K x V)
    res["eta_loc"] = np.load(os.path.join(param_dir, "eta_location_final.npy"))

    # Iota (K x L)
    res["iota_loc"] = np.load(os.path.join(param_dir, "iota_location_final.npy"))

    # Aggregated weighted ideal points
    author_indices = np.load(os.path.join(CLEAN_DIR, "author_indices114.npy"))
    n_authors = res["ideal_loc"].shape[0]
    K = res["ideal_loc"].shape[1]
    theta_agg = np.zeros((n_authors, K))
    for a in range(n_authors):
        mask = (author_indices == a)
        if mask.sum() > 0:
            theta_agg[a] = res["E_theta"][mask].mean(axis=0)
    w_norm = theta_agg / theta_agg.sum(axis=1, keepdims=True)
    res["agg_ip"] = (res["ideal_loc"] * w_norm).sum(axis=1)

    return res


def compare_results(baseline, variant, label):
    """Compare baseline and variant results. Returns summary dict."""
    summary = {"label": label}

    # Ideal points: per-topic correlation
    ip_base = baseline["ideal_loc"]
    ip_var = variant["ideal_loc"]
    topic_corrs = []
    for k in range(ip_base.shape[1]):
        r = np.corrcoef(ip_base[:, k], ip_var[:, k])[0, 1]
        topic_corrs.append(abs(r))  # abs because of possible label switching
    summary["ip_topic_corr_mean"] = np.mean(topic_corrs)
    summary["ip_topic_corr_min"] = np.min(topic_corrs)

    # Aggregated IP correlation (handle label switching)
    r_agg = np.corrcoef(baseline["agg_ip"], variant["agg_ip"])[0, 1]
    summary["ip_agg_corr"] = abs(r_agg)

    # RMSE
    summary["ip_rmse"] = np.sqrt(np.mean((ip_base - ip_var) ** 2))

    # Iota (regression coefficients)
    iota_base = baseline["iota_loc"]
    iota_var = variant["iota_loc"]
    r_iota = np.corrcoef(iota_base.ravel(), iota_var.ravel())[0, 1]
    summary["iota_corr"] = abs(r_iota)
    summary["iota_rmse"] = np.sqrt(np.mean((iota_base - iota_var) ** 2))

    # Eta (polarity loadings)
    eta_base = baseline["eta_loc"]
    eta_var = variant["eta_loc"]
    eta_corrs = []
    for k in range(eta_base.shape[0]):
        r = np.corrcoef(eta_base[k], eta_var[k])[0, 1]
        eta_corrs.append(abs(r))
    summary["eta_topic_corr_mean"] = np.mean(eta_corrs)

    # Beta (topic-word distributions)
    beta_base = baseline["E_beta"]
    beta_var = variant["E_beta"]
    beta_corrs = []
    for k in range(beta_base.shape[0]):
        r = np.corrcoef(beta_base[k], beta_var[k])[0, 1]
        beta_corrs.append(abs(r))
    summary["beta_topic_corr_mean"] = np.mean(beta_corrs)

    return summary


# ================================================================== #
# Plotting
# ================================================================== #

def create_plots(df, output_dir):
    """Create summary plots for hyperparameter sensitivity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(output_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    groups = df["group"].unique()
    metrics = [
        ("ip_agg_corr", "Agg. IP Correlation\n(vs baseline)"),
        ("ip_topic_corr_mean", "Mean Topic IP Corr.\n(vs baseline)"),
        ("iota_corr", "Iota Correlation\n(vs baseline)"),
        ("iota_rmse", "Iota RMSE\n(vs baseline)"),
        ("eta_topic_corr_mean", "Mean Eta Corr.\n(vs baseline)"),
        ("beta_topic_corr_mean", "Mean Beta Corr.\n(vs baseline)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    colors = {"omega": "#2196F3", "beta": "#4CAF50", "rho": "#FF9800",
              "theta": "#E91E63", "ideal": "#9C27B0"}

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes[ax_idx]
        n_groups = len(groups)
        bar_width = 0.8 / n_groups

        for gi, group in enumerate(groups):
            sub = df[df["group"] == group].sort_values("value")
            x_base = np.arange(len(sub))
            x_pos = x_base + gi * bar_width
            ax.bar(x_pos, sub[metric], width=bar_width,
                   label=group if ax_idx == 0 else None,
                   color=colors.get(group, "#666"), alpha=0.85)

        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(HP_VALUES)) + bar_width * (n_groups - 1) / 2)
        ax.set_xticklabels([str(v) for v in HP_VALUES])
        ax.set_xlabel("Hyperparameter value")
        if "corr" in metric.lower():
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, ls="--", color="gray", lw=0.8)

    fig.legend(groups, loc="upper center", ncol=n_groups,
               bbox_to_anchor=(0.5, 1.02), fontsize=10,
               title="Hyperparameter group")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(fig_dir, "hyperparameter_sensitivity.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "hyperparameter_sensitivity.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Plots saved to {fig_dir}/")


# ================================================================== #
# MAIN
# ================================================================== #

def run_group(group):
    """Run all non-baseline variants for one HP group."""
    description, _ = HP_GROUPS[group]
    print(f"\n{'#' * 60}")
    print(f"  GROUP: {group} — {description}")
    print(f"{'#' * 60}")

    succeeded = []
    for value in HP_VALUES:
        if value == 0.3:
            print(f"\n  {group}=0.3 is the baseline, skipping estimation.")
            succeeded.append(value)
            continue

        out_dir = run_variant(group, value, description)
        if out_dir is not None:
            succeeded.append(value)

        # Cooling pause between runs
        print(f"\n  Cooling down 5 min...")
        time.sleep(300)

    return succeeded


def run_analyze():
    """Collect all results, compare to baseline, create plots."""
    print("\n" + "=" * 60)
    print("  ANALYZE: comparing all variants to baseline")
    print("=" * 60)

    if not os.path.isdir(BASELINE_DIR):
        print(f"ERROR: Baseline not found at {BASELINE_DIR}")
        print(f"Run seed={SEED} K={NUM_TOPICS} first (via 02_run_multi_seeds.py)")
        sys.exit(1)

    print(f"  Loading baseline from {BASELINE_DIR}...")
    baseline = load_results(BASELINE_DIR)

    all_summaries = []
    for group in HP_GROUPS:
        description, _ = HP_GROUPS[group]
        for value in HP_VALUES:
            label = f"{group}={value}"

            if value == 0.3:
                # Baseline = identity comparison
                summary = compare_results(baseline, baseline, label)
            else:
                var_dir = os.path.join(RESULTS_BASE, f"hp_{group}_{value}_K{NUM_TOPICS}")
                if not os.path.isdir(var_dir):
                    print(f"  WARNING: {var_dir} not found, skipping {label}")
                    continue
                print(f"  Comparing {label}...")
                variant = load_results(var_dir)
                summary = compare_results(baseline, variant, label)

            summary["group"] = group
            summary["value"] = value
            summary["description"] = description
            all_summaries.append(summary)

    if not all_summaries:
        print("  ERROR: No results found!")
        return

    df = pd.DataFrame(all_summaries)
    csv_path = os.path.join(RESULTS_BASE, "hyperparameter_sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    print("\n" + df[["group", "value", "ip_agg_corr", "iota_corr",
                      "eta_topic_corr_mean", "beta_topic_corr_mean"]].to_string(index=False))

    create_plots(df, RESULTS_BASE)
    print("\nDone.")


def run_all():
    """Run all groups sequentially, then analyze."""
    start_time = time.time()
    for group in HP_GROUPS:
        run_group(group)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"ALL ESTIMATIONS DONE in {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"{'=' * 60}")

    run_analyze()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 02c_simulation_hyperparameters.py <command>")
        print()
        print("Commands:")
        print("  all        Run ALL groups sequentially + analyze (recommended)")
        print("  omega      Run omega group only (iota shrinkage): 0.1, 1.0")
        print("  beta       Run beta group only (topic-word sparsity): 0.1, 1.0")
        print("  rho        Run rho group only (polarity shrinkage): 0.1, 1.0")
        print("  theta      Run theta group only (doc-topic sparsity): 0.1, 1.0")
        print("  ideal      Run ideal group only (IP precision): 0.1, 1.0")
        print("  analyze    Compare all variants to baseline + create plots")
        print()
        print("Workflow:")
        print(f"  1. Ensure baseline exists: {BASELINE_DIR}")
        print(f"  2. Run: python3 02c_simulation_hyperparameters.py all")
        print(f"  3. Or run groups individually and then: analyze")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "all":
        run_all()
    elif cmd == "analyze":
        run_analyze()
    elif cmd in HP_GROUPS:
        run_group(cmd)
    else:
        print(f"Unknown command: {cmd}")
        print(f"Valid: all, {', '.join(HP_GROUPS.keys())}, analyze")
        sys.exit(1)
