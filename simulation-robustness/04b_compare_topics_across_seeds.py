#!/usr/bin/env python3
"""
04b_compare_topics_across_seeds.py
==================================
Compare topics across different seeds by solving the label-switching problem.

Topic k in seed A may correspond to topic j in seed B. We align topics using
the Hungarian algorithm on cosine similarity of topic-word distributions (beta).

After alignment, we compare:
  - Topic-word distributions (beta): cosine similarity
  - Ideological word loadings (eta): correlation
  - Regression coefficients (iota): correlation
  - Ideal points: correlation per aligned topic
  - Aggregated (theta-weighted) ideal points: correlation

Output:
  - Topic matching tables (which topic maps to which)
  - Per-topic similarity metrics after alignment
  - Summary statistics and plots
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================== #
# CONFIGURATION
# ================================================================== #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(SCRIPT_DIR, "stbs_cavi_results")
STBS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "STBS_CAVI"))
CLEAN_DIR = os.path.join(STBS_DIR, "data", "hein-daily", "clean")

SEEDS = [314159, 42, 123456]
NUM_TOPICS = 25


# ================================================================== #
# Load results for one seed
# ================================================================== #

def load_seed_results(seed, K=NUM_TOPICS):
    """Load all relevant parameters for one seed."""
    seed_dir = os.path.join(RESULTS_BASE, f"seed_{seed}_K{K}")
    param_dir = os.path.join(seed_dir, "params")

    res = {}
    # Beta: E[beta] = shape/rate, (K, V) — topic-word distributions
    beta_shape = np.load(os.path.join(param_dir, "beta_shape_final.npy"))
    beta_rate = np.load(os.path.join(param_dir, "beta_rate_final.npy"))
    res["beta"] = beta_shape / beta_rate

    # Normalize beta to probabilities per topic (for cleaner similarity)
    res["beta_prob"] = res["beta"] / res["beta"].sum(axis=1, keepdims=True)

    # Eta: ideological word loadings (K, V)
    res["eta"] = np.load(os.path.join(param_dir, "eta_location_final.npy"))

    # Iota: regression coefficients (K, L)
    res["iota"] = np.load(os.path.join(param_dir, "iota_location_final.npy"))

    # Ideal points (N_authors, K)
    res["ideal"] = np.load(os.path.join(param_dir, "ideal_point_location_final.npy"))

    # Theta -> aggregated weighted ideal points
    theta_shape = np.load(os.path.join(param_dir, "theta_shape_final.npy"))
    theta_rate = np.load(os.path.join(param_dir, "theta_rate_final.npy"))
    E_theta = theta_shape / theta_rate

    author_indices = np.load(os.path.join(CLEAN_DIR, "author_indices114.npy"))
    n_authors = res["ideal"].shape[0]
    theta_agg = np.zeros((n_authors, K))
    for a in range(n_authors):
        mask = (author_indices == a)
        if mask.sum() > 0:
            theta_agg[a] = E_theta[mask].mean(axis=0)
    w_norm = theta_agg / theta_agg.sum(axis=1, keepdims=True)
    res["agg_ip"] = (res["ideal"] * w_norm).sum(axis=1)
    res["theta_weights"] = w_norm

    return res


# ================================================================== #
# Topic matching via Hungarian algorithm
# ================================================================== #

def compute_similarity_matrix(beta_a, beta_b):
    """Compute cosine similarity matrix between topics of two seeds.

    Returns:
        sim: (K, K) matrix where sim[i,j] = cosine_similarity(topic_i_A, topic_j_B)
    """
    K = beta_a.shape[0]
    sim = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            sim[i, j] = 1.0 - cosine(beta_a[i], beta_b[j])
    return sim


def match_topics(beta_a, beta_b):
    """Find optimal 1-to-1 topic matching using Hungarian algorithm.

    Returns:
        matching: list of (topic_A, topic_B) pairs
        sim_matrix: full similarity matrix
        matched_sims: similarity for each matched pair
    """
    sim_matrix = compute_similarity_matrix(beta_a, beta_b)

    # Hungarian algorithm minimizes cost, so use negative similarity
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    matching = list(zip(row_ind, col_ind))
    matched_sims = [sim_matrix[i, j] for i, j in matching]

    return matching, sim_matrix, matched_sims


# ================================================================== #
# Compare aligned topics
# ================================================================== #

def compare_aligned(res_a, res_b, matching):
    """Compare two seed results after topic alignment.

    Returns DataFrame with per-topic comparison metrics.
    """
    rows = []
    for topic_a, topic_b in matching:
        row = {
            "topic_A": topic_a,
            "topic_B": topic_b,
        }

        # Beta similarity (already computed, but redo for normalized version)
        row["beta_cosine"] = 1.0 - cosine(res_a["beta_prob"][topic_a],
                                            res_b["beta_prob"][topic_b])

        # Eta correlation (abs to handle sign flipping of ideological dimension)
        r_eta = np.corrcoef(res_a["eta"][topic_a],
                            res_b["eta"][topic_b])[0, 1]
        row["eta_corr_raw"] = r_eta
        row["eta_corr"] = abs(r_eta)

        # Iota correlation (regression coefficients for this topic)
        r_iota = np.corrcoef(res_a["iota"][topic_a],
                             res_b["iota"][topic_b])[0, 1]
        row["iota_corr_raw"] = r_iota
        row["iota_corr"] = abs(r_iota)

        # Ideal point correlation for this topic (abs to handle sign flipping)
        r_ideal = np.corrcoef(res_a["ideal"][:, topic_a],
                              res_b["ideal"][:, topic_b])[0, 1]
        row["ideal_corr_raw"] = r_ideal
        row["ideal_corr"] = abs(r_ideal)

        # eta * ideal: the actual ideological effect that matters
        # exp(eta_kv * ideal_nk) determines partisan word usage.
        # If both eta and ideal flip sign, the product is unchanged.
        # Compare the full (N x V) effect matrix across seeds.
        effect_a = np.outer(res_a["ideal"][:, topic_a], res_a["eta"][topic_a])  # (N, V)
        effect_b = np.outer(res_b["ideal"][:, topic_b], res_b["eta"][topic_b])  # (N, V)
        row["eta_x_ideal_corr"] = np.corrcoef(effect_a.ravel(), effect_b.ravel())[0, 1]

        # Also: per-author mean ideological word profile
        # eta_k * mean(ideal_nk) gives the average ideological direction
        mean_effect_a = res_a["eta"][topic_a] * res_a["ideal"][:, topic_a].mean()
        mean_effect_b = res_b["eta"][topic_b] * res_b["ideal"][:, topic_b].mean()
        row["mean_ideo_effect_corr"] = np.corrcoef(mean_effect_a, mean_effect_b)[0, 1]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Aggregated IP correlation (not topic-specific)
    r_agg = np.corrcoef(res_a["agg_ip"], res_b["agg_ip"])[0, 1]
    df.attrs["agg_ip_corr"] = r_agg

    return df


# ================================================================== #
# Top words for topic labeling
# ================================================================== #

def get_top_words(beta_prob, vocab, n=5):
    """Get top n words for each topic."""
    K = beta_prob.shape[0]
    labels = []
    for k in range(K):
        top_idx = np.argsort(beta_prob[k])[::-1][:n]
        labels.append(", ".join(vocab[i] for i in top_idx))
    return labels


# ================================================================== #
# Plotting
# ================================================================== #

def plot_similarity_matrix(sim_matrix, matching, seed_a, seed_b, fig_dir):
    """Plot the topic similarity matrix with optimal matching highlighted."""
    K = sim_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

    # Highlight matched pairs
    for i, j in matching:
        ax.plot(j, i, "s", color="blue", markersize=8, markerfacecolor="none",
                markeredgewidth=2)

    ax.set_xlabel(f"Seed {seed_b} topics", fontsize=12)
    ax.set_ylabel(f"Seed {seed_a} topics", fontsize=12)
    ax.set_title(f"Topic Cosine Similarity (seed {seed_a} vs {seed_b})", fontsize=14)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()

    plt.savefig(os.path.join(fig_dir, f"topic_similarity_{seed_a}_vs_{seed_b}.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, f"topic_similarity_{seed_a}_vs_{seed_b}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()


def plot_alignment_summary(all_comparisons, fig_dir):
    """Bar chart of per-topic alignment quality across all seed pairs."""
    metrics = ["beta_cosine", "eta_corr", "iota_corr", "ideal_corr"]
    titles = ["Beta (cosine sim.)", "Eta (correlation)",
              "Iota (correlation)", "Ideal points (correlation)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]
        for pair_label, comp_df in all_comparisons.items():
            vals = comp_df[metric].values
            ax.plot(range(len(vals)), sorted(vals, reverse=True),
                    "o-", label=pair_label, markersize=4, alpha=0.8)
        ax.set_xlabel("Rank (sorted best to worst)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=8)
        if "corr" in metric:
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, ls="--", color="gray", lw=0.5)
        else:
            ax.set_ylim(0, 1.05)

    fig.suptitle("Topic alignment quality across seeds (sorted per pair)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(fig_dir, "topic_alignment_summary.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "topic_alignment_summary.png"),
                bbox_inches="tight", dpi=150)
    plt.close()


# ================================================================== #
# MAIN
# ================================================================== #

def main():
    print("=" * 60)
    print("  Topic comparison across seeds (label-switching alignment)")
    print("=" * 60)

    # Load vocabulary
    vocab_path = os.path.join(CLEAN_DIR, "vocabulary114.txt")
    with open(vocab_path) as f:
        vocab = np.array([line.strip() for line in f if line.strip()])
    print(f"  Vocabulary: {len(vocab)} terms")

    # Load all seed results
    seed_results = {}
    for seed in SEEDS:
        print(f"\n  Loading seed {seed}...")
        seed_results[seed] = load_seed_results(seed)
        print(f"    beta: {seed_results[seed]['beta'].shape}, "
              f"ideal: {seed_results[seed]['ideal'].shape}")

    # Output directory
    fig_dir = os.path.join(RESULTS_BASE, "topic_comparison")
    os.makedirs(fig_dir, exist_ok=True)

    # Compare all seed pairs
    all_comparisons = {}
    all_matchings = {}
    all_summaries = []

    for seed_a, seed_b in itertools.combinations(SEEDS, 2):
        pair_label = f"{seed_a} vs {seed_b}"
        print(f"\n{'=' * 60}")
        print(f"  Matching: {pair_label}")
        print(f"{'=' * 60}")

        res_a = seed_results[seed_a]
        res_b = seed_results[seed_b]

        # Match topics
        matching, sim_matrix, matched_sims = match_topics(
            res_a["beta_prob"], res_b["beta_prob"]
        )

        # Print matching table with top words
        labels_a = get_top_words(res_a["beta_prob"], vocab, n=3)
        labels_b = get_top_words(res_b["beta_prob"], vocab, n=3)

        print(f"\n  {'Topic A':>8s} -> {'Topic B':>8s}  {'Cos.Sim':>8s}  "
              f"{'Seed A top words':<40s}  {'Seed B top words':<40s}")
        print(f"  {'-'*8}    {'-'*8}  {'-'*8}  {'-'*40}  {'-'*40}")
        for (ta, tb), sim in zip(matching, matched_sims):
            print(f"  {ta:>8d} -> {tb:>8d}  {sim:>8.4f}  "
                  f"{labels_a[ta]:<40s}  {labels_b[tb]:<40s}")

        print(f"\n  Mean cosine similarity: {np.mean(matched_sims):.4f}")
        print(f"  Min  cosine similarity: {np.min(matched_sims):.4f}")
        print(f"  Max  cosine similarity: {np.max(matched_sims):.4f}")

        # Compare after alignment
        comp_df = compare_aligned(res_a, res_b, matching)
        all_comparisons[pair_label] = comp_df
        all_matchings[pair_label] = matching

        agg_corr = comp_df.attrs["agg_ip_corr"]
        print(f"\n  After alignment:")
        print(f"    Beta cosine (mean):      {comp_df['beta_cosine'].mean():.4f}")
        print(f"    Eta |corr| (mean):       {comp_df['eta_corr'].mean():.4f}")
        print(f"    Ideal |corr| (mean):     {comp_df['ideal_corr'].mean():.4f}")
        print(f"    eta*ideal corr (mean):   {comp_df['eta_x_ideal_corr'].mean():.4f}")
        print(f"    Iota |corr| (mean):      {comp_df['iota_corr'].mean():.4f}")
        print(f"    Agg. IP corr:            {agg_corr:.4f}")

        # Save per-pair comparison
        comp_df.to_csv(os.path.join(fig_dir, f"topic_comparison_{seed_a}_vs_{seed_b}.csv"),
                       index=False)

        # Plot similarity matrix
        plot_similarity_matrix(sim_matrix, matching, seed_a, seed_b, fig_dir)

        # Summary row
        all_summaries.append({
            "seed_A": seed_a, "seed_B": seed_b,
            "beta_cosine_mean": comp_df["beta_cosine"].mean(),
            "beta_cosine_min": comp_df["beta_cosine"].min(),
            "eta_x_ideal_corr_mean": comp_df["eta_x_ideal_corr"].mean(),
            "iota_corr_mean": comp_df["iota_corr"].mean(),
            "agg_ip_corr": agg_corr,
        })

    # Summary table
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(fig_dir, "topic_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print("  SUMMARY across all seed pairs")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))

    # Plot alignment summary
    plot_alignment_summary(all_comparisons, fig_dir)

    print(f"\n  Results saved to: {fig_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
