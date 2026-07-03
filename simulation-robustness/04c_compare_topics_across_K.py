#!/usr/bin/env python3
"""
04c_compare_topics_across_K.py
==============================
Compare STBS-CAVI results across different numbers of topics (K).

All runs use the same seed (123456) so differences are purely due to K.
We compare K in {15, 20, 25, 30} on:

  1. DW-NOMINATE correlation (aggregated ideal points)
  2. Topic-word alignment: for each (K_a, K_b) pair, use Hungarian matching
     on the rectangular cosine-similarity matrix of beta.
     min(K_a, K_b) topics are matched; the rest remain unmatched.
  3. ELBO (final training loss)
  4. Aggregated ideal point correlation across K values

Output:
  - DW-NOMINATE correlation per K
  - Topic matching tables and similarity metrics across K pairs
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

SEED = 123456
K_VALUES = [15, 20, 25, 30]

# ================================================================== #
# DW-NOMINATE scores (114th Senate, 1st dimension)
# ================================================================== #
DW_NOMINATE = {
    "Addison Mcconnell":  0.402, "Al Franken":        -0.399,
    "Amy Klobuchar":     -0.278, "Angus King":        -0.167,
    "Barbara Boxer":     -0.450, "Barbara Mikulski":  -0.370,
    "Benjamin Cardin":   -0.320, "Benjamin Sasse":     0.669,
    "Bernard Sanders":   -0.545, "Bill Cassidy":       0.464,
    "Brian Schatz":      -0.448, "Charles Grassley":   0.362,
    "Charles Roberts":    0.413, "Charles Schumer":   -0.355,
    "Christopher Coons": -0.238, "Christopher Murphy":-0.287,
    "Claire Mccaskill":  -0.143, "Clarence Nelson":   -0.193,
    "Cory Booker":       -0.558, "Cory Gardner":       0.444,
    "Daniel Coats":       0.375, "Daniel Sullivan":    0.476,
    "David Perdue":       0.562, "David Vitter":       0.495,
    "Dean Heller":        0.463, "Deborah Stabenow":  -0.343,
    "Debra Fischer":      0.460, "Dianne Feinstein":  -0.268,
    "Edward Markey":     -0.517, "Elizabeth Warren":  -0.745,
    "Gary Peters":       -0.250, "Harry Reid":        -0.276,
    "James Inhofe":       0.552, "James Lankford":     0.590,
    "Jeanne Shaheen":    -0.221, "Jeff Flake":         0.855,
    "Jeff Merkley":      -0.453, "Jefferson Sessions": 0.549,
    "Jerry Moran":        0.413, "Joe Donnelly":      -0.119,
    "Joe Manchin":       -0.057, "John Barrasso":      0.540,
    "John Boozman":       0.426, "John Cornyn":        0.471,
    "John Hoeven":        0.401, "John Mccain":        0.381,
    "John Reed":         -0.360, "John Thune":         0.432,
    "Johnny Isakson":     0.402, "Jon Tester":        -0.213,
    "Joni Ernst":         0.537, "Kelly Ayotte":       0.351,
    "Kirsten Gillibrand":-0.449, "Lamar Alexander":    0.324,
    "Lindsey Graham":     0.367, "Lisa Murkowski":     0.203,
    "Marco Rubio":        0.616, "Maria Cantwell":    -0.308,
    "Marion Rounds":      0.351, "Mark Kirk":          0.274,
    "Mark Warner":       -0.213, "Martin Heinrich":   -0.334,
    "Mary Heitkamp":     -0.122, "Mazie Hirono":      -0.512,
    "Michael Bennet":    -0.242, "Michael Crapo":      0.505,
    "Michael Enzi":       0.545, "Mike Lee":           0.891,
    "Orrin Hatch":        0.382, "Patrick Leahy":     -0.360,
    "Patrick Toomey":     0.624, "Patty Murray":      -0.356,
    "Rafael Cruz":        0.781, "Rand Paul":          0.891,
    "Richard Blumenthal":-0.434, "Richard Burr":       0.435,
    "Richard Durbin":    -0.340, "Richard Shelby":     0.446,
    "Robert Casey":      -0.313, "Robert Corker":      0.395,
    "Robert Menendez":   -0.366, "Robert Portman":     0.371,
    "Roger Wicker":       0.377, "Ron Johnson":        0.643,
    "Ronald Wyden":      -0.334, "Roy Blunt":          0.398,
    "Sheldon Whitehouse":-0.349, "Shelley Capito":     0.284,
    "Sherrod Brown":     -0.425, "Steve Daines":       0.590,
    "Susan Collins":      0.124, "Tammy Baldwin":     -0.486,
    "Thomas Carper":     -0.179, "Thomas Tillis":      0.386,
    "Thomas Udall":      -0.453, "Tim Scott":          0.629,
    "Timothy Kaine":     -0.237, "Tom Cotton":         0.578,
    "William Cochran":    0.287,
}

AUTHOR_MAP_TO_NOMINATE = {
    "Alan Franken":      "Al Franken",
    "Bill Nelson":       "Clarence Nelson",
    "Bob Corker":        "Robert Corker",
    "Chris Coons":       "Christopher Coons",
    "Dan Sullivan":      "Daniel Sullivan",
    "Deb Fischer":       "Debra Fischer",
    "Debbie Stabenow":   "Deborah Stabenow",
    "Heidi Heitkamp":    "Mary Heitkamp",
    "John Isakson":      "Johnny Isakson",
    "Mike Rounds":       "Marion Rounds",
    "Mitch Mcconnell":   "Addison Mcconnell",
    "Pat Roberts":       "Charles Roberts",
    "Ron Wyden":         "Ronald Wyden",
    "Ted Cruz":          "Rafael Cruz",
    "Thad Cochran":      "William Cochran",
    "Thom Tillis":       "Thomas Tillis",
    "Tom Udall":         "Thomas Udall",
}


# ================================================================== #
# Load results for one K
# ================================================================== #

def load_K_results(K):
    """Load all relevant parameters for one K value (fixed seed)."""
    seed_dir = os.path.join(RESULTS_BASE, f"seed_{SEED}_K{K}")
    param_dir = os.path.join(seed_dir, "params")

    res = {"K": K}

    # Beta: E[beta] = shape/rate, (K, V)
    beta_shape = np.load(os.path.join(param_dir, "beta_shape_final.npy"))
    beta_rate = np.load(os.path.join(param_dir, "beta_rate_final.npy"))
    res["beta"] = beta_shape / beta_rate
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

    # ELBO (final value from training_loss.csv)
    loss_path = os.path.join(seed_dir, "training_loss.csv")
    if os.path.exists(loss_path):
        loss_df = pd.read_csv(loss_path)
        res["final_elbo"] = loss_df["ELBO"].iloc[-1]
    else:
        res["final_elbo"] = np.nan

    return res


# ================================================================== #
# DW-NOMINATE comparison
# ================================================================== #

def compute_dw_nominate_corr(res):
    """Compute Pearson correlation of aggregated IPs with DW-NOMINATE."""
    # Load author names
    author_map_path = os.path.join(CLEAN_DIR, "author_map114.txt")
    with open(author_map_path) as f:
        author_names = [line.strip().rsplit(" (", 1)[0] for line in f if line.strip()]

    agg_ip = res["agg_ip"]
    n_authors = len(agg_ip)

    ip_vals = []
    dw_vals = []
    matched_names = []

    for idx in range(n_authors):
        if idx >= len(author_names):
            continue
        name = author_names[idx]
        nominate_key = AUTHOR_MAP_TO_NOMINATE.get(name, name)
        if nominate_key in DW_NOMINATE:
            ip_vals.append(agg_ip[idx])
            dw_vals.append(DW_NOMINATE[nominate_key])
            matched_names.append(name)

    ip_arr = np.array(ip_vals)
    dw_arr = np.array(dw_vals)

    r = np.corrcoef(ip_arr, dw_arr)[0, 1]
    # Handle label switching: flip if negative
    if r < 0:
        r = -r
        ip_arr = -ip_arr

    return r, len(matched_names)


# ================================================================== #
# Topic matching (rectangular Hungarian)
# ================================================================== #

def compute_similarity_matrix(beta_a, beta_b):
    """Cosine similarity matrix between topics. Can be rectangular."""
    Ka, Kb = beta_a.shape[0], beta_b.shape[0]
    sim = np.zeros((Ka, Kb))
    for i in range(Ka):
        for j in range(Kb):
            sim[i, j] = 1.0 - cosine(beta_a[i], beta_b[j])
    return sim


def match_topics(beta_a, beta_b):
    """Hungarian matching on (possibly rectangular) similarity matrix.

    Returns:
        matching: list of (topic_A, topic_B) pairs (length = min(Ka, Kb))
        sim_matrix: full (Ka x Kb) similarity matrix
        matched_sims: similarity for each matched pair
    """
    sim_matrix = compute_similarity_matrix(beta_a, beta_b)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    matching = list(zip(row_ind, col_ind))
    matched_sims = [sim_matrix[i, j] for i, j in matching]
    return matching, sim_matrix, matched_sims


# ================================================================== #
# Compare aligned topics
# ================================================================== #

def compare_aligned(res_a, res_b, matching):
    """Compare two K-results after topic alignment."""
    rows = []
    for topic_a, topic_b in matching:
        row = {
            "topic_A": topic_a,
            "topic_B": topic_b,
            "K_A": res_a["K"],
            "K_B": res_b["K"],
        }

        # Beta cosine similarity
        row["beta_cosine"] = 1.0 - cosine(res_a["beta_prob"][topic_a],
                                            res_b["beta_prob"][topic_b])

        # Eta correlation (abs for sign identification)
        r_eta = np.corrcoef(res_a["eta"][topic_a],
                            res_b["eta"][topic_b])[0, 1]
        row["eta_corr"] = abs(r_eta)

        # Iota correlation
        r_iota = np.corrcoef(res_a["iota"][topic_a],
                             res_b["iota"][topic_b])[0, 1]
        row["iota_corr"] = abs(r_iota)

        # eta * ideal: the ideological effect that matters
        effect_a = np.outer(res_a["ideal"][:, topic_a], res_a["eta"][topic_a])
        effect_b = np.outer(res_b["ideal"][:, topic_b], res_b["eta"][topic_b])
        row["eta_x_ideal_corr"] = np.corrcoef(effect_a.ravel(),
                                               effect_b.ravel())[0, 1]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Aggregated IP correlation
    r_agg = np.corrcoef(res_a["agg_ip"], res_b["agg_ip"])[0, 1]
    df.attrs["agg_ip_corr"] = abs(r_agg)

    return df


# ================================================================== #
# Top words for labeling
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

def plot_similarity_matrix(sim_matrix, matching, K_a, K_b, fig_dir):
    """Plot rectangular similarity matrix with matching highlighted."""
    fig, ax = plt.subplots(figsize=(max(8, K_b * 0.4), max(6, K_a * 0.35)))
    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    for i, j in matching:
        ax.plot(j, i, "s", color="blue", markersize=7, markerfacecolor="none",
                markeredgewidth=2)

    ax.set_xlabel(f"K={K_b} topics", fontsize=12)
    ax.set_ylabel(f"K={K_a} topics", fontsize=12)
    ax.set_title(f"Topic Cosine Similarity (K={K_a} vs K={K_b})", fontsize=14)
    ax.set_xticks(range(sim_matrix.shape[1]))
    ax.set_yticks(range(sim_matrix.shape[0]))
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()

    plt.savefig(os.path.join(fig_dir, f"topic_similarity_K{K_a}_vs_K{K_b}.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, f"topic_similarity_K{K_a}_vs_K{K_b}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()


def plot_dw_nominate_by_K(dw_results, fig_dir):
    """Bar chart of DW-NOMINATE correlation by K."""
    fig, ax = plt.subplots(figsize=(8, 5))
    Ks = [r["K"] for r in dw_results]
    corrs = [r["dw_corr"] for r in dw_results]

    bars = ax.bar(range(len(Ks)), corrs, color="steelblue", width=0.6)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"K={k}" for k in Ks], fontsize=12)
    ax.set_ylabel("Pearson r (with DW-NOMINATE)", fontsize=12)
    ax.set_title("DW-NOMINATE Correlation by Number of Topics", fontsize=14)
    ax.set_ylim(0, 1.0)

    for bar, corr in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{corr:.3f}", ha="center", va="bottom", fontsize=11)

    ax.axhline(0.864, ls="--", color="gray", lw=1, label="Original paper (0.864)")
    ax.legend(fontsize=10)
    fig.tight_layout()

    plt.savefig(os.path.join(fig_dir, "dw_nominate_by_K.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "dw_nominate_by_K.png"), bbox_inches="tight", dpi=150)
    plt.close()


def plot_alignment_summary_K(all_comparisons, fig_dir):
    """Sorted per-topic alignment quality across K pairs."""
    metrics = ["beta_cosine", "eta_corr", "iota_corr", "eta_x_ideal_corr"]
    titles = ["Beta (cosine sim.)", "Eta |corr|",
              "Iota |corr|", "Eta x Ideal corr"]

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
        ax.legend(fontsize=7)
        if "corr" in metric:
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, ls="--", color="gray", lw=0.5)
        else:
            ax.set_ylim(0, 1.05)

    fig.suptitle("Topic alignment quality across K values (sorted per pair)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(fig_dir, "topic_alignment_summary_K.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "topic_alignment_summary_K.png"),
                bbox_inches="tight", dpi=150)
    plt.close()


def plot_ideal_point_scatter(K_results, fig_dir):
    """Scatter plots of aggregated ideal points: each K vs K=25."""
    ref_K = 25
    if ref_K not in K_results:
        return
    ref = K_results[ref_K]

    other_Ks = [k for k in sorted(K_results.keys()) if k != ref_K]
    n_plots = len(other_Ks)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, k in zip(axes, other_Ks):
        res = K_results[k]
        ip_ref = ref["agg_ip"]
        ip_k = res["agg_ip"]

        r = abs(np.corrcoef(ip_ref, ip_k)[0, 1])
        ax.scatter(ip_ref, ip_k, s=20, alpha=0.7, c="steelblue")
        ax.set_xlabel(f"Agg. ideal point (K={ref_K})", fontsize=11)
        ax.set_ylabel(f"Agg. ideal point (K={k})", fontsize=11)
        ax.set_title(f"K={ref_K} vs K={k}  (|r| = {r:.3f})", fontsize=12)

        # Add 45-degree line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)

    fig.suptitle("Aggregated Ideal Points Across K", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(fig_dir, "ideal_points_across_K.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "ideal_points_across_K.png"),
                bbox_inches="tight", dpi=150)
    plt.close()


def plot_elbo_by_K(K_results, fig_dir):
    """Bar chart of final ELBO by K."""
    Ks = sorted(K_results.keys())
    elbos = [K_results[k]["final_elbo"] for k in Ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(Ks)), elbos, color="darkorange", width=0.6)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"K={k}" for k in Ks], fontsize=12)
    ax.set_ylabel("Final ELBO", fontsize=12)
    ax.set_title("Evidence Lower Bound by Number of Topics", fontsize=14)

    for bar, elbo in zip(bars, elbos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{elbo:,.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, "elbo_by_K.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "elbo_by_K.png"), bbox_inches="tight", dpi=150)
    plt.close()


# ================================================================== #
# MAIN
# ================================================================== #

def main():
    print("=" * 60)
    print("  Topic comparison across K values (seed fixed at %d)" % SEED)
    print("=" * 60)

    # Load vocabulary
    vocab_path = os.path.join(CLEAN_DIR, "vocabulary114.txt")
    with open(vocab_path) as f:
        vocab = np.array([line.strip() for line in f if line.strip()])
    print(f"  Vocabulary: {len(vocab)} terms")

    # Output directory
    fig_dir = os.path.join(RESULTS_BASE, "K_comparison")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Load all K results ---- #
    K_results = {}
    dw_results = []

    for K in K_VALUES:
        print(f"\n  Loading K={K}...")
        res = load_K_results(K)
        K_results[K] = res
        print(f"    beta: {res['beta'].shape}, ideal: {res['ideal'].shape}")
        print(f"    Final ELBO: {res['final_elbo']:,.0f}")

        # DW-NOMINATE correlation
        r, n_matched = compute_dw_nominate_corr(res)
        dw_results.append({"K": K, "dw_corr": r, "n_matched": n_matched})
        print(f"    DW-NOMINATE corr: r = {r:.4f} (n = {n_matched})")

    # ---- DW-NOMINATE summary ---- #
    dw_df = pd.DataFrame(dw_results)
    dw_df.to_csv(os.path.join(fig_dir, "dw_nominate_by_K.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("  DW-NOMINATE correlations by K")
    print(f"{'=' * 60}")
    print(dw_df.to_string(index=False))

    # ---- ELBO summary ---- #
    elbo_rows = [{"K": K, "final_ELBO": K_results[K]["final_elbo"]}
                 for K in K_VALUES]
    elbo_df = pd.DataFrame(elbo_rows)
    elbo_df.to_csv(os.path.join(fig_dir, "elbo_by_K.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("  Final ELBO by K")
    print(f"{'=' * 60}")
    print(elbo_df.to_string(index=False))

    # ---- Pairwise topic comparison ---- #
    all_comparisons = {}
    all_summaries = []

    for K_a, K_b in itertools.combinations(K_VALUES, 2):
        pair_label = f"K={K_a} vs K={K_b}"
        print(f"\n{'=' * 60}")
        print(f"  Matching: {pair_label}")
        print(f"{'=' * 60}")

        res_a = K_results[K_a]
        res_b = K_results[K_b]

        matching, sim_matrix, matched_sims = match_topics(
            res_a["beta_prob"], res_b["beta_prob"]
        )

        n_matched = len(matching)
        K_min = min(K_a, K_b)
        K_max = max(K_a, K_b)
        n_unmatched = K_max - K_min

        print(f"  Matched: {n_matched} topics, Unmatched in K={K_max}: {n_unmatched}")

        # Print matching table
        labels_a = get_top_words(res_a["beta_prob"], vocab, n=3)
        labels_b = get_top_words(res_b["beta_prob"], vocab, n=3)

        print(f"\n  {'Topic A':>8s} -> {'Topic B':>8s}  {'Cos.Sim':>8s}  "
              f"{'K={K_a} top words':<35s}  {'K={K_b} top words':<35s}")
        print(f"  {'-'*8}    {'-'*8}  {'-'*8}  {'-'*35}  {'-'*35}")
        for (ta, tb), sim in sorted(zip(matching, matched_sims),
                                     key=lambda x: -x[1]):
            print(f"  {ta:>8d} -> {tb:>8d}  {sim:>8.4f}  "
                  f"{labels_a[ta]:<35s}  {labels_b[tb]:<35s}")

        mean_sim = np.mean(matched_sims)
        print(f"\n  Mean cosine similarity: {mean_sim:.4f}")
        print(f"  Min  cosine similarity: {np.min(matched_sims):.4f}")
        print(f"  Max  cosine similarity: {np.max(matched_sims):.4f}")

        # Count well-matched
        n_well = sum(1 for s in matched_sims if s > 0.5)
        n_high = sum(1 for s in matched_sims if s > 0.8)
        print(f"  Well-matched (cos > 0.5): {n_well}/{n_matched}")
        print(f"  Highly matched (cos > 0.8): {n_high}/{n_matched}")

        # Compare after alignment
        comp_df = compare_aligned(res_a, res_b, matching)
        all_comparisons[pair_label] = comp_df

        agg_corr = comp_df.attrs["agg_ip_corr"]
        print(f"\n  After alignment:")
        print(f"    Beta cosine (mean):      {comp_df['beta_cosine'].mean():.4f}")
        print(f"    Eta |corr| (mean):       {comp_df['eta_corr'].mean():.4f}")
        print(f"    eta*ideal corr (mean):   {comp_df['eta_x_ideal_corr'].mean():.4f}")
        print(f"    Iota |corr| (mean):      {comp_df['iota_corr'].mean():.4f}")
        print(f"    Agg. IP |corr|:          {agg_corr:.4f}")

        # Save per-pair comparison
        comp_df.to_csv(os.path.join(fig_dir,
                       f"topic_comparison_K{K_a}_vs_K{K_b}.csv"), index=False)

        # Plot similarity matrix
        plot_similarity_matrix(sim_matrix, matching, K_a, K_b, fig_dir)

        # Summary row
        all_summaries.append({
            "K_A": K_a, "K_B": K_b,
            "n_matched": n_matched,
            "n_unmatched_in_larger": n_unmatched,
            "beta_cosine_mean": comp_df["beta_cosine"].mean(),
            "beta_cosine_min": comp_df["beta_cosine"].min(),
            "n_well_matched_05": n_well,
            "n_highly_matched_08": n_high,
            "eta_x_ideal_corr_mean": comp_df["eta_x_ideal_corr"].mean(),
            "iota_corr_mean": comp_df["iota_corr"].mean(),
            "agg_ip_corr": agg_corr,
        })

    # ---- Summary table ---- #
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(fig_dir, "K_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print("  SUMMARY across all K pairs")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))

    # ---- Plots ---- #
    plot_dw_nominate_by_K(dw_results, fig_dir)
    plot_alignment_summary_K(all_comparisons, fig_dir)
    plot_ideal_point_scatter(K_results, fig_dir)
    plot_elbo_by_K(K_results, fig_dir)

    print(f"\n  Results saved to: {fig_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
