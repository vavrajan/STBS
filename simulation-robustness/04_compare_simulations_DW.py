"""
Compare STBS-CAVI estimated ideal points across seeds
with DW-NOMINATE scores (1st dimension) for the 114th US Senate.

Adapted from Revision_code/01b_compare_simulations.py for CAVI results.

For each seed:
  1. Load ideal points from ideal_points.csv
  2. Reconstruct theta from Gamma params (shape/rate), aggregate per author
  3. Weight topic-specific ideal points by author-theta -> single ideal point
  4. Correlate with DW-NOMINATE dim1 (flip sign if negative = label switching)
  5. Report correlation and variability across seeds
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(SCRIPT_DIR, "stbs_cavi_results")
STBS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "STBS_CAVI"))
CLEAN_DIR = os.path.join(STBS_DIR, "data", "hein-daily", "clean")

SEEDS = [314159, 42, 123456]
NUM_TOPICS = 25

# ------------------------------------------------------------------ #
# 1.  DW-NOMINATE scores for the 114th Senate (Voteview)
#     Key = "First Last" (matching author_map114.txt after stripping party)
# ------------------------------------------------------------------ #
DW_NOMINATE = {
    "Addison Mcconnell":  0.402,
    "Al Franken":        -0.399,
    "Amy Klobuchar":     -0.278,
    "Angus King":        -0.167,
    "Barbara Boxer":     -0.450,
    "Barbara Mikulski":  -0.370,
    "Benjamin Cardin":   -0.320,
    "Benjamin Sasse":     0.669,
    "Bernard Sanders":   -0.545,
    "Bill Cassidy":       0.464,
    "Brian Schatz":      -0.448,
    "Charles Grassley":   0.362,
    "Charles Roberts":    0.413,
    "Charles Schumer":   -0.355,
    "Christopher Coons": -0.238,
    "Christopher Murphy":-0.287,
    "Claire Mccaskill":  -0.143,
    "Clarence Nelson":   -0.193,
    "Cory Booker":       -0.558,
    "Cory Gardner":       0.444,
    "Daniel Coats":       0.375,
    "Daniel Sullivan":    0.476,
    "David Perdue":       0.562,
    "David Vitter":       0.495,
    "Dean Heller":        0.463,
    "Deborah Stabenow":  -0.343,
    "Debra Fischer":      0.460,
    "Dianne Feinstein":  -0.268,
    "Edward Markey":     -0.517,
    "Elizabeth Warren":  -0.745,
    "Gary Peters":       -0.250,
    "Harry Reid":        -0.276,
    "James Inhofe":       0.552,
    "James Lankford":     0.590,
    "Jeanne Shaheen":    -0.221,
    "Jeff Flake":         0.855,
    "Jeff Merkley":      -0.453,
    "Jefferson Sessions": 0.549,
    "Jerry Moran":        0.413,
    "Joe Donnelly":      -0.119,
    "Joe Manchin":       -0.057,
    "John Barrasso":      0.540,
    "John Boozman":       0.426,
    "John Cornyn":        0.471,
    "John Hoeven":        0.401,
    "John Mccain":        0.381,
    "John Reed":         -0.360,
    "John Thune":         0.432,
    "Johnny Isakson":     0.402,
    "Jon Tester":        -0.213,
    "Joni Ernst":         0.537,
    "Kelly Ayotte":       0.351,
    "Kirsten Gillibrand":-0.449,
    "Lamar Alexander":    0.324,
    "Lindsey Graham":     0.367,
    "Lisa Murkowski":     0.203,
    "Marco Rubio":        0.616,
    "Maria Cantwell":    -0.308,
    "Marion Rounds":      0.351,
    "Mark Kirk":          0.274,
    "Mark Warner":       -0.213,
    "Martin Heinrich":   -0.334,
    "Mary Heitkamp":     -0.122,
    "Mazie Hirono":      -0.512,
    "Michael Bennet":    -0.242,
    "Michael Crapo":      0.505,
    "Michael Enzi":       0.545,
    "Mike Lee":           0.891,
    "Orrin Hatch":        0.382,
    "Patrick Leahy":     -0.360,
    "Patrick Toomey":     0.624,
    "Patty Murray":      -0.356,
    "Rafael Cruz":        0.781,
    "Rand Paul":          0.891,
    "Richard Blumenthal":-0.434,
    "Richard Burr":       0.435,
    "Richard Durbin":    -0.340,
    "Richard Shelby":     0.446,
    "Robert Casey":      -0.313,
    "Robert Corker":      0.395,
    "Robert Menendez":   -0.366,
    "Robert Portman":     0.371,
    "Roger Wicker":       0.377,
    "Ron Johnson":        0.643,
    "Ronald Wyden":      -0.334,
    "Roy Blunt":          0.398,
    "Sheldon Whitehouse":-0.349,
    "Shelley Capito":     0.284,
    "Sherrod Brown":     -0.425,
    "Steve Daines":       0.590,
    "Susan Collins":      0.124,
    "Tammy Baldwin":     -0.486,
    "Thomas Carper":     -0.179,
    "Thomas Tillis":      0.386,
    "Thomas Udall":      -0.453,
    "Tim Scott":          0.629,
    "Timothy Kaine":     -0.237,
    "Tom Cotton":         0.578,
    "William Cochran":    0.287,
}

# ------------------------------------------------------------------ #
# Helper: build name mapping  ConcatenatedName -> DW_NOMINATE key
# ------------------------------------------------------------------ #
# author_map114.txt uses common names (e.g. "Ted Cruz", "Mitch Mcconnell")
# while DW_NOMINATE uses legal/formal names (e.g. "Rafael Cruz",
# "Addison Mcconnell").  This lookup bridges the two.
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

def build_name_map():
    """Read author_map114.txt and build mapping from concatenated names
    (e.g. 'AlanFranken') to DW_NOMINATE keys (e.g. 'Al Franken').
    """
    author_map_path = os.path.join(CLEAN_DIR, "author_map114.txt")
    with open(author_map_path) as f:
        raw_names = [line.strip() for line in f if line.strip()]

    name_map = {}
    for raw in raw_names:
        # "Alan Franken (D)" -> "Alan Franken"
        proper = raw.rsplit(" (", 1)[0]
        # Map to DW_NOMINATE key if different
        nominate_key = AUTHOR_MAP_TO_NOMINATE.get(proper, proper)
        # "Alan Franken" -> "AlanFranken"
        concat = proper.replace(" ", "")
        name_map[concat] = nominate_key

    return name_map


# ------------------------------------------------------------------ #
# 2.  Process each seed
# ------------------------------------------------------------------ #
print("=" * 60)
print("  STBS-CAVI: Compare ideal points with DW-NOMINATE")
print("=" * 60)

name_map = build_name_map()

# Load author_indices for theta aggregation
author_indices = np.load(os.path.join(CLEAN_DIR, "author_indices114.npy"))

# Read author_map for proper names by index
with open(os.path.join(CLEAN_DIR, "author_map114.txt")) as f:
    author_names_by_idx = [line.strip().rsplit(" (", 1)[0] for line in f if line.strip()]

aggregated_results = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  Processing seed {seed}")
    print(f"{'='*60}")

    seed_dir = os.path.join(RESULTS_BASE, f"seed_{seed}_K{NUM_TOPICS}")
    param_dir = os.path.join(seed_dir, "params")

    # -- load ideal points (N_authors x K_topics) --
    ip_df = pd.read_csv(os.path.join(seed_dir, "ideal_points.csv"), index_col="author")
    topic_cols = [c for c in ip_df.columns if c.startswith("topic_")]
    print(f"  Ideal points: {ip_df.shape}")

    # -- reconstruct theta from Gamma parameters --
    theta_shape = np.load(os.path.join(param_dir, "theta_shape_final.npy"))
    theta_rate = np.load(os.path.join(param_dir, "theta_rate_final.npy"))
    theta_mean = theta_shape / theta_rate  # E[theta] for Gamma distribution
    print(f"  Theta (docs): {theta_mean.shape}")

    # -- aggregate theta per author: mean across documents --
    n_authors = len(author_names_by_idx)
    theta_agg = np.zeros((n_authors, NUM_TOPICS))
    for a in range(n_authors):
        mask = (author_indices == a)
        if mask.sum() > 0:
            theta_agg[a] = theta_mean[mask].mean(axis=0)

    # Build DataFrame with proper names as index
    theta_agg_df = pd.DataFrame(theta_agg, columns=topic_cols,
                                 index=[n.replace(" ", "") for n in author_names_by_idx])
    print(f"  Theta aggregated per author: {theta_agg_df.shape}")

    # -- normalise theta to sum to 1 per author (weights) --
    theta_weights = theta_agg_df.div(theta_agg_df.sum(axis=1), axis=0)

    # -- weighted ideal point = sum_k  w_k * ip_k  per author --
    common_authors = ip_df.index.intersection(theta_weights.index)
    ip_aligned = ip_df.loc[common_authors, topic_cols].values
    tw_aligned = theta_weights.loc[common_authors, topic_cols].values

    weighted_ip = (ip_aligned * tw_aligned).sum(axis=1)

    result = pd.DataFrame({
        "author_concat": common_authors,
        "author": [name_map.get(a, a) for a in common_authors],
        "weighted_ip": weighted_ip,
    })

    # -- add DW-NOMINATE --
    result["dw_nominate"] = result["author"].map(DW_NOMINATE)
    missing = result[result["dw_nominate"].isna()]["author"].tolist()
    if missing:
        print(f"  WARNING: no DW-NOMINATE for: {missing}")
    result_matched = result.dropna(subset=["dw_nominate"])

    # -- handle label switching: flip sign if correlation is negative --
    corr_raw = np.corrcoef(result_matched["weighted_ip"],
                           result_matched["dw_nominate"])[0, 1]
    if corr_raw < 0:
        result_matched = result_matched.copy()
        result_matched["weighted_ip"] = -result_matched["weighted_ip"]
        print(f"  Label switching detected (r={corr_raw:.10f}), flipped sign.")
        corr_raw = -corr_raw

    result_matched["seed"] = seed
    aggregated_results.append(result_matched)
    print(f"  Pearson r = {corr_raw:.10f}  (N = {len(result_matched)} authors)")

# ------------------------------------------------------------------ #
# 3.  Combine all seeds and compute variability
# ------------------------------------------------------------------ #
all_df = pd.concat(aggregated_results, ignore_index=True)

# Pivot: authors x seeds -> weighted ideal points
pivot = all_df.pivot(index="author", columns="seed", values="weighted_ip")
pivot.columns = [f"ip_seed{s}" for s in pivot.columns]

# DW-NOMINATE (one value per author)
dw = all_df[["author", "dw_nominate"]].drop_duplicates().set_index("author")
pivot = pivot.join(dw)

# -- variability measures across seeds per author --
seed_cols = [c for c in pivot.columns if c.startswith("ip_seed")]
pivot["ip_mean"] = pivot[seed_cols].mean(axis=1)
pivot["ip_std"]  = pivot[seed_cols].std(axis=1)
pivot["ip_range"] = pivot[seed_cols].max(axis=1) - pivot[seed_cols].min(axis=1)

# -- correlations per seed --
print("\n" + "=" * 60)
print("  SUMMARY: Correlation with DW-NOMINATE (dim1) per seed")
print("=" * 60)
corr_per_seed = {}
for sc in seed_cols:
    r = np.corrcoef(pivot[sc], pivot["dw_nominate"])[0, 1]
    seed_label = sc.replace("ip_seed", "")
    corr_per_seed[seed_label] = r
    print(f"  Seed {seed_label:>7s}:  r = {r:.10f}")

corr_values = list(corr_per_seed.values())
print(f"\n  Mean r  = {np.mean(corr_values):.10f}")
print(f"  Std  r  = {np.std(corr_values):.10f}")
print(f"  Min  r  = {np.min(corr_values):.10f}")
print(f"  Max  r  = {np.max(corr_values):.10f}")

# -- mean across-author variability --
print(f"\n  Mean author-level IP std across seeds:   {pivot['ip_std'].mean():.8f}")
print(f"  Mean author-level IP range across seeds: {pivot['ip_range'].mean():.8f}")
print(f"  Max  author-level IP std across seeds:   {pivot['ip_std'].max():.8f}")
print(f"  Max  author-level IP range across seeds: {pivot['ip_range'].max():.8f}")

# -- pairwise max absolute differences between seeds --
print("\n" + "-" * 60)
print("  Pairwise max|mean absolute difference of ideal points")
print("-" * 60)
for i, s1 in enumerate(SEEDS):
    ip1 = pd.read_csv(os.path.join(RESULTS_BASE, f"seed_{s1}_K{NUM_TOPICS}",
                                     "ideal_points.csv"), index_col="author")
    for s2 in SEEDS[i+1:]:
        ip2 = pd.read_csv(os.path.join(RESULTS_BASE, f"seed_{s2}_K{NUM_TOPICS}",
                                         "ideal_points.csv"), index_col="author")
        common = ip1.index.intersection(ip2.index)
        diff = np.abs(ip1.loc[common].values - ip2.loc[common].values)
        print(f"  Seeds {s1:>6d} vs {s2:>6d}:  max={diff.max():.8f}  mean={diff.mean():.8f}")

# ------------------------------------------------------------------ #
# 4.  Save results
# ------------------------------------------------------------------ #
out_path = os.path.join(RESULTS_BASE, "comparison_dw_nominate.csv")
pivot.to_csv(out_path)
print(f"\nSaved: {out_path}")

corr_summary = pd.DataFrame({
    "seed": list(corr_per_seed.keys()),
    "pearson_r": list(corr_per_seed.values()),
})
corr_summary.loc[len(corr_summary)] = ["mean", np.mean(corr_values)]
corr_summary.loc[len(corr_summary)] = ["std", np.std(corr_values)]
corr_path = os.path.join(RESULTS_BASE, "correlation_summary.csv")
corr_summary.to_csv(corr_path, index=False)
print(f"Saved: {corr_path}")

print("\nDone.")
