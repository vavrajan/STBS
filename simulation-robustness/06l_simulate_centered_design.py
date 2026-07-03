#!/usr/bin/env python3
"""
06l_simulate_centered_design.py
===============================
Variant of 06d that constrains per-topic mean(ideal_sim) ≈ 0 to
remove the location-identifiability bias that plagued §1A and §1L.

Two changes vs §1A:
1. c1_uniform_all = +0.30 (was +0.70) — smaller systematic shift
2. Sign-balancing per topic: for each topic k, after drawing
   magnitudes |ι_kj| ~ U(0.5, 1.5) for c2/c3/c4 active entries,
   the SIGNS are chosen (brute force 2^n_active) such that
       bar{X} · ι_k = X_mean.dot(ι_k) ≈ 0
   This ensures the per-topic ideal mean is close to zero.

Output: data_simulation/simdata_centered_design/

VOCABULARY REDUCTION: for each topic we keep only the top-N words by
E[beta_kv]; the new vocabulary is the union across topics (V_new ~ 2000
for N=100 → 2.5x reduction, still covering the full topic signatures).
This makes the fit ~2-3x faster and reduces memory pressure.

Default base-params-dir is
    originalPolAn_results/fits/TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25/params

Dimensions (inherited from the base fit):
    N =   99 authors      D = 14,672 docs
    K =   25 topics       V_full = 5,031 vocab, V_new ~ 2,000

Synthetic design (iota / X / ideal):
    L = 6 covariates, binary dummies on the 99 authors
        c0_zero          - no effect  (prevalence 0.50)
        c1_zero          - no effect  (prevalence 0.30)
        c2_uniform       - +0.70 on all 25 topics (prevalence 0.50)
        c3_topic4        - non-zero on 4 random topics (prevalence 0.25)
        c4_topic16       - non-zero on 16 random topics (prevalence 0.40)
        c5_topic25       - non-zero on all 25 topics  (prevalence 0.20)
    ideal_n_k = X_n . iota_k + epsilon,  epsilon ~ N(0, 0.1^2)
    y_dv ~ Poisson( sum_k  theta_dk * beta_kv * exp(eta_kv * ideal_{a(d),k}) )

Output: data_simulation/24_04_26_simdata/  (overwrites previous version)
    clean/                drop-in for STBS (with REDUCED vocabulary)
    ground_truth/         .npy files + R-friendly .csv for X, iota_sim,
                          ideal_sim, beta, eta
    warm_start_truth/     shape/rate for beta & theta on reduced vocab
"""

import os
import json
import shutil
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================================================================== #
# CONFIGURATION
# ================================================================== #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REVISION_BASE = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
STBS_CAVI_DIR = os.path.join(REVISION_BASE, "STBS_CAVI")
CLEAN_DIR = os.path.join(STBS_CAVI_DIR, "data", "hein-daily", "clean")

DEFAULT_BASE = os.path.join(
    REVISION_BASE, "originalPolAn_results", "fits",
    "TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25", "params",
)

OUT_ROOT  = os.path.join(SCRIPT_DIR, "data_simulation", "simdata_centered_design")
CLEAN_OUT = os.path.join(OUT_ROOT, "clean")
GT_DIR    = os.path.join(OUT_ROOT, "ground_truth")
FIG_DIR   = os.path.join(GT_DIR, "figs")
WS_DIR    = os.path.join(OUT_ROOT, "warm_start_truth")
R_DIR     = os.path.join(GT_DIR, "R_exports")   # human-readable CSV for R

for d in (CLEAN_OUT, GT_DIR, FIG_DIR, WS_DIR, R_DIR):
    os.makedirs(d, exist_ok=True)

MASTER_SEED = 20260422
POISSON_SEED = 20260423

TOP_N_PER_TOPIC = 100       # vocabulary reduction: top-N words per topic

NUM_TOPICS = 25
NUM_COVS = 5
# c_0 (the high-prevalence no-effect covariate) was removed because it
# attracted spurious signals through its large variance. The remaining
# five cover: one zero, one uniform, three topic-varying.
COV_LABELS = ["c0_zero",
              "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]
COV_PREVALENCE = [0.30, 0.50, 0.25, 0.40, 0.20]

IOTA_UNIFORM_VALUE = 0.30     # was 0.70 in §1A — smaller systematic shift
IOTA_LOW = 0.50
IOTA_HIGH = 1.50
IDEAL_NOISE_SIGMA = 0.10


# ================================================================== #
# Load real-data nuisance parameters from the base fit
# ================================================================== #

def _load_first(dirpath, bases):
    for base in bases:
        for ext in ("_final.npy", ".npy"):
            p = os.path.join(dirpath, f"{base}{ext}")
            if os.path.exists(p):
                return np.load(p)
        p = os.path.join(dirpath, f"{base}.csv")
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0).to_numpy()
    raise FileNotFoundError(f"none of {bases} found in {dirpath}")


def load_base(base_dir):
    print("=" * 60)
    print(f"  Loading base nuisance parameters from:")
    print(f"  {base_dir}")
    print("=" * 60)

    bshp = _load_first(base_dir, ["beta_shape", "beta_shp"])
    brte = _load_first(base_dir, ["beta_rate",  "beta_rte"])
    tshp = _load_first(base_dir, ["theta_shape", "theta_shp"])
    trte = _load_first(base_dir, ["theta_rate",  "theta_rte"])
    eta  = _load_first(base_dir, ["eta_location", "eta_loc"])

    beta  = (bshp / brte).astype(np.float32)
    theta = (tshp / trte).astype(np.float32)
    eta   = eta.astype(np.float32)

    ai_path = os.path.join(base_dir, "all_author_indices.npy")
    if os.path.exists(ai_path):
        author_indices = np.load(ai_path)
    else:
        author_indices = np.load(os.path.join(CLEAN_DIR, "author_indices114.npy"))
    author_indices = author_indices.astype(np.int32)

    K, V_full = beta.shape
    D, _ = theta.shape
    N = int(author_indices.max()) + 1
    print(f"  beta :  {beta.shape}   row-sum mean = {beta.sum(axis=1).mean():.2f}")
    print(f"  theta:  {theta.shape}   mean = {theta.mean():.4e}")
    print(f"  eta  :  {eta.shape}    std = {eta.std():.4f}")
    print(f"  author_indices: {author_indices.shape}   unique = {N}")

    return dict(beta=beta, theta=theta, eta=eta,
                author_indices=author_indices,
                K=K, V=V_full, D=D, N=N,
                bshp=bshp.astype(np.float32),
                brte=brte.astype(np.float32),
                tshp=tshp.astype(np.float32),
                trte=trte.astype(np.float32))


# ================================================================== #
# VOCABULARY REDUCTION: top-N words per topic
# ================================================================== #

def reduce_vocab_top_n(base, top_n, vocab_path):
    print("\n" + "=" * 60)
    print(f"  VOCABULARY REDUCTION: top-{top_n} words per topic")
    print("=" * 60)

    beta = base["beta"]
    K, V_full = beta.shape

    top_sets = []
    for k in range(K):
        top_idx = np.argsort(beta[k])[::-1][:top_n]
        top_sets.append(set(int(i) for i in top_idx))

    kept_indices = np.array(sorted(set().union(*top_sets)), dtype=np.int64)
    V_new = len(kept_indices)
    print(f"  total kept entries (with replacement) : {K * top_n}")
    print(f"  UNIQUE kept words (union)              : {V_new}")
    print(f"  reduction factor vs. V_full={V_full}   : {V_full / V_new:.2f}x smaller")

    counter = Counter()
    for s in top_sets:
        for v in s:
            counter[v] += 1
    overlap_dist = Counter(counter.values())
    print(f"\n  Overlap across topics (how many topics each kept word occurs in):")
    for n_t, cnt in sorted(overlap_dist.items()):
        print(f"    in {n_t:>2d}/25 topics: {cnt:>4d} words")

    # Subset beta, eta, beta shape/rate
    base["beta"] = beta[:, kept_indices].copy()
    base["eta"]  = base["eta"][:, kept_indices].copy()
    base["bshp"] = base["bshp"][:, kept_indices].copy()
    base["brte"] = base["brte"][:, kept_indices].copy()
    base["V"]    = V_new
    base["kept_indices"] = kept_indices
    base["overlap_dist"] = dict(overlap_dist)

    with open(vocab_path) as fh:
        vocab_all = [ln.strip() for ln in fh if ln.strip()]
    base["vocab_new"] = [vocab_all[i] for i in kept_indices]

    print(f"\n  beta  -> {base['beta'].shape}")
    print(f"  eta   -> {base['eta'].shape}")
    return base


# ================================================================== #
# Covariate matrix X (binary dummies, independent)
# ================================================================== #

def sample_X(N, rng):
    X = np.zeros((N, NUM_COVS), dtype=np.float32)
    for j, p in enumerate(COV_PREVALENCE):
        X[:, j] = rng.binomial(1, p, N).astype(np.float32)
    return X


# ================================================================== #
# iota_sim
# ================================================================== #

def make_iota(K, rng, X_mean=None):
    """Build iota with per-topic mean(ideal) ≈ 0 by sign-balancing.

    For each topic k:
      1. Draw magnitudes |ι_kj| ~ U(IOTA_LOW, IOTA_HIGH) for active j
      2. Brute-force enumerate 2^n_active sign combinations
      3. Pick the combination that minimises |bar{X} . ι_k|
    """
    if X_mean is None:
        raise ValueError("06l requires X_mean for per-topic sign-balancing")

    iota = np.zeros((K, NUM_COVS), dtype=np.float32)
    pattern = np.zeros((K, NUM_COVS), dtype=np.int32)
    active = {}

    # c0_zero (index 0): inactive everywhere
    active["c0_zero"] = np.array([], dtype=np.int32)

    # c1_uniform_all (index 1): constant +IOTA_UNIFORM_VALUE on all 25 topics
    t_uni = np.arange(K)
    iota[t_uni, 1] = IOTA_UNIFORM_VALUE
    pattern[t_uni, 1] = 1
    active["c1_uniform_all"] = t_uni.copy()

    # c2_topic4 (index 2): 4 random topics
    t_c2 = np.sort(rng.choice(K, 4, replace=False))
    pattern[t_c2, 2] = 1
    active["c2_topic4"] = t_c2

    # c3_topic16 (index 3): 16 random topics
    t_c3 = np.sort(rng.choice(K, 16, replace=False))
    pattern[t_c3, 3] = 1
    active["c3_topic16"] = t_c3

    # c4_topic25 (index 4): all 25 topics
    t_c4 = np.arange(K)
    pattern[t_c4, 4] = 1
    active["c4_topic25"] = t_c4.copy()

    # For each topic: draw magnitudes for active topic-varying covariates,
    # then pick signs to minimise per-topic mean |bar{X} . iota_k|.
    print(f"\n  SIGN-BALANCING per topic to minimise |bar(X) . iota_k|")
    print(f"    bar(X)            = {[round(x, 3) for x in X_mean]}")
    print(f"    c1 contribution   = bar(X)[1] * IOTA_UNIFORM_VALUE = {X_mean[1] * IOTA_UNIFORM_VALUE:+.4f}")
    abs_means = []
    for k in range(K):
        active_js = [j for j in (2, 3, 4) if pattern[k, j] == 1]
        n_active = len(active_js)
        magnitudes = rng.uniform(IOTA_LOW, IOTA_HIGH, n_active)
        c1_contribution = X_mean[1] * IOTA_UNIFORM_VALUE
        # try all 2^n_active sign combos
        best_abs = float('inf')
        best_signs = None
        for combo in range(1 << n_active):
            signs = np.array([1.0 if ((combo >> i) & 1) else -1.0
                               for i in range(n_active)])
            mean_contrib = c1_contribution + sum(
                X_mean[active_js[i]] * magnitudes[i] * signs[i]
                for i in range(n_active)
            )
            if abs(mean_contrib) < best_abs:
                best_abs = abs(mean_contrib)
                best_signs = signs.copy()
        # apply best signs
        for i, j in enumerate(active_js):
            iota[k, j] = magnitudes[i] * best_signs[i]
        abs_means.append(best_abs)
    print(f"    achieved |bar(X) . iota_k|: mean = {np.mean(abs_means):.4f}, "
          f"max = {np.max(abs_means):.4f}")

    return iota, pattern, active


# ================================================================== #
# DTM sampling (chunked einsum)
# ================================================================== #

def sample_dtm(theta, beta, eta, ideal, author_indices, rng):
    N = ideal.shape[0]
    D, K = theta.shape
    _, V = beta.shape

    print(f"\n  Pre-computing exp(eta * ideal) for N={N} authors ...")
    eta_effect = np.exp(eta[None, :, :] * ideal[:, :, None]).astype(np.float32)
    print(f"  eta_effect: {eta_effect.shape}   size = {eta_effect.nbytes/1e6:.0f} MB")
    print(f"  eta_effect range = [{eta_effect.min():.3f}, {eta_effect.max():.3f}]")

    CHUNK = 512
    rows, cols, vals = [], [], []
    total_rate = 0.0
    for start in range(0, D, CHUNK):
        stop = min(start + CHUNK, D)
        a_idx = author_indices[start:stop]
        eta_chunk = eta_effect[a_idx]
        theta_chunk = theta[start:stop]
        rate = np.einsum("bk,kv,bkv->bv",
                         theta_chunk, beta, eta_chunk).astype(np.float32)
        total_rate += float(rate.sum())
        samp = rng.poisson(rate).astype(np.float32)
        nz_r, nz_c = np.nonzero(samp)
        if nz_r.size == 0:
            continue
        rows.append(nz_r + start); cols.append(nz_c)
        vals.append(samp[nz_r, nz_c])

    rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    vals = np.concatenate(vals) if vals else np.array([], dtype=np.float32)
    counts = sparse.csr_matrix((vals, (rows, cols)), shape=(D, V))
    print(f"  DTM: {counts.shape}  nnz = {counts.nnz}  total tokens = {int(counts.sum()):,}")
    print(f"  mean Poisson rate per cell = {total_rate / (D * V):.4g}")
    return counts


# ================================================================== #
# Save — npy (for Python) + R-friendly CSV
# ================================================================== #

def save_clean(counts, X, author_indices, vocab_new):
    sparse.save_npz(os.path.join(CLEAN_OUT, "counts114.npz"), counts)
    np.save(os.path.join(CLEAN_OUT, "X_override.npy"), X.astype(np.float32))
    np.save(os.path.join(CLEAN_OUT, "author_indices114.npy"), author_indices)
    np.save(os.path.join(CLEAN_OUT, "speech_id_indices114.npy"),
            np.arange(counts.shape[0], dtype=np.int32))
    with open(os.path.join(CLEAN_OUT, "vocabulary114.txt"), "w") as fh:
        for w in vocab_new:
            fh.write(w + "\n")
    for f in ("author_map114.txt", "author_info114.csv",
              "author_detailed_info114.csv",
              "author_detailed_info_with_religion114.csv"):
        src = os.path.join(CLEAN_DIR, f)
        dst = os.path.join(CLEAN_OUT, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print(f"\n  wrote clean/:  counts114.npz ({counts.shape}), "
          f"vocabulary114.txt (V_new={len(vocab_new)}), X_override.npy")


def save_warm_start(base):
    np.save(os.path.join(WS_DIR, "beta_shape_final.npy"),
            base["bshp"].astype(np.float32))
    np.save(os.path.join(WS_DIR, "beta_rate_final.npy"),
            base["brte"].astype(np.float32))
    np.save(os.path.join(WS_DIR, "theta_shape_final.npy"),
            base["tshp"].astype(np.float32))
    np.save(os.path.join(WS_DIR, "theta_rate_final.npy"),
            base["trte"].astype(np.float32))
    print(f"  wrote warm_start_truth/: reduced beta_shape/rate + theta_shape/rate")


def save_gt(iota, beta, eta, theta, ideal, X, pattern, author_indices,
            active, base_dir, base_dims, kept_indices, overlap_dist):
    # ---- .npy (Python-friendly, binary) ----
    np.save(os.path.join(GT_DIR, "iota_sim.npy"), iota)
    np.save(os.path.join(GT_DIR, "beta.npy"), beta)
    np.save(os.path.join(GT_DIR, "eta.npy"), eta)
    np.save(os.path.join(GT_DIR, "theta.npy"), theta)
    np.save(os.path.join(GT_DIR, "ideal_sim.npy"), ideal)
    np.save(os.path.join(GT_DIR, "X.npy"), X)
    np.save(os.path.join(GT_DIR, "effect_pattern.npy"), pattern)
    np.save(os.path.join(GT_DIR, "author_indices.npy"), author_indices)
    np.save(os.path.join(GT_DIR, "kept_word_indices.npy"), kept_indices)

    meta = dict(
        master_seed=MASTER_SEED,
        poisson_seed=POISSON_SEED,
        num_topics=int(base_dims["K"]),
        num_covariates=int(NUM_COVS),
        num_authors=int(base_dims["N"]),
        num_documents=int(base_dims["D"]),
        vocab_size_full=5031,
        vocab_size_reduced=int(base_dims["V"]),
        top_n_per_topic=int(TOP_N_PER_TOPIC),
        overlap_distribution={int(k): int(v) for k, v in overlap_dist.items()},
        iota_uniform_value=IOTA_UNIFORM_VALUE,
        iota_low=IOTA_LOW, iota_high=IOTA_HIGH,
        ideal_noise_sigma=IDEAL_NOISE_SIGMA,
        base_params_dir=base_dir,
        covariate_labels=COV_LABELS,
        covariate_prevalence=COV_PREVALENCE,
        active_topics={k: [int(x) for x in v] for k, v in active.items()},
    )
    with open(os.path.join(GT_DIR, "simulation_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def save_R_exports(iota, X, ideal, beta, eta, vocab_new, author_indices):
    """CSV exports for R (readable via read.csv)."""
    K = iota.shape[0]
    N, L = X.shape
    _, V = beta.shape

    # X (N, L)  -- rows = authors, cols = c0..c5
    pd.DataFrame(
        X, columns=COV_LABELS,
        index=[f"author_{a:03d}" for a in range(N)],
    ).to_csv(os.path.join(R_DIR, "X.csv"), index_label="author")

    # iota_sim (K, L) -- rows = topics, cols = c0..c5
    pd.DataFrame(
        iota, columns=COV_LABELS,
        index=[f"topic_{k:02d}" for k in range(K)],
    ).to_csv(os.path.join(R_DIR, "iota_sim.csv"), index_label="topic")

    # ideal_sim (N, K) -- rows = authors, cols = topics
    pd.DataFrame(
        ideal,
        columns=[f"topic_{k:02d}" for k in range(K)],
        index=[f"author_{a:03d}" for a in range(N)],
    ).to_csv(os.path.join(R_DIR, "ideal_sim.csv"), index_label="author")

    # beta  (K, V) -- rows = topics, cols = word strings
    # Use reduced vocabulary as column names; safe strings with underscores
    safe_cols = [w.replace(" ", "_") for w in vocab_new]
    pd.DataFrame(
        beta, columns=safe_cols,
        index=[f"topic_{k:02d}" for k in range(K)],
    ).to_csv(os.path.join(R_DIR, "beta.csv"), index_label="topic")

    # eta  (K, V) -- same layout as beta
    pd.DataFrame(
        eta, columns=safe_cols,
        index=[f"topic_{k:02d}" for k in range(K)],
    ).to_csv(os.path.join(R_DIR, "eta.csv"), index_label="topic")

    # Author -> document index (long format)
    pd.DataFrame(dict(
        doc_id=np.arange(len(author_indices)),
        author=author_indices,
    )).to_csv(os.path.join(R_DIR, "author_indices.csv"), index=False)

    # Vocabulary table with original index
    pd.DataFrame(dict(
        new_idx=np.arange(V),
        original_idx=np.load(os.path.join(GT_DIR, "kept_word_indices.npy")),
        term=vocab_new,
    )).to_csv(os.path.join(R_DIR, "vocabulary.csv"), index=False)

    print(f"  wrote R_exports/: X.csv, iota_sim.csv, ideal_sim.csv, "
          f"beta.csv, eta.csv, author_indices.csv, vocabulary.csv")


# ================================================================== #
# Plots
# ================================================================== #

def plot_diagnostics(ideal, iota, pattern, counts):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ideal.ravel(), bins=60, color="tab:orange",
            edgecolor="black", alpha=0.85)
    ax.axvline(0, ls="--", color="black", lw=0.5)
    ax.set_title(f"Simulated ideal points (N={ideal.shape[0]}, K={ideal.shape[1]})\n"
                 f"std={ideal.std():.2f}, range=[{ideal.min():+.2f},{ideal.max():+.2f}]")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ideal_points_hist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 7))
    vmax = np.abs(iota).max()
    im = ax.imshow(iota, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(NUM_COVS))
    ax.set_xticklabels(COV_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(iota.shape[0]))
    ax.set_yticklabels([f"k={k}" for k in range(iota.shape[0])], fontsize=7)
    ax.set_title("Ground-truth iota (K x L)")
    plt.colorbar(im, ax=ax, label="iota")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "iota_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    word_freq = np.asarray(counts.sum(axis=0)).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(doc_len, bins=60, color="steelblue", edgecolor="black")
    axes[0].set_title(f"Doc length (mean {doc_len.mean():.1f})")
    axes[0].set_xlabel("tokens per document")
    axes[1].hist(np.log10(word_freq + 1), bins=60, color="tab:green", edgecolor="black")
    axes[1].set_title(f"Word frequencies (log10)")
    axes[1].set_xlabel(r"$\log_{10}(1 + \text{count})$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dtm_stats.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================== #
# MAIN
# ================================================================== #

def main():
    global TOP_N_PER_TOPIC
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-params-dir", default=DEFAULT_BASE,
                    help="Directory of the base fit providing beta, theta, eta")
    ap.add_argument("--top-n", type=int, default=TOP_N_PER_TOPIC,
                    help="Top-N words per topic to keep in the reduced vocabulary")
    args = ap.parse_args()
    TOP_N_PER_TOPIC = args.top_n

    rng = np.random.default_rng(MASTER_SEED)
    rng_pois = np.random.default_rng(POISSON_SEED)

    base = load_base(args.base_params_dir)

    # Vocabulary reduction
    vocab_path = os.path.join(CLEAN_DIR, "vocabulary114.txt")
    base = reduce_vocab_top_n(base, args.top_n, vocab_path)

    K, N = base["K"], base["N"]

    # X
    X = sample_X(N, rng)
    print(f"\n  X: {X.shape}")
    for j, lab in enumerate(COV_LABELS):
        print(f"    {lab:<18s}  prev = {X[:, j].mean():.2f}  ({int(X[:, j].sum())}/{N})")

    # iota
    iota, pattern, active = make_iota(K, rng, X_mean=X.mean(axis=0))
    print("\n  iota design:")
    for j, lab in enumerate(COV_LABELS):
        n_act = int(pattern[:, j].sum())
        if n_act > 0:
            v = iota[pattern[:, j] == 1, j]
            print(f"    {lab:<18s}  active {n_act}/{K}  |iota| mean {np.abs(v).mean():.3f}"
                  f"  range [{v.min():+.3f}, {v.max():+.3f}]")
        else:
            print(f"    {lab:<18s}  all zero")

    # ideal
    mean_part = X @ iota.T
    noise = rng.normal(0.0, IDEAL_NOISE_SIGMA,
                       size=mean_part.shape).astype(np.float32)
    ideal = (mean_part + noise).astype(np.float32)
    print(f"\n  ideal: {ideal.shape}  range [{ideal.min():+.3f}, {ideal.max():+.3f}]"
          f"  std {ideal.std():.3f}  (regression std {mean_part.std():.3f},"
          f" noise sigma {IDEAL_NOISE_SIGMA})")

    # DTM (on REDUCED vocab)
    counts = sample_dtm(base["theta"], base["beta"], base["eta"],
                        ideal, base["author_indices"], rng_pois)

    # Save
    print("\n  Saving ...")
    save_clean(counts, X, base["author_indices"], base["vocab_new"])
    save_gt(iota, base["beta"], base["eta"], base["theta"],
            ideal, X, pattern, base["author_indices"], active,
            args.base_params_dir,
            dict(K=K, V=base["V"], D=base["D"], N=N),
            base["kept_indices"], base["overlap_dist"])
    save_warm_start(base)
    save_R_exports(iota, X, ideal,
                   base["beta"], base["eta"],
                   base["vocab_new"], base["author_indices"])

    plot_diagnostics(ideal, iota, pattern, counts)

    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  OUTPUT: {OUT_ROOT}")
    print(f"  clean/:           STBS drop-in (V_new={base['V']}, D={base['D']}, N={N})")
    print(f"  ground_truth/:    iota_sim.npy, ideal_sim.npy, beta/eta.npy, "
          f"theta.npy, X.npy, pattern, meta, kept_word_indices.npy")
    print(f"  ground_truth/R_exports/: X.csv, iota_sim.csv, ideal_sim.csv, "
          f"beta.csv, eta.csv, author_indices.csv, vocabulary.csv")
    print(f"  warm_start_truth/: reduced beta/theta shape+rate")
    print("=" * 60)


if __name__ == "__main__":
    main()
