#!/usr/bin/env python3
"""
06n_simulate_dgp_family.py
==========================
Build a 20-fold Monte Carlo over the DGP family of the centered
simulation. In contrast to 06m (which fixes the truth and only re-draws
Poisson counts), this script re-draws the entire generative truth per
replicate:

    - X is drawn fresh per replicate (same prevalences, same N=99)
    - Active-topic sets for c2/c3/c4 are drawn fresh per replicate
    - iota magnitudes and signs are drawn fresh per replicate
      (same uniform[0.5, 1.5] + sign-balancing per topic)
    - ideal = X . iota + epsilon with fresh epsilon
    - counts ~ Poisson with fresh seed

Only the topic-content nuisances (beta, eta, theta from the original
PolAn fit) and the model dimensions (N, D, K, V, L, prevalences,
c1_uniform_value=0.30) are held constant across replicates.

Output: data_simulation/sim_dgp_NN/  (NN = 01..20)
    clean/                drop-in for STBS, fully self-contained
    ground_truth/         X, iota_sim, ideal_sim, beta, eta, ...
    warm_start_truth/     symlinks to simdata_centered_design (shared
                          beta/theta Gamma posteriors)
    sim_meta.json         {rep_idx, master_seed, poisson_seed, ...}

Run:
    ./STBS_CAVI/venv_gpu/bin/python3 06n_simulate_dgp_family.py
        [--n-sims 20] [--seed-base 20260700]
"""
import os, json, shutil, argparse
from collections import Counter
import numpy as np
from scipy import sparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REVISION_BASE = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
STBS_CAVI_DIR = os.path.join(REVISION_BASE, "STBS_CAVI")
CLEAN_DIR_REAL = os.path.join(STBS_CAVI_DIR, "data", "hein-daily", "clean")

DEFAULT_BASE = os.path.join(
    REVISION_BASE, "originalPolAn_results", "fits",
    "TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25", "params",
)

SIM_ROOT = os.path.join(SCRIPT_DIR, "data_simulation")
SHARED_WS = os.path.join(SIM_ROOT, "simdata_centered_design", "warm_start_truth")
SHARED_KEPT = os.path.join(SIM_ROOT, "simdata_centered_design", "ground_truth", "kept_word_indices.npy")

# Design parameters — SAME as 06l_simulate_centered_design.py
TOP_N_PER_TOPIC = 100
NUM_TOPICS = 25
NUM_COVS = 5
COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]
COV_PREVALENCE = [0.30, 0.50, 0.25, 0.40, 0.20]
IOTA_UNIFORM_VALUE = 0.30
IOTA_LOW = 0.50
IOTA_HIGH = 1.50
IDEAL_NOISE_SIGMA = 0.10


def _load_first(dirpath, bases):
    """Try multiple base-names and extensions (npy/csv); raise if none found."""
    import pandas as pd
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
    bshp = _load_first(base_dir, ["beta_shape", "beta_shp"])
    brte = _load_first(base_dir, ["beta_rate", "beta_rte"])
    tshp = _load_first(base_dir, ["theta_shape", "theta_shp"])
    trte = _load_first(base_dir, ["theta_rate", "theta_rte"])
    eta = _load_first(base_dir, ["eta_location", "eta_loc"])
    beta = (bshp / brte).astype(np.float32)
    theta = (tshp / trte).astype(np.float32)
    eta = eta.astype(np.float32)
    ai_path = os.path.join(base_dir, "all_author_indices.npy")
    if os.path.exists(ai_path):
        ai = np.load(ai_path)
    else:
        ai = np.load(os.path.join(CLEAN_DIR_REAL, "author_indices114.npy"))
    return beta, eta, theta, ai.astype(np.int32)


def make_X(N, prev, rng):
    X = np.zeros((N, NUM_COVS), dtype=np.float32)
    for j, p in enumerate(prev):
        X[:, j] = rng.binomial(1, p, N).astype(np.float32)
    return X


def make_iota_centered(K, X, rng):
    """Reproduces 06l_simulate_centered_design.make_iota with brute-force
    sign-balancing per topic to enforce X̄·ι_k ≈ 0. The number of active
    topics per covariate is also drawn (centred at the nominal value) so
    each replicate samples a slightly different sparsity pattern within
    the same DGP family.
    """
    X_mean = X.mean(axis=0)
    iota = np.zeros((K, NUM_COVS), dtype=np.float32)
    pattern = np.zeros((K, NUM_COVS), dtype=np.int32)

    # c1 uniform constant
    iota[:, 1] = IOTA_UNIFORM_VALUE
    pattern[:, 1] = 1
    c1_contribution = X_mean[1] * IOTA_UNIFORM_VALUE

    # Variable n_active per replicate (sampled from small ranges
    # around the nominal 4 / 16 / 25):
    #   c2 ∈ {3, 4, 5}    nominal 4
    #   c3 ∈ {14..18}     nominal 16
    #   c4 ∈ {23, 24, 25} nominal 25
    n_c2 = int(rng.choice([3, 4, 5]))
    n_c3 = int(rng.choice([14, 15, 16, 17, 18]))
    n_c4 = int(rng.choice([23, 24, 25]))

    t_c2 = np.sort(rng.choice(K, n_c2, replace=False))
    pattern[t_c2, 2] = 1
    t_c3 = np.sort(rng.choice(K, n_c3, replace=False))
    pattern[t_c3, 3] = 1
    t_c4 = np.sort(rng.choice(K, n_c4, replace=False))
    pattern[t_c4, 4] = 1

    n_active_per_cov = {"c2_topic4": n_c2, "c3_topic16": n_c3,
                         "c4_topic25": n_c4}

    # per-topic magnitudes + sign-balanced signs
    for k in range(K):
        active_js = [j for j in (2, 3, 4) if pattern[k, j] == 1]
        n = len(active_js)
        mags = rng.uniform(IOTA_LOW, IOTA_HIGH, n)
        best_abs = np.inf
        best_signs = None
        for combo in range(1 << n):
            signs = np.array([1.0 if ((combo >> i) & 1) else -1.0
                              for i in range(n)])
            mc = c1_contribution + sum(
                X_mean[active_js[i]] * mags[i] * signs[i] for i in range(n)
            )
            if abs(mc) < best_abs:
                best_abs = abs(mc); best_signs = signs.copy()
        for i, j in enumerate(active_js):
            iota[k, j] = mags[i] * best_signs[i]

    active = {
        "c0_zero": np.array([], dtype=np.int32),
        "c1_uniform_all": np.arange(K),
        "c2_topic4": t_c2,
        "c3_topic16": t_c3,
        "c4_topic25": t_c4,
    }
    return iota, pattern, active, n_active_per_cov


def reduce_vocab_top_n(beta, eta, top_n):
    """Top-N words per topic, union → reduced vocab (deterministic, depends
    only on the shared beta from PolAn)."""
    K, V_full = beta.shape
    top_sets = [set(int(i) for i in np.argsort(beta[k])[::-1][:top_n])
                for k in range(K)]
    kept = np.array(sorted(set().union(*top_sets)), dtype=np.int64)
    return beta[:, kept], eta[:, kept], kept


def sample_dtm(theta, beta, eta, ideal, author_indices, rng):
    N = ideal.shape[0]
    D, K = theta.shape
    _, V = beta.shape
    eta_effect = np.exp(eta[None, :, :] * ideal[:, :, None]).astype(np.float32)
    CHUNK = 512
    rows, cols, vals = [], [], []
    for start in range(0, D, CHUNK):
        stop = min(start + CHUNK, D)
        a_idx = author_indices[start:stop]
        rate = np.einsum("bk,kv,bkv->bv", theta[start:stop],
                          beta, eta_effect[a_idx]).astype(np.float32)
        samp = rng.poisson(rate).astype(np.float32)
        nz_r, nz_c = np.nonzero(samp)
        if nz_r.size == 0:
            continue
        rows.append(nz_r + start); cols.append(nz_c)
        vals.append(samp[nz_r, nz_c])
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(D, V))


def build_replicate(rep_idx, master_seed, poisson_seed,
                    beta_full, eta_full, theta, author_indices,
                    overwrite=False):
    out_dir = os.path.join(SIM_ROOT, f"sim_dgp_{rep_idx:02d}")
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"{out_dir} already exists")
    clean = os.path.join(out_dir, "clean")
    gt = os.path.join(out_dir, "ground_truth")
    ws = os.path.join(out_dir, "warm_start_truth")
    for d in (clean, gt, ws):
        os.makedirs(d, exist_ok=True)

    # Vocab reduction (deterministic — same kept words for all reps)
    beta_red, eta_red, kept = reduce_vocab_top_n(beta_full, eta_full,
                                                  TOP_N_PER_TOPIC)
    V_new = beta_red.shape[1]

    rng = np.random.default_rng(master_seed)
    N = int(author_indices.max()) + 1
    K = NUM_TOPICS

    X = make_X(N, COV_PREVALENCE, rng)
    iota, pattern, active, n_active = make_iota_centered(K, X, rng)

    # Ideal points = X iota + epsilon, with sign-balanced iota (sample mean ≈ 0)
    ideal_clean = X @ iota.T
    eps = rng.normal(0, IDEAL_NOISE_SIGMA, ideal_clean.shape).astype(np.float32)
    ideal = (ideal_clean + eps).astype(np.float32)

    # Sample DTM
    rng_pois = np.random.default_rng(poisson_seed)
    counts = sample_dtm(theta, beta_red, eta_red, ideal, author_indices,
                         rng_pois)

    # Save GT
    np.save(os.path.join(gt, "iota_sim.npy"), iota)
    np.save(os.path.join(gt, "ideal_sim.npy"), ideal)
    np.save(os.path.join(gt, "X.npy"), X)
    np.save(os.path.join(gt, "beta.npy"), beta_red)
    np.save(os.path.join(gt, "eta.npy"), eta_red)
    np.save(os.path.join(gt, "theta.npy"), theta)
    np.save(os.path.join(gt, "author_indices.npy"), author_indices)
    np.save(os.path.join(gt, "effect_pattern.npy"), pattern)
    np.save(os.path.join(gt, "kept_word_indices.npy"), kept)
    meta = dict(
        rep_idx=rep_idx, master_seed=int(master_seed),
        poisson_seed=int(poisson_seed),
        num_topics=int(K), num_covariates=int(NUM_COVS),
        num_authors=int(N),
        num_documents=int(theta.shape[0]),
        vocab_size_full=int(beta_full.shape[1]),
        vocab_size_reduced=int(V_new),
        iota_uniform_value=IOTA_UNIFORM_VALUE,
        iota_low=IOTA_LOW, iota_high=IOTA_HIGH,
        ideal_noise_sigma=IDEAL_NOISE_SIGMA,
        covariate_labels=COV_LABELS,
        covariate_prevalence=COV_PREVALENCE,
        active_topics={k: [int(x) for x in v] for k, v in active.items()},
        n_active_per_cov=n_active,
        notes=("Replicate of centered_design DGP family: fresh X, iota, "
               "ideal, counts per replicate. n_active per covariate also "
               "varies (c2 in {3,4,5}, c3 in {14..18}, c4 in {23,24,25}). "
               "Shared β/η/θ from PolAn, shared warm-start posteriors "
               "via symlink."),
    )
    with open(os.path.join(gt, "simulation_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Save clean
    sparse.save_npz(os.path.join(clean, "counts114.npz"), counts)
    np.save(os.path.join(clean, "author_indices114.npy"), author_indices)
    np.save(os.path.join(clean, "X_override.npy"), X)
    # symlink shared static files
    for fname in ("vocabulary114.txt",
                  "speech_id_indices114.npy",
                  "author_info114.csv",
                  "author_map114.txt",
                  "author_detailed_info114.csv",
                  "author_detailed_info_with_religion114.csv"):
        src = os.path.join(SIM_ROOT, "simdata_centered_design", "clean", fname)
        if os.path.exists(src):
            dst = os.path.join(clean, fname)
            if os.path.lexists(dst):
                os.remove(dst)
            try:
                os.symlink(os.path.relpath(src, clean), dst)
            except OSError:
                shutil.copy2(src, dst)

    # warm-start: symlink to simdata_centered_design/warm_start_truth/
    for fname in ("beta_shape_final.npy", "beta_rate_final.npy",
                  "theta_shape_final.npy", "theta_rate_final.npy"):
        src = os.path.join(SHARED_WS, fname)
        dst = os.path.join(ws, fname)
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.symlink(os.path.relpath(src, ws), dst)
        except OSError:
            shutil.copy2(src, dst)

    return dict(rep_idx=rep_idx, master_seed=master_seed,
                poisson_seed=poisson_seed,
                counts_nnz=int(counts.nnz),
                counts_total=int(counts.sum()),
                X_prev=[round(float(X[:, j].mean()), 3)
                        for j in range(NUM_COVS)],
                ideal_mean_max=float(np.abs(ideal.mean(axis=0)).max()),
                n_c2=n_active["c2_topic4"],
                n_c3=n_active["c3_topic16"],
                n_c4=n_active["c4_topic25"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=20260700,
                    help="master_seed_r = seed_base + 10*r,  "
                         "poisson_seed_r = master_seed_r + 1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--base-params-dir", default=DEFAULT_BASE)
    args = ap.parse_args()

    print("Loading PolAn nuisance parameters ...")
    beta, eta, theta, ai = load_base(args.base_params_dir)
    print(f"  beta: {beta.shape}, eta: {eta.shape}, theta: {theta.shape}")

    rows = []
    print(f"\nBuilding {args.n_sims} DGP replicates from seed_base={args.seed_base}")
    for r in range(1, args.n_sims + 1):
        ms = args.seed_base + 10 * r
        ps = ms + 1
        print(f"\n  [sim_dgp_{r:02d}] master_seed={ms}, poisson_seed={ps}")
        meta = build_replicate(r, ms, ps, beta, eta, theta, ai,
                                overwrite=args.overwrite)
        rows.append(meta)
        print(f"    counts: nnz={meta['counts_nnz']:,}, total={meta['counts_total']:,}")
        print(f"    X prevalences: {meta['X_prev']}")
        print(f"    max |mean(ideal_k)|: {meta['ideal_mean_max']:.3f}")

    import pandas as pd
    idx = pd.DataFrame(rows)
    idx_path = os.path.join(SIM_ROOT, "dgp_replicate_index.csv")
    idx.to_csv(idx_path, index=False)
    print(f"\nIndex written: {idx_path}")
    print(idx[["rep_idx","master_seed","counts_total","ideal_mean_max",
                "n_c2","n_c3","n_c4"]].to_string(index=False))


if __name__ == "__main__":
    main()
