#!/usr/bin/env python3
"""
06m_simulate_centered_replicates.py
===================================
Build a 20-fold Monte Carlo replication study on top of
`simdata_centered_design/`.

The generative truth — beta, eta, theta, X, iota_sim, ideal_sim,
author_indices, vocabulary — is **fixed**. Only the Poisson sampling
varies, with an independent random seed per replicate. This isolates
the Poisson-noise contribution to STBS recovery variability.

Output: data_simulation/sim_NN/  (NN = 01..20)
    clean/                drop-in for STBS — same layout as the master
                          sim's clean/, only `counts114.npz` differs;
                          author_indices, vocabulary, X_override are
                          *symlinks* back to the master sim.
    sim_meta.json         {sim_idx, poisson_seed, counts_nnz,
                           counts_total, master_dir}

The per-replicate clean/ is only ~3-5 MB (counts.npz only); the full
GT and warm-start NPYs are shared with the master sim and are read
from there at fit time via:
    --data-dir       data_simulation/sim_NN/clean
    --x-override     data_simulation/simdata_centered_design/clean/X_override.npy
    --warm-start-dir data_simulation/simdata_centered_design/warm_start_truth

Run:
    ./STBS_CAVI/venv_gpu/bin/python3 06e_simulate_replicates.py
        [--n-sims 20] [--seed-base 20260601]
"""

import os
import json
import argparse
import shutil

import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(SCRIPT_DIR, "data_simulation", "simdata_centered_design")
GT_DIR     = os.path.join(MASTER_DIR, "ground_truth")
CLEAN_DIR  = os.path.join(MASTER_DIR, "clean")
SIM_ROOT   = os.path.join(SCRIPT_DIR, "data_simulation")


def sample_dtm(theta, beta, eta, ideal, author_indices, rng):
    """Re-implementation of 06d_simulate_realbeta.py's sample_dtm in a
    chunked-einsum form, returning a sparse CSR DTM."""
    N = ideal.shape[0]
    D, K = theta.shape
    _, V = beta.shape

    eta_effect = np.exp(eta[None, :, :] * ideal[:, :, None]).astype(np.float32)

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
    return counts, total_rate


def build_replicate(sim_idx, poisson_seed,
                    theta, beta, eta, ideal, author_indices,
                    vocab_lines, x_override_path, master_dir,
                    overwrite=False):
    out_dir = os.path.join(SIM_ROOT, f"sim_{sim_idx:02d}")
    clean_out = os.path.join(out_dir, "clean")
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"{out_dir} already exists; pass --overwrite to redo")
    os.makedirs(clean_out, exist_ok=True)

    rng = np.random.default_rng(poisson_seed)
    counts, total_rate = sample_dtm(theta, beta, eta, ideal,
                                    author_indices, rng)
    sparse.save_npz(os.path.join(clean_out, "counts114.npz"), counts)

    # Hard-copy author_indices and vocabulary (small) so each sim is
    # self-contained for the STBS data loader. Symlink X_override to
    # the master sim (~MBs of ID, no need to duplicate).
    np.save(os.path.join(clean_out, "author_indices114.npy"), author_indices)
    with open(os.path.join(clean_out, "vocabulary114.txt"), "w") as fh:
        fh.writelines(vocab_lines)

    # Optional: symlink X_override and any other fixed metadata files
    for fname in ("X_override.npy",
                  "speech_id_indices114.npy",
                  "author_info114.csv",
                  "author_map114.txt",
                  "author_detailed_info114.csv",
                  "author_detailed_info_with_religion114.csv"):
        src = os.path.join(CLEAN_DIR, fname)
        dst = os.path.join(clean_out, fname)
        if not os.path.exists(src):
            continue
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.symlink(os.path.relpath(src, clean_out), dst)
        except OSError:
            shutil.copy2(src, dst)

    meta = dict(
        sim_idx=sim_idx,
        poisson_seed=int(poisson_seed),
        counts_nnz=int(counts.nnz),
        counts_total=int(counts.sum()),
        master_dir=os.path.relpath(master_dir, out_dir),
        notes=("Replicate of simdata_centered_design: same generative truth, "
               "only Poisson seed differs. GT and warm_start live in "
               "../simdata_centered_design/."),
    )
    with open(os.path.join(out_dir, "sim_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=20260601,
                    help="Per-replicate seed = seed_base + sim_idx")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Load fixed truth
    print("=" * 60)
    print("  Loading shared GT from simdata_centered_design/ ...")
    print("=" * 60)
    theta = np.load(os.path.join(GT_DIR, "theta.npy"))           # (D, K)
    beta  = np.load(os.path.join(GT_DIR, "beta.npy"))            # (K, V)
    eta   = np.load(os.path.join(GT_DIR, "eta.npy"))             # (K, V)
    ideal = np.load(os.path.join(GT_DIR, "ideal_sim.npy"))       # (N, K)
    author_indices = np.load(os.path.join(GT_DIR, "author_indices.npy"))
    print(f"  theta: {theta.shape}   beta: {beta.shape}   eta: {eta.shape}")
    print(f"  ideal: {ideal.shape}   author_indices: {author_indices.shape}")

    with open(os.path.join(CLEAN_DIR, "vocabulary114.txt"), "r") as fh:
        vocab_lines = fh.readlines()
    print(f"  vocab : {len(vocab_lines)} lines")

    rows = []
    print(f"\n  Building {args.n_sims} replicates (seed_base={args.seed_base})")
    for sim_idx in range(1, args.n_sims + 1):
        seed = args.seed_base + sim_idx
        print(f"\n  [sim_{sim_idx:02d}] poisson_seed = {seed}")
        meta = build_replicate(
            sim_idx, seed, theta, beta, eta, ideal,
            author_indices, vocab_lines,
            x_override_path=os.path.join(CLEAN_DIR, "X_override.npy"),
            master_dir=MASTER_DIR,
            overwrite=args.overwrite,
        )
        rows.append(meta)
        print(f"    counts: nnz={meta['counts_nnz']}, "
              f"total tokens={meta['counts_total']:,}")

    # Index file
    idx = pd.DataFrame(rows)[["sim_idx", "poisson_seed",
                               "counts_nnz", "counts_total"]]
    idx_path = os.path.join(SIM_ROOT, "simulation_index.csv")
    idx.to_csv(idx_path, index=False)
    print(f"\n  Index written: {idx_path}")
    print(idx.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
