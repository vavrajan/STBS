#!/usr/bin/env python3
"""
13_scaling_STBS.py
==================
Empirical scalability exercise for STBS-CAVI.

We vary two axes:
  1. The number of documents D, by replicating or subsampling the
     real Hein-Daily 114th Senate document-term matrix along the
     row (document) axis. Replication preserves the existing
     author_indices vector (the original $A=99$ senators are
     re-used; only the count of documents-per-senator changes).
     Subsampling draws a random subset of documents without
     replacement.
  2. The number of topics K, taking the values requested by the
     user: K in {10, 20, 30}.

The vocabulary size $V$ and the number of authors $A$ are held
fixed at the real-corpus values ($V=5031$, $A=99$). Per the
theoretical analysis in the scalability section, these contribute
negligibly to per-epoch CAVI compute time when $A\ll D$.

For each $(D\text{-factor}, K)$ configuration the script
  (a) materialises a scaled `clean/` data folder under
      `data_simulation/scaling_Dx{factor}_K{K}/clean/`,
  (b) runs `01_estimate_STBS.py` for `--epochs` (default 200)
      with the requested K, writing parameters to
      `results_simulation/scaling_Dx{factor}_K{K}/`,
  (c) parses the run.log for per-epoch wall-clock times and ELBO.

The aggregate timings are saved to
  `results_simulation/scaling_summary.csv`
with one row per configuration. Re-running the script skips any
configuration whose final iota file already exists, so the run is
resumable.

Run:
  python3 13_scaling_STBS.py [--epochs 200] [--seed 314159]
                              [--D-factors 0.5 1.0 2.0 3.0]
                              [--K-values 10 20 30]
"""
import os, sys, argparse, subprocess, time, re, shutil
import numpy as np
import pandas as pd
from scipy import sparse

REPO = os.path.dirname(os.path.abspath(__file__))
STBS_CAVI = os.path.normpath(os.path.join(REPO, "..", "STBS_CAVI"))
PY = os.path.join(STBS_CAVI, "venv_gpu", "bin", "python3")
ORIG_CLEAN = os.path.join(STBS_CAVI, "data", "hein-daily", "clean")
SIM_BASE = os.path.join(REPO, "data_simulation")
RES_BASE = os.path.join(REPO, "results_simulation")


# ============================================================== #
def build_scaled_corpus(D_factor, out_dir, rng):
    """Replicate or subsample counts114.npz along the document axis.

      D_factor = 1.0  -> exact copy of the original corpus
      D_factor = n + f (with integer n and fractional f in [0,1))
                      -> n full copies concatenated, plus an extra
                         f-fraction random sample
      D_factor < 1.0  -> random subsample of the corresponding fraction
                         of documents (without replacement)

    Author indices are tiled accordingly; the $A=99$ author identities
    are preserved. Static files (vocabulary, author_info, ...) are
    symlinked from the original clean/ directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    counts = sparse.load_npz(os.path.join(ORIG_CLEAN, "counts114.npz"))
    ai = np.load(os.path.join(ORIG_CLEAN, "author_indices114.npy"))
    D_orig = counts.shape[0]

    if D_factor >= 1.0:
        n_reps = int(D_factor)
        frac = D_factor - n_reps
        parts = [counts] * n_reps
        ai_parts = [ai] * n_reps
        if frac > 1e-6:
            n_extra = int(round(frac * D_orig))
            idx = rng.choice(D_orig, size=n_extra, replace=False)
            parts.append(counts[idx])
            ai_parts.append(ai[idx])
        new_counts = sparse.vstack(parts).tocsr()
        new_ai = np.concatenate(ai_parts).astype(np.int32)
    else:
        n_keep = max(1, int(round(D_factor * D_orig)))
        idx = rng.choice(D_orig, size=n_keep, replace=False)
        idx.sort()
        new_counts = counts[idx].tocsr()
        new_ai = ai[idx].astype(np.int32)

    sparse.save_npz(os.path.join(out_dir, "counts114.npz"), new_counts)
    np.save(os.path.join(out_dir, "author_indices114.npy"), new_ai)

    # Symlink unchanged files. We also need speech_id_indices114.npy
    # if it exists; otherwise build_input_pipeline can derive ranges.
    for fname in ("vocabulary114.txt",
                  "author_info114.csv",
                  "author_map114.txt",
                  "author_detailed_info114.csv",
                  "author_detailed_info_with_religion114.csv"):
        src = os.path.join(ORIG_CLEAN, fname)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, fname)
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    return new_counts.shape


# ============================================================== #
EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+done\s+\|\s+ELBO:\s+([-+\d.eE]+)\s+\|\s+"
    r"\(([\d.]+)\s+sec/epoch\)"
)

def parse_epoch_times(run_log_path):
    """Read per-epoch (epoch_idx, elbo, sec_per_epoch) from the run.log."""
    epochs, elbos, secs = [], [], []
    if not os.path.exists(run_log_path):
        return np.array([]), np.array([]), np.array([])
    with open(run_log_path) as fh:
        for line in fh:
            m = EPOCH_RE.match(line)
            if m:
                epochs.append(int(m.group(1)))
                elbos.append(float(m.group(2)))
                secs.append(float(m.group(3)))
    return np.asarray(epochs), np.asarray(elbos), np.asarray(secs)


# ============================================================== #
def run_one(D_factor, K, epochs, seed, rng):
    """Build the scaled corpus (if needed), run STBS, parse the log."""
    tag = f"Dx{D_factor:.2f}_K{K}"
    data_dir = os.path.join(SIM_BASE, f"scaling_{tag}", "clean")
    out_dir = os.path.join(RES_BASE, f"scaling_{tag}")
    fit_done_marker = os.path.join(out_dir, "params",
                                    "iota_location_final.npy")

    if not os.path.exists(fit_done_marker):
        # Build scaled corpus if not present
        if not os.path.exists(os.path.join(data_dir, "counts114.npz")):
            D_shape = build_scaled_corpus(D_factor, data_dir, rng)
        else:
            counts = sparse.load_npz(os.path.join(data_dir,
                                                   "counts114.npz"))
            D_shape = counts.shape

        os.makedirs(out_dir, exist_ok=True)
        env = os.environ.copy()
        env["TF_USE_LEGACY_KERAS"] = "1"
        cmd = [PY, "-u", os.path.join(REPO, "01_estimate_STBS.py"),
               "--seed", str(seed),
               "--num-epochs", str(epochs),
               "--num-topics", str(K),
               "--data-dir", data_dir,
               "--output-dir", out_dir]
        log_path = os.path.join(out_dir, "run.log")
        t0 = time.time()
        with open(log_path, "w") as fh:
            ret = subprocess.run(cmd, stdout=fh, stderr=fh, env=env)
        wall = time.time() - t0
        if ret.returncode != 0:
            raise RuntimeError(f"{tag} fit failed (exit {ret.returncode}); "
                               f"see {log_path}")
        print(f"  [{tag}] D={D_shape[0]}, K={K}: total {wall:.0f}s")

    # Parse timings (whether we just ran or reusing prior run)
    eps, elbos, secs = parse_epoch_times(
        os.path.join(out_dir, "run.log"))
    counts = sparse.load_npz(os.path.join(data_dir, "counts114.npz"))
    A = int(np.load(os.path.join(data_dir,
                                  "author_indices114.npy")).max()) + 1
    V = counts.shape[1]

    rec = dict(
        tag=tag,
        D_factor=D_factor,
        D_actual=int(counts.shape[0]),
        A=A, V=V, K=K, epochs=epochs,
        n_epochs_observed=int(secs.size),
        total_seconds=float(secs.sum()) if secs.size else None,
        sec_per_epoch_mean=float(secs.mean()) if secs.size else None,
        sec_per_epoch_std=float(secs.std(ddof=1)) if secs.size > 1 else None,
        sec_per_epoch_p10=float(np.percentile(secs, 10)) if secs.size else None,
        sec_per_epoch_p50=float(np.percentile(secs, 50)) if secs.size else None,
        sec_per_epoch_p90=float(np.percentile(secs, 90)) if secs.size else None,
        first_epoch_seconds=float(secs[0]) if secs.size else None,
        final_elbo=float(elbos[-1]) if elbos.size else None,
        nnz=int(counts.nnz),
    )
    return rec


# ============================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--D-factors", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--K-values", type=int, nargs="+",
                    default=[10, 20, 30])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(RES_BASE, exist_ok=True)

    configs = [(D, K) for D in args.D_factors for K in args.K_values]
    print(f"\nRunning {len(configs)} configurations: "
          f"{len(args.D_factors)} D-factors x {len(args.K_values)} K-values")
    print(f"  D-factors : {args.D_factors}")
    print(f"  K-values  : {args.K_values}")
    print(f"  epochs    : {args.epochs}")
    print(f"  seed      : {args.seed}\n")

    results = []
    for i, (D_factor, K) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]  D_factor={D_factor:.2f}  K={K}")
        try:
            rec = run_one(D_factor, K, args.epochs, args.seed, rng)
            results.append(rec)
            if rec["sec_per_epoch_mean"] is not None:
                print(f"  -> D={rec['D_actual']}, sec/epoch="
                      f"{rec['sec_per_epoch_mean']:.2f} +- "
                      f"{rec['sec_per_epoch_std'] or 0:.2f}, "
                      f"total={rec['total_seconds']:.0f}s, "
                      f"nnz={rec['nnz']:,}")
            else:
                print(f"  -> no timing parsed; check run.log")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append(dict(
                tag=f"Dx{D_factor:.2f}_K{K}",
                D_factor=D_factor, K=K, error=str(e),
            ))

    df = pd.DataFrame(results)
    out_csv = os.path.join(RES_BASE, "scaling_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print(f"Results: {out_csv}")
    print(f"{'='*70}")
    cols = ["tag", "D_actual", "K", "nnz", "n_epochs_observed",
            "sec_per_epoch_mean", "sec_per_epoch_std",
            "total_seconds", "final_elbo"]
    show_cols = [c for c in cols if c in df.columns]
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
