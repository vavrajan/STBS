#!/usr/bin/env python3
"""
13b_scaling_summarize.py
========================
Build the scaling_summary.csv from already-completed
results_simulation/scaling_Dx*_K*/ directories without re-running
13_scaling_STBS.py. We mark a config as DONE if
params/iota_location_final.npy exists, and parse run.log for per-epoch
times. Partial / interrupted configs are skipped.
"""
import os, re
import numpy as np
import pandas as pd
from scipy import sparse

REPO = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(REPO, "results_simulation")
SIM = os.path.join(REPO, "data_simulation")

EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+done\s+\|\s+ELBO:\s+([-+\d.eE]+)\s+\|\s+"
    r"\(([\d.]+)\s+sec/epoch\)"
)


def parse_epoch_times(run_log_path):
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


def main():
    tag_re = re.compile(r"scaling_Dx([\d.]+)_K(\d+)$")
    rows = []
    for tag in sorted(os.listdir(RES)):
        d = os.path.join(RES, tag)
        if not os.path.isdir(d):
            continue
        m = tag_re.match(tag)
        if not m:
            continue
        D_factor = float(m.group(1))
        K = int(m.group(2))

        marker = os.path.join(d, "params", "iota_location_final.npy")
        done = os.path.exists(marker)

        eps, elbos, secs = parse_epoch_times(os.path.join(d, "run.log"))

        data_clean = os.path.join(SIM, tag, "clean")
        counts_p = os.path.join(data_clean, "counts114.npz")
        ai_p = os.path.join(data_clean, "author_indices114.npy")
        if os.path.exists(counts_p) and os.path.exists(ai_p):
            counts = sparse.load_npz(counts_p)
            ai = np.load(ai_p)
            A = int(ai.max()) + 1
            V = counts.shape[1]
            D_actual = counts.shape[0]
            nnz = int(counts.nnz)
        else:
            A = V = D_actual = nnz = None

        rec = dict(
            tag=tag,
            D_factor=D_factor,
            D_actual=D_actual,
            A=A, V=V, K=K,
            done=done,
            n_epochs_observed=int(secs.size),
            total_seconds=float(secs.sum()) if secs.size else None,
            sec_per_epoch_mean=float(secs.mean()) if secs.size else None,
            sec_per_epoch_std=float(secs.std(ddof=1)) if secs.size > 1 else None,
            sec_per_epoch_p10=float(np.percentile(secs, 10)) if secs.size else None,
            sec_per_epoch_p50=float(np.percentile(secs, 50)) if secs.size else None,
            sec_per_epoch_p90=float(np.percentile(secs, 90)) if secs.size else None,
            first_epoch_seconds=float(secs[0]) if secs.size else None,
            final_elbo=float(elbos[-1]) if elbos.size else None,
            nnz=nnz,
        )
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values(["D_factor", "K"]).reset_index(drop=True)
    out = os.path.join(RES, "scaling_summary.csv")
    df.to_csv(out, index=False)

    print(f"\nWrote {out}\n")
    show = ["tag", "D_actual", "K", "nnz", "n_epochs_observed", "done",
            "sec_per_epoch_mean", "sec_per_epoch_std",
            "total_seconds", "final_elbo"]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
