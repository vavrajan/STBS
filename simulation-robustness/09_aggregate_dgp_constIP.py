#!/usr/bin/env python3
"""
09_aggregate_dgp_constIP.py
===========================
Aggregate the 20 misspecified DGP-family fits where STBS was run with
ideal_dim="a" and iota_dim="l" (constant ideal point per author, shared
regression coefficients across topics) on data simulated under the
topic-varying truth (ideal_dim="ak", iota_dim="kl").

Per-replicate scalars:
  - cor(x_hat_a, mean_k x_sim_a,k)    [Pearson, Spearman]
  - cor(x_hat_a, party_scalar)        [-1 = D, +1 = R, 0 = I/other]
  - cor(x_hat_a, polarisation-weighted mean over k)
  - For each j=0..4:
      * iota_hat_l_j  (the single coefficient learned for cov j)
      * iota_sim_mean_j = mean_k iota_sim[k, j]
      * |z| = |iota_hat_l_j| / sigma_hat_l_j   -> detection at HPD_0.05
                                                 if |z| >= 1.96
  - mean per-topic |cor(eta_sim_k, eta_hat_k)|  (Hungarian-aligned)

Aggregations:
  Mean ± std over the 20 reps for each scalar; per-covariate detection
  rate; per-covariate mean signed iota_hat_l vs mean signed iota_sim_mean.

Outputs: results_simulation/dgp_constIP_summary/
  global_replicate_metrics.csv      (one row per sim)
  global_replicate_summary.csv      (mean ± std + 95% replicate band)
  per_covariate_summary.csv         (iota_hat_l per cov, det rate)
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.abspath(__file__))
RES_BASE = os.path.join(REPO, "results_simulation")
DATA_BASE = os.path.join(REPO, "data_simulation")
OUT_DIR = os.path.join(RES_BASE, "dgp_constIP_summary")
os.makedirs(OUT_DIR, exist_ok=True)

COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]
N_SIMS = 20
Z_CRIT = 1.959963984540054  # qnorm(0.975)


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def _party_scalar(author_info_path):
    """Return (-1, 0, +1) per author from the party column."""
    df = pd.read_csv(author_info_path)
    party = df["party"].astype(str)
    s = (-1.0 * (party == "D").to_numpy(dtype=float)
         + 1.0 * (party == "R").to_numpy(dtype=float))
    return s


def _hungarian_align_eta(eta_sim, eta_hat):
    """Match topics between sim and hat by max |cor|, return aligned hat."""
    K = eta_sim.shape[0]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            v = np.corrcoef(eta_sim[i], eta_hat[j])[0, 1]
            C[i, j] = abs(v) if not np.isnan(v) else 0.0
    row, col = linear_sum_assignment(-C)
    perm = {int(rr): int(cc) for rr, cc in zip(row, col)}
    eta_ha = np.stack([eta_hat[perm[k]] for k in range(K)])
    return eta_ha


global_records = []
all_per_cov = []

for r in range(1, N_SIMS + 1):
    fit_dir = os.path.join(RES_BASE, f"sim_dgp_{r:02d}_constIP")
    gt_dir = os.path.join(DATA_BASE, f"sim_dgp_{r:02d}", "ground_truth")
    auth_csv = os.path.join(DATA_BASE, f"sim_dgp_{r:02d}", "clean",
                            "author_detailed_info114.csv")

    # ----- load truths ------------------------------------------------
    ideal_sim = np.load(os.path.join(gt_dir, "ideal_sim.npy"))      # (N, K)
    iota_sim = np.load(os.path.join(gt_dir, "iota_sim.npy"))        # (K, J)
    eta_sim = np.load(os.path.join(gt_dir, "eta.npy"))              # (K, V)
    # ground-truth theta + doc->author map for the theta-weighted
    # aggregate of S.1.5 (w_{ak} prop. sum_{d in a} E theta_{dk})
    theta_gt = np.load(os.path.join(gt_dir, "theta.npy"))          # (D, K)
    author_idx = np.load(os.path.join(DATA_BASE, f"sim_dgp_{r:02d}",
                                      "clean", "author_indices114.npy"))
    N, K = ideal_sim.shape
    J = iota_sim.shape[1]

    # ----- load misspecified fit -------------------------------------
    x_hat = np.load(os.path.join(fit_dir, "params",
                                 "ideal_point_location_final.npy"))   # (N, 1)
    iota_hat = np.load(os.path.join(fit_dir, "params",
                                    "iota_location_final.npy"))       # (1, J)
    # Variance of iota_hat per coefficient: from scale_tril. For
    # iota_coef_jointly=True the tril is shared (J, J) Cholesky factor
    # of the joint coef covariance.
    tril = np.load(os.path.join(fit_dir, "params",
                                "iota_scale_tril_final.npy"))
    if tril.ndim == 2 and tril.shape == (J, J):
        cov = tril @ tril.T
        iota_sd = np.sqrt(np.diag(cov))                                # (J,)
    else:
        # Fallback: per-row tril
        cov = tril @ np.swapaxes(tril, -1, -2)
        iota_sd = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
        if iota_sd.ndim == 2:
            iota_sd = iota_sd.mean(axis=0)                             # (J,)
    eta_hat = np.load(os.path.join(fit_dir, "params",
                                   "eta_location_final.npy"))         # (K, V)

    x_hat = x_hat.ravel()
    iota_hat = iota_hat.ravel()

    # ----- IDEAL-POINT recovery --------------------------------------
    x_sim_mean = ideal_sim.mean(axis=1)                                # (N,)
    # theta-weighted aggregate of the true topic-varying ideal points,
    # using the S.1.5 author weights w_{ak} prop. sum_{d in a} E theta_{dk}.
    theta_auth = np.zeros((N, K))                                      # (N, K)
    for a in range(N):
        theta_auth[a] = theta_gt[author_idx == a].sum(axis=0)
    w_ak = theta_auth / theta_auth.sum(axis=1, keepdims=True)          # row-normalise
    x_sim_polmean = (ideal_sim * w_ak).sum(axis=1)                     # (N,) theta-weighted

    party = _party_scalar(auth_csv)
    if party.shape[0] != N:
        # author_detailed_info covers same row order as ideal
        # if length mismatch, pad/truncate (shouldn't happen)
        m = min(party.shape[0], N)
        party = party[:m]
        x_hat_p = x_hat[:m]
    else:
        x_hat_p = x_hat

    # Sign-flip (x_hat, iota_hat) jointly so that the AGGREGATE x_hat
    # signal is positively aligned with the topic-mean of the truth.
    # The constant-IP model is identifiable only up to a global sign on
    # (x_a, eta_kv); when x flips, the regression coefficients iota_l
    # must flip too. Aligning to the topic-mean is the natural choice
    # since that is the signal the misspecified model can possibly
    # recover (party-alignment is broken by sign-balancing across
    # topics — see §sec:mc_dgp_constIP).
    iota_sim_mean = iota_sim.mean(axis=0)                              # (J,)
    cor_raw_mean = _safe_corr(x_hat, x_sim_mean)
    if not np.isnan(cor_raw_mean) and cor_raw_mean < 0:
        x_hat = -x_hat
        x_hat_p = -x_hat_p
        iota_hat = -iota_hat

    cor_x_pearson_mean = _safe_corr(x_hat, x_sim_mean)
    cor_x_spearman_mean = _safe_spearman(x_hat, x_sim_mean)
    cor_x_pearson_polmean = _safe_corr(x_hat, x_sim_polmean)
    cor_x_pearson_party = _safe_corr(x_hat_p, party)
    abs_cor_x_party = abs(cor_x_pearson_party)

    # ----- IOTA recovery ---------------------------------------------
    cor_iota_l = _safe_corr(iota_hat, iota_sim_mean)

    # Per-coef detection (HPD_0.05 excludes 0 iff |loc| / sd >= z_crit)
    z_abs = np.abs(iota_hat) / np.maximum(iota_sd, 1e-12)
    detected = (z_abs >= Z_CRIT).astype(int)

    # ----- ETA recovery ----------------------------------------------
    eta_ha = _hungarian_align_eta(eta_sim, eta_hat)
    mean_abs_cor_eta_topic = float(np.nanmean([
        abs(np.corrcoef(eta_sim[k], eta_ha[k])[0, 1]) for k in range(K)
    ]))

    # ----- record ----------------------------------------------------
    global_records.append(dict(
        rep=r,
        cor_x_pearson_mean=cor_x_pearson_mean,
        cor_x_spearman_mean=cor_x_spearman_mean,
        cor_x_pearson_polmean=cor_x_pearson_polmean,
        cor_x_pearson_party=cor_x_pearson_party,
        abs_cor_x_party=abs_cor_x_party,
        cor_iota_l=cor_iota_l,
        mean_abs_cor_eta_topic=mean_abs_cor_eta_topic,
    ))
    for j in range(J):
        all_per_cov.append(dict(
            rep=r, j=j, covariate=COV_LABELS[j],
            iota_hat=float(iota_hat[j]),
            iota_sd=float(iota_sd[j]),
            iota_sim_mean=float(iota_sim_mean[j]),
            iota_sim_std_over_k=float(iota_sim[:, j].std()),
            z=float(z_abs[j]),
            detected=int(detected[j]),
        ))

global_df = pd.DataFrame(global_records)
global_df.to_csv(os.path.join(OUT_DIR, "global_replicate_metrics.csv"),
                 index=False)
print("\n=== Global per-replicate metrics (constant-IP fit) ===")
print(global_df.round(3).to_string(index=False))

# Summary statistics over the reps
summary_rows = []
metric_cols = ["cor_x_pearson_mean", "cor_x_spearman_mean",
               "cor_x_pearson_polmean", "cor_x_pearson_party",
               "abs_cor_x_party",
               "cor_iota_l", "mean_abs_cor_eta_topic"]
for col in metric_cols:
    v = global_df[col].dropna().values
    summary_rows.append(dict(
        metric=col,
        mean=float(np.mean(v)) if v.size else float("nan"),
        std=float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        q025=float(np.quantile(v, 0.025)) if v.size else float("nan"),
        q500=float(np.quantile(v, 0.500)) if v.size else float("nan"),
        q975=float(np.quantile(v, 0.975)) if v.size else float("nan"),
        min=float(np.min(v)) if v.size else float("nan"),
        max=float(np.max(v)) if v.size else float("nan"),
    ))
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "global_replicate_summary.csv"),
                  index=False)
print("\n=== Summary over 20 replicates (mean ± std, 95% replicate band) ===")
print(summary_df.round(3).to_string(index=False))

# Per-covariate aggregation
percov_df = pd.DataFrame(all_per_cov)
agg_rows = []
for j in range(5):
    sub = percov_df[percov_df["j"] == j]
    agg_rows.append(dict(
        j=j, covariate=COV_LABELS[j],
        n_reps=len(sub),
        iota_hat_mean=float(sub["iota_hat"].mean()),
        iota_hat_std=float(sub["iota_hat"].std(ddof=1)),
        iota_sim_mean_mean=float(sub["iota_sim_mean"].mean()),
        iota_sim_mean_std=float(sub["iota_sim_mean"].std(ddof=1)),
        iota_sim_std_over_k_mean=float(sub["iota_sim_std_over_k"].mean()),
        detect_rate=float(sub["detected"].mean()),
        z_abs_mean=float(sub["z"].mean()),
    ))
percov_summary = pd.DataFrame(agg_rows)
percov_summary.to_csv(os.path.join(OUT_DIR, "per_covariate_summary.csv"),
                      index=False)
print("\n=== Per-covariate summary (constant-IP fit, 20 reps) ===")
print(percov_summary.round(3).to_string(index=False))

# Also save per-rep per-cov
percov_df.to_csv(os.path.join(OUT_DIR, "per_replicate_per_covariate.csv"),
                 index=False)

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
