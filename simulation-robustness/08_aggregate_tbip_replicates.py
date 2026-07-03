#!/usr/bin/env python3
"""
08_aggregate_tbip_replicates.py
================================
Aggregate the 20 TBIP-family fits (sim_tbip_01..sim_tbip_20) into a
single summary CSV. The DGP truth has CONSTANT ideal points across
topics; we fit STBS with ideal_dim="ak" (topic-varying) i.e. an
over-specified model. Per-replicate metrics:

  Ideal-point recovery
    - cor(x_hat_weighted, x_sim_a) : pol-weighted aggregate of the
      topic-varying x_hat_ak compared with the constant truth x_a
      (Pearson + Spearman)
    - cor(x_hat_mean, x_sim_a)     : simple per-author mean of x_hat_ak
    - mean_k cor(x_hat_ak, x_sim_a): how close are the per-topic
      x_hat columns to the constant truth, on average

  Iota recovery
    - cor flat over (K, J) cells of iota_hat vs iota_sim (broadcast)
    - per-cov: mean over k of iota_hat_kj   (should approximate iota_j)
    - per-cov: std  over k of iota_hat_kj   (should be small under correct DGP)
    - per-cov detection rate (HPD_0.05 excludes 0): for c0_zero this
      is the type-I rate, for c1..c4 the power.

  CCP-style detection (using existing 07c_iota_ccp output if present):
    - TP/FP/FN/TN, Precision, Recall, Specificity (per the saved
      iota_ccp_table.csv if available; else recomputed locally)

Aggregations:
  Mean +/- std over the 20 reps for each scalar; per-cov pooling.

Outputs: results_simulation/tbip_replicate_summary/
  global_replicate_metrics.csv
  global_replicate_summary.csv
  per_covariate_summary.csv
  per_replicate_per_covariate.csv
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.abspath(__file__))
RES_BASE = os.path.join(REPO, "results_simulation")
DATA_BASE = os.path.join(REPO, "data_simulation")
OUT_DIR = os.path.join(RES_BASE, "tbip_replicate_summary")
os.makedirs(OUT_DIR, exist_ok=True)

COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]
ACTIVE_COVS_FOR_ALPHA = [2, 3, 4]
N_SIMS = 20
Z_CRIT = norm.ppf(1.0 - 0.05 / 2.0)


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


def _hungarian_align_topics(eta_sim, eta_hat):
    """Match topics between sim and hat by max |cor|. Returns perm dict
    {sim_k -> hat_k} and per-topic sign needed to align eta."""
    K = eta_sim.shape[0]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            v = np.corrcoef(eta_sim[i], eta_hat[j])[0, 1]
            C[i, j] = abs(v) if not np.isnan(v) else 0.0
    row, col = linear_sum_assignment(-C)
    perm = {int(rr): int(cc) for rr, cc in zip(row, col)}
    eta_aligned = np.stack([eta_hat[perm[k]] for k in range(K)])
    signs = np.array([
        np.sign(np.corrcoef(eta_sim[k], eta_aligned[k])[0, 1])
        if not np.isnan(np.corrcoef(eta_sim[k], eta_aligned[k])[0, 1])
        else 1.0
        for k in range(K)
    ])
    return perm, signs


global_records = []
all_per_cov = []

for r in range(1, N_SIMS + 1):
    fit_dir = os.path.join(RES_BASE, f"sim_tbip_{r:02d}")
    gt_dir = os.path.join(DATA_BASE, f"sim_tbip_{r:02d}", "ground_truth")

    # -------- truth --------
    iota_sim = np.load(os.path.join(gt_dir, "iota_sim.npy"))            # (K, J), all rows equal
    iota_sim_vec = np.load(os.path.join(gt_dir, "iota_sim_vec.npy"))    # (J,) the constant
    ideal_sim = np.load(os.path.join(gt_dir, "ideal_sim.npy"))          # (N, K), all cols equal
    x_sim = np.load(os.path.join(gt_dir, "ideal_sim_vec.npy"))          # (N,) the constant author IP
    eta_sim = np.load(os.path.join(gt_dir, "eta.npy"))                  # (K, V)
    X = np.load(os.path.join(gt_dir, "X.npy"))                          # (N, J)
    K, J = iota_sim.shape

    # -------- fit --------
    iota_hat = np.load(os.path.join(fit_dir, "params",
                                    "iota_location_final.npy"))         # (K, J)
    eta_hat = np.load(os.path.join(fit_dir, "params",
                                   "eta_location_final.npy"))           # (K, V)
    x_hat_ak = np.load(os.path.join(fit_dir, "params",
                                    "ideal_point_location_final.npy"))  # (N, K)
    tril = np.load(os.path.join(fit_dir, "params",
                                "iota_scale_tril_final.npy"))           # (J, J)

    # Hungarian-align topics on |cor(eta)| and apply sign per topic
    perm, signs = _hungarian_align_topics(eta_sim, eta_hat)
    iota_hat_aligned = np.stack(
        [iota_hat[perm[k]] * signs[k] for k in range(K)]
    )                                                                    # (K, J)
    x_hat_ak_aligned = np.stack(
        [x_hat_ak[:, perm[k]] * signs[k] for k in range(K)], axis=1
    )                                                                    # (N, K)

    # ---------- ideal-point recovery ----------
    # theta-weighted aggregate of the fitted topic-varying ideal points,
    # using the same author weights w_{ak} prop. sum_{d in a} E theta_{dk}
    # as the headline aggregate of S.1.5 (theta is the STBS fit's own
    # document-topic intensity). This keeps the aggregation consistent
    # with the rest of the paper rather than using a study-specific
    # polarisation weighting.
    theta_fit = (np.load(os.path.join(fit_dir, "params",
                                      "theta_shape_final.npy"))
                 / np.load(os.path.join(fit_dir, "params",
                                        "theta_rate_final.npy")))          # (D, K)
    author_idx = np.load(os.path.join(DATA_BASE, f"sim_tbip_{r:02d}",
                                      "clean", "author_indices114.npy"))
    Nauth = x_hat_ak_aligned.shape[0]
    theta_auth = np.zeros((Nauth, K))
    for a in range(Nauth):
        theta_auth[a] = theta_fit[author_idx == a].sum(axis=0)
    # align theta columns to the same Hungarian permutation as x_hat
    theta_auth = np.stack([theta_auth[:, perm[k]] for k in range(K)], axis=1)
    w_ak = theta_auth / theta_auth.sum(axis=1, keepdims=True)
    x_hat_polmean = (x_hat_ak_aligned * w_ak).sum(axis=1)

    # Plain per-author mean across topics
    x_hat_mean = x_hat_ak_aligned.mean(axis=1)

    # If the global sign of (x, eta, iota) came out flipped relative to
    # the truth, the Pearson cor is negative. Flip jointly to align.
    cor_raw = _safe_corr(x_hat_polmean, x_sim)
    sign_flipped = bool((not np.isnan(cor_raw)) and cor_raw < 0)
    if sign_flipped:
        x_hat_polmean = -x_hat_polmean
        x_hat_mean = -x_hat_mean
        x_hat_ak_aligned = -x_hat_ak_aligned
        iota_hat_aligned = -iota_hat_aligned

    cor_x_polmean_pearson = _safe_corr(x_hat_polmean, x_sim)
    cor_x_polmean_spearman = _safe_spearman(x_hat_polmean, x_sim)
    cor_x_mean_pearson = _safe_corr(x_hat_mean, x_sim)
    # per-topic cor(x_hat_ak[:, k], x_sim_a) — should be high for all k
    cor_per_topic = np.array([
        _safe_corr(x_hat_ak_aligned[:, k], x_sim) for k in range(K)
    ])
    mean_cor_per_topic = float(np.nanmean(cor_per_topic))
    std_cor_per_topic = float(np.nanstd(cor_per_topic, ddof=1))

    # ---------- iota recovery ----------
    # Truth is constant; fit can vary. We measure both the closeness of
    # the AVERAGE of iota_hat over k to the truth, and the spread of
    # iota_hat across k.
    iota_hat_mean_k = iota_hat_aligned.mean(axis=0)          # (J,)
    iota_hat_std_k = iota_hat_aligned.std(axis=0)            # (J,)
    cor_iota_flat = _safe_corr(iota_hat_aligned.flatten(),
                               iota_sim.flatten())

    # CCP-style detection: per-cell HPD covers 0?
    cov = tril @ tril.T
    sd_per_coef = np.sqrt(np.diag(cov))                      # (J,) shared across k
    z_per_cell = np.abs(iota_hat_aligned) / np.maximum(sd_per_coef[None, :], 1e-12)
    detected_per_cell = (z_per_cell >= Z_CRIT).astype(int)   # (K, J)

    # Per-rep classification metrics: c0 active set is empty (true null
    # everywhere); c1..c4 active everywhere under TBIP.
    # gt_active_per_cell: c0 -> all 0, others -> all 1
    gt_active = np.zeros((K, J), dtype=int)
    gt_active[:, 1:] = 1
    TP = int(((detected_per_cell == 1) & (gt_active == 1)).sum())
    FP = int(((detected_per_cell == 1) & (gt_active == 0)).sum())
    FN = int(((detected_per_cell == 0) & (gt_active == 1)).sum())
    TN = int(((detected_per_cell == 0) & (gt_active == 0)).sum())
    pre = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    spe = TN / max(TN + FP, 1)

    # alpha* (slope of regression iota_hat on iota_sim, on truly active cells)
    mask = (gt_active == 1).flatten()
    if mask.sum() > 0:
        a = iota_sim.flatten()[mask]
        b = iota_hat_aligned.flatten()[mask]
        alpha_star = float(np.dot(a, b) / np.dot(a, a)) if np.dot(a, a) > 0 else float("nan")
    else:
        alpha_star = float("nan")

    # ---------- record ----------
    global_records.append(dict(
        rep=r,
        sign_flipped=sign_flipped,
        cor_x_polmean_pearson=cor_x_polmean_pearson,
        cor_x_polmean_spearman=cor_x_polmean_spearman,
        cor_x_mean_pearson=cor_x_mean_pearson,
        mean_cor_per_topic=mean_cor_per_topic,
        std_cor_per_topic=std_cor_per_topic,
        cor_iota_flat=cor_iota_flat,
        TP=TP, FP=FP, FN=FN, TN=TN,
        precision=pre, recall=rec, specificity=spe,
        alpha_star=alpha_star,
    ))

    for j in range(J):
        all_per_cov.append(dict(
            rep=r, j=j, covariate=COV_LABELS[j],
            iota_sim_const=float(iota_sim_vec[j]),
            iota_hat_mean_over_k=float(iota_hat_mean_k[j]),
            iota_hat_std_over_k=float(iota_hat_std_k[j]),
            detect_rate_over_k=float(detected_per_cell[:, j].mean()),
            sigma_l_post=float(sd_per_coef[j]),
        ))

global_df = pd.DataFrame(global_records)
global_df.to_csv(os.path.join(OUT_DIR, "global_replicate_metrics.csv"),
                 index=False)
print("\n=== Global per-replicate metrics (TBIP truth, fit STBS-x_ak) ===")
print(global_df.round(3).to_string(index=False))

summary_rows = []
metric_cols = ["cor_x_polmean_pearson", "cor_x_polmean_spearman",
               "cor_x_mean_pearson", "mean_cor_per_topic",
               "std_cor_per_topic", "cor_iota_flat",
               "TP", "FP", "FN", "TN",
               "precision", "recall", "specificity", "alpha_star"]
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
print("\n=== Summary over 20 replicates ===")
print(summary_df.round(3).to_string(index=False))

percov_df = pd.DataFrame(all_per_cov)
percov_df.to_csv(os.path.join(OUT_DIR, "per_replicate_per_covariate.csv"),
                 index=False)
agg = percov_df.groupby(["j", "covariate"]).agg(
    n_reps=("rep", "size"),
    iota_sim_const_mean=("iota_sim_const", "mean"),
    iota_hat_mean_over_reps=("iota_hat_mean_over_k", "mean"),
    iota_hat_std_over_reps=("iota_hat_mean_over_k", "std"),
    iota_hat_within_rep_std_mean=("iota_hat_std_over_k", "mean"),
    detect_rate_pooled=("detect_rate_over_k", "mean"),
    sigma_l_post_mean=("sigma_l_post", "mean"),
).reset_index()
agg.to_csv(os.path.join(OUT_DIR, "per_covariate_summary.csv"), index=False)
print("\n=== Per-covariate summary ===")
print(agg.round(3).to_string(index=False))

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
