#!/usr/bin/env python3
"""
08_aggregate_dgp_replicates.py
==============================
Aggregate the 20 DGP-family Monte Carlo fits (sim_dgp_01..sim_dgp_20)
into a single summary CSV. In contrast to the fixed-truth aggregation,
each replicate has its OWN ground truth (different X, iota, active
patterns). Per-cell aggregation is therefore not meaningful — only
SCALAR metrics per replicate, then aggregated over reps.

Per-replicate scalars:
  TP / FP / FN / TN, Precision, Recall, Specificity
  alpha_star (over GT-active entries of c2/c3/c4)
  flat-cor(iota_sim, iota_hat) over all 125 cells
  flat-cor(eta_sim, eta_hat) over all 51K eta values
  per-topic mean cor(eta_k_sim, eta_hat_k)
  per-topic mean |cor(ideal_sim_k, ideal_hat_k)|

Aggregations:
  Mean ± std over 20 reps for each scalar.
  Per-covariate HPD-coverage and detection-rate, separately for
  active and inactive cells (pooled across reps).

Outputs: results_simulation/dgp_replicate_summary/
  global_replicate_metrics.csv      (one row per sim)
  global_replicate_summary.csv      (mean ± std + 95% replicate band)
  per_covariate_summary.csv         (HPD cov & detect rate per cov,
                                     pooled across reps)
"""
import os, json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.abspath(__file__))
RES_BASE = os.path.join(REPO, "results_simulation")
DATA_BASE = os.path.join(REPO, "data_simulation")
OUT_DIR = os.path.join(RES_BASE, "dgp_replicate_summary")
os.makedirs(OUT_DIR, exist_ok=True)

COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]
ACTIVE_COVS_FOR_ALPHA = [2, 3, 4]
N_SIMS = 20

global_records = []
all_per_cov = []   # per (rep, cov) — for pooled per-covariate aggregation

for r in range(1, N_SIMS + 1):
    fit_dir = os.path.join(RES_BASE, f"sim_dgp_{r:02d}")
    gt_dir = os.path.join(DATA_BASE, f"sim_dgp_{r:02d}", "ground_truth")

    # --- per-rep scalar global metrics ---
    df = pd.read_csv(os.path.join(fit_dir, "iota_ccp_table.csv"))
    cls = df["classification"].value_counts().to_dict()
    TP = int(cls.get("TP", 0)); FP = int(cls.get("FP", 0))
    FN = int(cls.get("FN", 0)); TN = int(cls.get("TN", 0))
    pre = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    spe = TN / max(TN + FP, 1)

    mask = (df["gt_active"] == 1) & df["j"].isin(ACTIVE_COVS_FOR_ALPHA)
    if mask.sum() > 0:
        alpha = (df.loc[mask, "iota_sim"] * df.loc[mask, "iota_hat"]).sum() \
                / (df.loc[mask, "iota_sim"] ** 2).sum()
    else:
        alpha = float("nan")
    cor_iota_flat = float(np.corrcoef(df["iota_sim"], df["iota_hat"])[0, 1])

    # --- eta and ideal alignment + correlations ---
    eta_sim = np.load(os.path.join(gt_dir, "eta.npy"))
    ideal_sim = np.load(os.path.join(gt_dir, "ideal_sim.npy"))
    eta_hat = np.load(os.path.join(fit_dir, "params", "eta_location_final.npy"))
    ideal_hat = np.load(os.path.join(fit_dir, "params",
                                      "ideal_point_location_final.npy"))
    K = eta_sim.shape[0]

    # Hungarian on |cor(ideal)|
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            v = np.corrcoef(ideal_sim[:, i], ideal_hat[:, j])[0, 1]
            C[i, j] = abs(v) if not np.isnan(v) else 0.0
    row, col = linear_sum_assignment(-C)
    perm = {int(rr): int(cc) for rr, cc in zip(row, col)}
    eta_ha = np.stack([eta_hat[perm[k]] for k in range(K)])
    ideal_ha = np.stack([ideal_hat[:, perm[k]] for k in range(K)], axis=1)
    signs = np.array([
        np.sign(np.corrcoef(ideal_sim[:, k], ideal_ha[:, k])[0, 1])
        if not np.isnan(np.corrcoef(ideal_sim[:, k], ideal_ha[:, k])[0, 1])
        else 1.0
        for k in range(K)
    ])
    eta_ha = eta_ha * signs[:, None]
    ideal_ha = ideal_ha * signs[None, :]

    cor_eta_flat = float(np.corrcoef(eta_sim.flatten(),
                                      eta_ha.flatten())[0, 1])
    cor_eta_per = np.array([
        np.corrcoef(eta_sim[k], eta_ha[k])[0, 1] for k in range(K)
    ])
    mean_cor_eta_topic = float(np.nanmean(cor_eta_per))
    mean_abs_cor_ideal_topic = float(np.nanmean([
        abs(np.corrcoef(ideal_sim[:, k], ideal_ha[:, k])[0, 1])
        for k in range(K)
    ]))

    # --- HPD coverage of truth + detect-rate per cell, separately for
    #     active and inactive (pooled across reps below) ---
    df = df.assign(hpd_covers_truth=(
        (df["hpd_lo"] <= df["iota_sim"]) &
        (df["iota_sim"] <= df["hpd_hi"])
    ).astype(int))
    df = df.assign(detected=df["classification"].isin(["TP", "FP"]).astype(int))

    for j in range(5):
        sub = df[df["j"] == j]
        n_act = int((sub["gt_active"] == 1).sum())
        n_inact = int((sub["gt_active"] == 0).sum())
        all_per_cov.append(dict(
            rep=r, j=j, covariate=COV_LABELS[j],
            n_active=n_act, n_inactive=n_inact,
            cov_active_sum=int(sub.loc[sub["gt_active"] == 1, "hpd_covers_truth"].sum()),
            cov_inactive_sum=int(sub.loc[sub["gt_active"] == 0, "hpd_covers_truth"].sum()),
            det_active_sum=int(sub.loc[sub["gt_active"] == 1, "detected"].sum()),
            det_inactive_sum=int(sub.loc[sub["gt_active"] == 0, "detected"].sum()),
        ))

    meta = json.load(open(os.path.join(fit_dir, "iota_ccp_meta.json")))
    global_records.append(dict(
        rep=r, TP=TP, FP=FP, FN=FN, TN=TN,
        precision=pre, recall=rec, specificity=spe,
        alpha_star=alpha,
        cor_iota_flat=cor_iota_flat,
        cor_eta_flat=cor_eta_flat,
        mean_cor_eta_topic=mean_cor_eta_topic,
        mean_abs_cor_ideal_topic=mean_abs_cor_ideal_topic,
        sigma_c0=meta["sigma_j"][0],
        sigma_c1=meta["sigma_j"][1],
        sigma_c2=meta["sigma_j"][2],
        sigma_c3=meta["sigma_j"][3],
        sigma_c4=meta["sigma_j"][4],
    ))

global_df = pd.DataFrame(global_records)
global_df.to_csv(os.path.join(OUT_DIR, "global_replicate_metrics.csv"),
                  index=False)
print("\n=== Global per-replicate metrics ===")
print(global_df.round(3).to_string(index=False))

# Summary statistics over the 20 reps
summary_rows = []
for col in ["TP", "FP", "FN", "TN", "precision", "recall", "specificity",
            "alpha_star", "cor_iota_flat", "cor_eta_flat",
            "mean_cor_eta_topic", "mean_abs_cor_ideal_topic"]:
    v = global_df[col].values
    summary_rows.append(dict(
        metric=col,
        mean=float(np.mean(v)),
        std=float(np.std(v, ddof=1)),
        q025=float(np.quantile(v, 0.025)),
        q500=float(np.quantile(v, 0.500)),
        q975=float(np.quantile(v, 0.975)),
        min=float(np.min(v)),
        max=float(np.max(v)),
    ))
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "global_replicate_summary.csv"),
                   index=False)
print("\n=== Summary over 20 replicates (mean ± std, 95% replicate band) ===")
print(summary_df.round(3).to_string(index=False))

# Per-covariate aggregation: pool counts across reps, then compute rates
percov_df = pd.DataFrame(all_per_cov)
agg_rows = []
for j in range(5):
    sub = percov_df[percov_df["j"] == j]
    n_act_total = sub["n_active"].sum()
    n_inact_total = sub["n_inactive"].sum()
    cov_act = sub["cov_active_sum"].sum() / max(n_act_total, 1)
    cov_in = sub["cov_inactive_sum"].sum() / max(n_inact_total, 1)
    det_act = sub["det_active_sum"].sum() / max(n_act_total, 1)
    det_in = sub["det_inactive_sum"].sum() / max(n_inact_total, 1)
    agg_rows.append(dict(
        j=j, covariate=COV_LABELS[j],
        total_active_cells=int(n_act_total),
        total_inactive_cells=int(n_inact_total),
        hpd_cov_active=float(cov_act) if n_act_total > 0 else float("nan"),
        hpd_cov_inactive=float(cov_in) if n_inact_total > 0 else float("nan"),
        detect_rate_active=float(det_act) if n_act_total > 0 else float("nan"),
        detect_rate_inactive=float(det_in) if n_inact_total > 0 else float("nan"),
    ))
percov_summary = pd.DataFrame(agg_rows)
percov_summary.to_csv(os.path.join(OUT_DIR, "per_covariate_summary.csv"),
                       index=False)
print("\n=== Per-covariate summary (pooled across all reps) ===")
print(percov_summary.round(3).to_string(index=False))

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
