#!/usr/bin/env python3
"""
08_aggregate_replicates.py
==========================
Aggregate the 20 Monte Carlo STBS fits (sim_01..sim_20) into a single
summary CSV + a small set of figures, ready to be cited from the
simulation.tex appendix.

Per-(k, j) cell:
  - mean / std / 2.5%-97.5% quantiles of iota_hat across 20 reps
  - HPD-coverage: fraction of reps whose HPD_0.05 contains iota_sim
  - Detection-rate:  fraction of reps with CCP < 0.05

Per-replicate scalars:
  - TP / FP / FN / TN, Precision / Recall / Specificity
  - alpha* (over c2,c3,c4 active entries)
  - flat-cor(iota), flat-cor(eta), per-topic mean cor(eta)

Outputs: results_simulation/replicate_summary/
  iota_replicate_per_cell.csv
  iota_replicate_per_covariate.csv
  global_replicate_metrics.csv
  global_replicate_summary.csv         (mean ± std over 20 reps)
  replicate_forest.png                 (forest with 20-sim mean + band)
  hpd_coverage_heatmap.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

REPO = os.path.dirname(os.path.abspath(__file__))
RES_BASE = os.path.join(REPO, "results_simulation")
GT_DIR   = os.path.join(REPO, "data_simulation", "simdata_centered_design", "ground_truth")
OUT_DIR  = os.path.join(RES_BASE, "centered_replicate_summary")
os.makedirs(OUT_DIR, exist_ok=True)

COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]
ACTIVE_COVS_FOR_ALPHA = [2, 3, 4]   # c2,c3,c4: topic-varying
N_SIMS = 20
ALPHA = 0.05
Z = chi2.ppf(1 - ALPHA, df=1) ** 0.5   # ≈ 1.96

# ================================================================== #
# Load shared GT once
# ================================================================== #
iota_sim_full = np.load(os.path.join(GT_DIR, "iota_sim.npy"))      # (25, 5)
ideal_sim = np.load(os.path.join(GT_DIR, "ideal_sim.npy"))          # (99, 25)
eta_sim   = np.load(os.path.join(GT_DIR, "eta.npy"))                # (25, 2066)
K, L = iota_sim_full.shape

# Polarization metric (kept for annotation; topics are NOT re-ordered).
pol_sim = np.array([ideal_sim[:, k].std() * eta_sim[k].std()
                    for k in range(K)])
# Natural topic order 0..K-1 on the y-axis of forest / heatmap plots.
pol_order = np.arange(K)


# ================================================================== #
# Walk the 20 replicates
# ================================================================== #

per_cell_records = []        # one row per (sim_idx, k, j)
global_records  = []         # one row per sim_idx

for sim_idx in range(1, N_SIMS + 1):
    sim_dir = os.path.join(RES_BASE, f"sim_{sim_idx:02d}")
    ccp_table = pd.read_csv(os.path.join(sim_dir, "iota_ccp_table.csv"))
    meta = json.load(open(os.path.join(sim_dir, "iota_ccp_meta.json")))

    # Ensure (k, j) order is stable
    ccp_table = ccp_table.sort_values(["k", "j"]).reset_index(drop=True)
    ccp_table["sim"] = sim_idx
    per_cell_records.append(ccp_table)

    # Global scalars from the per-cell table
    iota_sim_arr = ccp_table["iota_sim"].values
    iota_hat_arr = ccp_table["iota_hat"].values
    gt_active    = ccp_table["gt_active"].values.astype(bool)
    sigma_arr    = ccp_table["sigma_j"].values
    classification = ccp_table["classification"].values

    # alpha*: over c2, c3, c4 active entries
    mask_alpha = (np.isin(ccp_table["j"].values, ACTIVE_COVS_FOR_ALPHA)
                  & gt_active)
    if mask_alpha.sum() > 0:
        num = (iota_sim_arr[mask_alpha] * iota_hat_arr[mask_alpha]).sum()
        den = (iota_sim_arr[mask_alpha] ** 2).sum()
        alpha_star = num / den
    else:
        alpha_star = np.nan

    # flat cor(iota)
    cor_iota_flat = np.corrcoef(iota_sim_arr, iota_hat_arr)[0, 1]

    # Counts
    counts = {c: int((classification == c).sum())
              for c in ("TP", "FP", "FN", "TN")}
    TP, FP, FN, TN = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)
    specificity = TN / max(TN + FP, 1)

    # eta / ideal flat cor — need to load NPYs and do alignment
    fit_dir = os.path.join(sim_dir, "params")
    eta_hat = np.load(os.path.join(fit_dir, "eta_location_final.npy"))
    ideal_hat = np.load(os.path.join(fit_dir, "ideal_point_location_final.npy"))

    # Hungarian on |cor(ideal)|
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            r = np.corrcoef(ideal_sim[:, i], ideal_hat[:, j])[0, 1]
            C[i, j] = abs(r) if not np.isnan(r) else 0.0
    row, col = linear_sum_assignment(-C)
    perm = {int(r): int(c) for r, c in zip(row, col)}
    ideal_hat_a = np.stack([ideal_hat[:, perm[k]] for k in range(K)], axis=1)
    eta_hat_a   = np.stack([eta_hat[perm[k]] for k in range(K)])
    # sign per topic
    signs = np.array([np.sign(np.corrcoef(ideal_sim[:, k],
                                           ideal_hat_a[:, k])[0, 1])
                       if not np.isnan(np.corrcoef(ideal_sim[:, k],
                                                    ideal_hat_a[:, k])[0, 1])
                       else 1.0
                      for k in range(K)])
    ideal_hat_a = ideal_hat_a * signs[None, :]
    eta_hat_a   = eta_hat_a * signs[:, None]

    cor_eta_flat = np.corrcoef(eta_sim.flatten(), eta_hat_a.flatten())[0, 1]
    cor_eta_per_topic = np.array([np.corrcoef(eta_sim[k], eta_hat_a[k])[0, 1]
                                   for k in range(K)])
    mean_cor_eta_topic = float(np.nanmean(cor_eta_per_topic))
    mean_abs_cor_ideal_topic = float(np.nanmean(
        [abs(np.corrcoef(ideal_sim[:, k], ideal_hat_a[:, k])[0, 1])
         for k in range(K)]))

    global_records.append(dict(
        sim=sim_idx,
        TP=TP, FP=FP, FN=FN, TN=TN,
        precision=precision, recall=recall, specificity=specificity,
        alpha_star=alpha_star,
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
print("\nGlobal per-replicate metrics:")
print(global_df.round(3).to_string(index=False))

# Summary stats over 20 reps
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
print("\nSummary over 20 replicates (mean ± std, 95% replicate band):")
print(summary_df.round(3).to_string(index=False))


# ================================================================== #
# Per-cell aggregation
# ================================================================== #
all_cells = pd.concat(per_cell_records, ignore_index=True)
all_cells.to_csv(os.path.join(OUT_DIR, "iota_replicate_long.csv"), index=False)

# pivot: mean/std/quantiles of iota_hat across reps
agg = (
    all_cells
    .groupby(["k", "j"])
    .agg(
        iota_sim   = ("iota_sim",   "first"),
        gt_active  = ("gt_active",  "first"),
        iota_hat_mean = ("iota_hat", "mean"),
        iota_hat_std  = ("iota_hat", "std"),
        iota_hat_q025 = ("iota_hat", lambda v: np.quantile(v, 0.025)),
        iota_hat_q975 = ("iota_hat", lambda v: np.quantile(v, 0.975)),
        sigma_j_mean  = ("sigma_j", "mean"),
        detect_rate   = ("classification",
                          lambda v: float((v.isin(["TP", "FP"])).mean())),
    )
    .reset_index()
)
# HPD coverage per cell: fraction of reps whose [hpd_lo, hpd_hi] contains iota_sim
hpd_cov = (
    all_cells.assign(cov=lambda d: ((d.hpd_lo <= d.iota_sim)
                                     & (d.hpd_hi >= d.iota_sim)).astype(int))
              .groupby(["k", "j"])["cov"].mean()
              .reset_index().rename(columns={"cov": "hpd_coverage"})
)
agg = agg.merge(hpd_cov, on=["k", "j"])
agg["covariate"] = agg["j"].map({i: lab for i, lab in enumerate(COV_LABELS)})
agg.to_csv(os.path.join(OUT_DIR, "iota_replicate_per_cell.csv"),
            index=False)

# ================================================================== #
# Per-covariate aggregation
# ================================================================== #
per_cov_rows = []
for j, lab in enumerate(COV_LABELS):
    sub = agg[agg["j"] == j]
    n_active = int(sub["gt_active"].sum())
    n_inactive = int(len(sub) - n_active)
    cov_active = sub.loc[sub["gt_active"] == 1, "hpd_coverage"]
    cov_inactive = sub.loc[sub["gt_active"] == 0, "hpd_coverage"]
    detect_active = sub.loc[sub["gt_active"] == 1, "detect_rate"]
    detect_inactive = sub.loc[sub["gt_active"] == 0, "detect_rate"]
    per_cov_rows.append(dict(
        j=j, covariate=lab,
        n_active=n_active, n_inactive=n_inactive,
        hpd_cov_active_mean   = float(cov_active.mean())   if n_active   else np.nan,
        hpd_cov_inactive_mean = float(cov_inactive.mean()) if n_inactive else np.nan,
        detect_rate_active_mean   = float(detect_active.mean())   if n_active   else np.nan,
        detect_rate_inactive_mean = float(detect_inactive.mean()) if n_inactive else np.nan,
        sigma_j_mean_over_reps    = float(sub["sigma_j_mean"].mean()),
    ))
per_cov_df = pd.DataFrame(per_cov_rows)
per_cov_df.to_csv(os.path.join(OUT_DIR, "iota_replicate_per_covariate.csv"),
                   index=False)
print("\nPer-covariate replicate summary:")
print(per_cov_df.round(3).to_string(index=False))


# ================================================================== #
# Figure 1: replicate forest plot
# ================================================================== #
row_of_k = {int(k): i for i, k in enumerate(pol_order)}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()

for j in range(L):
    ax = axes[j]
    sub = agg[agg["j"] == j].reset_index(drop=True)
    for _, r in sub.iterrows():
        k = int(r["k"])
        y = row_of_k[k]
        m = float(r["iota_hat_mean"])
        lo = float(r["iota_hat_q025"]); hi = float(r["iota_hat_q975"])
        gt = float(r["iota_sim"])
        gt_a = bool(r["gt_active"])
        # color by HPD-coverage and detection
        cov = float(r["hpd_coverage"])
        det = float(r["detect_rate"])
        if gt_a:
            color = "tab:green" if det > 0.5 else "tab:orange"
        else:
            color = "tab:red" if det > 0.5 else "lightgray"
        face = color if gt_a else "white"
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]],
                     fmt='o', color=color, markerfacecolor=face,
                     markeredgecolor=color, markersize=6,
                     capsize=2, elinewidth=0.7)
        if gt != 0:
            ax.scatter(gt, y, marker='x', color='black', s=40, zorder=5)
        # annotate coverage with small text right of bar
        ax.text(hi + 0.03, y, f"{cov*100:.0f}%",
                fontsize=6, color=color, va='center')
    ax.axvline(0, color='black', ls='--', lw=0.6, alpha=0.5)
    ax.set_yticks(range(K)); ax.invert_yaxis()
    ax.set_yticklabels([f"{int(k)}  (pol={pol_sim[int(k)]:.2f})"
                        for k in pol_order], fontsize=7)
    ax.set_xlabel(r"$\hat\iota_{k,j}$ — mean & 2.5–97.5% replicate band")
    n_active = int((agg[(agg["j"] == j) & (agg["gt_active"] == 1)]
                     ["k"].nunique()))
    ax.set_title(f"{COV_LABELS[j]}  (GT act. on {n_active})", fontsize=10)
    vmax = max(1.5, np.abs(agg[agg["j"] == j]["iota_sim"]).max() * 1.25)
    ax.set_xlim(-vmax, vmax)

# Hide the empty 6th subplot
axes[-1].axis('off')

fig.suptitle(f"$\\hat\\iota_{{k,j}}$ across 20 Monte Carlo replicates "
              "(20 Poisson seeds, fixed truth)\n"
              "Topics ordered by polarization (most at top); "
              "% to right of bar = fraction of reps whose HPD$_{0.05}$ covers $\\iota^{sim}$",
              fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "replicate_forest.png"),
             dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  [fig] {OUT_DIR}/replicate_forest.png")


# ================================================================== #
# Figure 2: HPD-coverage heatmap
# ================================================================== #
cov_grid = np.zeros((K, L))
for _, r in agg.iterrows():
    cov_grid[int(r["k"]), int(r["j"])] = float(r["hpd_coverage"])

fig, ax = plt.subplots(figsize=(6.5, 8.5))
im = ax.imshow(cov_grid[pol_order], aspect="auto", cmap="RdYlGn",
                vmin=0, vmax=1)
ax.set_xticks(range(L))
ax.set_xticklabels(COV_LABELS, rotation=30, ha="right")
ax.set_yticks(range(K))
ax.set_yticklabels([f"{int(k)} (pol={pol_sim[int(k)]:.2f})"
                    for k in pol_order], fontsize=7)
ax.set_title(f"HPD$_{{{ALPHA}}}$ coverage of $\\iota^{{sim}}$ "
              "across 20 replicates")

# annotate
gt_active_grid = (iota_sim_full != 0).astype(int)
for k in range(K):
    for j in range(L):
        i = row_of_k[k]
        v = cov_grid[k, j]
        active = gt_active_grid[k, j]
        ax.text(j, i, f"{v*100:.0f}",
                ha='center', va='center', fontsize=7,
                color='black' if 0.30 < v < 0.85 else 'white',
                fontweight='bold' if active else 'normal')

cb = fig.colorbar(im, ax=ax, fraction=0.04)
cb.set_label("Coverage (target = 95%)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "hpd_coverage_heatmap.png"),
             dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  [fig] {OUT_DIR}/hpd_coverage_heatmap.png")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
