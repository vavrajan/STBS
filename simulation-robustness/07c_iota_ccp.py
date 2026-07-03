#!/usr/bin/env python3
"""
07c_iota_ccp.py
===============
Inference on the STBS regression coefficients iota using the paper's
Complementary Coverage Probability (CCP) methodology. Mirrors the
CCPvalue() routine from ../Revision_code_CAVI/05_run_R_plots.R:

    CCPvalue(C, mu, Sigma) =
        pchisq( (C mu)' (C Sigma C')^{-1} (C mu), df=nrow(C), lower.tail=F )

For the single-coefficient test (C = e_j) this reduces to
    chi^2 = (mu_{k,j} / sigma_j)^2
    CCP   = P(chi^2_1 > chi^2)
with sigma_j = sqrt( Sigma_jj ) = sqrt( sum_l T_{j,l}^2 ).

Workflow:
  1. Load fit parameters (iota_location_final.npy, iota_scale_tril_final.npy,
     beta, ideal)  and ground truth (iota_sim, ideal_sim, effect_pattern).
  2. Align topics to the GT via Hungarian matching on
     |cor(ideal_sim[:, k_gt], ideal_hat[:, k_hat])|.
  3. Sign-align each topic using sign( cor(ideal_sim, ideal_hat_aligned) ).
  4. Compute CCP p-value for every (k, j) using the single-coef formula.
  5. Classify every entry into TP / FP / FN / TN against CCP < alpha.
  6. Write CSV + forest-plot PNG + CCP-bin-summary table.

Outputs (under --out-dir, default is --fit-dir):
  iota_ccp_table.csv            row per (k, j) with mu, sigma, chi2, CCP,
                                 signif code, HPD bounds, classification
  iota_ccp_summary.csv          counts per covariate x CCP bin
  iota_recovery_forest_ccp.png  forest plot (HPD_alpha intervals
                                 + CCP significance codes)
  iota_ccp_meta.json            scalar diagnostics

Usage:
  python3 07c_iota_ccp.py \
      --fit-dir    results_simulation/24_04_26_simdata_fit \
      --gt-dir     data_simulation/24_04_26_simdata/ground_truth \
      --out-dir    results_simulation/24_04_26_simdata_fit \
      --alpha      0.05
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# CCP significance bins and codes (as in the paper / R pipeline)
CCP_BREAKS = [0.001, 0.01, 0.05, 0.10]
CCP_CODES = ["***", "**", "*", "."]


# ================================================================== #
# Core CCP routines (mirror of 05_run_R_plots.R: CCPvalue, EstSECCP)
# ================================================================== #

def ccp_value(C, mu, Sigma):
    """General CCP for a linear contrast matrix C (m, L) on iota_k ~ N(mu, Sigma).

    mu: (L,)          posterior mean of iota_k (or a matrix of means if joint across k -- use one k at a time)
    Sigma: (L, L)     posterior covariance of iota (shared in STBS)
    C: (m, L)         contrast matrix

    Returns: p-value  P(chi^2_m > chi2)
    """
    C = np.atleast_2d(C)
    mu = np.atleast_1d(mu)
    m = C.shape[0]
    Cmu = C @ mu                  # (m,)
    CSigmaC = C @ Sigma @ C.T     # (m, m)
    # Mahalanobis-like quadratic form
    x = np.linalg.solve(CSigmaC, Cmu)
    chi_stat = float(Cmu @ x)
    return 1.0 - chi2.cdf(chi_stat, df=m), chi_stat


def ccp_single_coef(mu, sigma):
    """Shortcut for C = e_j: scalar inputs mu, sigma -> (CCP, chi2)."""
    chi_stat = (mu / sigma) ** 2
    return 1.0 - chi2.cdf(chi_stat, df=1), chi_stat


def ccp_code(p):
    """Map a CCP value to the paper's significance code."""
    for b, c in zip(CCP_BREAKS, CCP_CODES):
        if p < b:
            return c
    return ""


# ================================================================== #
# Loading helpers
# ================================================================== #

def load_fit(fit_dir):
    pdir = os.path.join(fit_dir, "params")
    iota_loc = np.load(os.path.join(pdir, "iota_location_final.npy"))
    iota_tril = np.load(os.path.join(pdir, "iota_scale_tril_final.npy"))
    beta_shp = np.load(os.path.join(pdir, "beta_shape_final.npy"))
    beta_rte = np.load(os.path.join(pdir, "beta_rate_final.npy"))
    beta = beta_shp / beta_rte
    ideal = np.load(os.path.join(pdir, "ideal_point_location_final.npy"))
    return dict(iota_loc=iota_loc, iota_tril=iota_tril,
                beta=beta, ideal=ideal)


def load_ground_truth(gt_dir):
    iota_sim = np.load(os.path.join(gt_dir, "iota_sim.npy"))
    pattern = np.load(os.path.join(gt_dir, "effect_pattern.npy"))
    ideal_sim_path = os.path.join(gt_dir, "ideal_sim.npy")
    if not os.path.exists(ideal_sim_path):
        ideal_sim_path = os.path.join(gt_dir, "ideal_points_sim.npy")
    ideal_sim = np.load(ideal_sim_path)
    eta_sim_path = os.path.join(gt_dir, "eta.npy")
    eta_sim = np.load(eta_sim_path) if os.path.exists(eta_sim_path) else None
    return dict(iota_sim=iota_sim, pattern=pattern, ideal_sim=ideal_sim,
                eta_sim=eta_sim)


def polarization_order(ideal_sim, eta_sim):
    """Return topic ordering (most → least polarizing) using the centered
    product amplitude pol_k = std(ideal_sim[:, k]) * std(eta_sim[k, :]).

    If eta_sim is None, falls back to std(ideal_sim[:, k]) alone.
    """
    K = ideal_sim.shape[1]
    if eta_sim is None:
        pol = np.array([ideal_sim[:, k].std() for k in range(K)])
    else:
        pol = np.array([ideal_sim[:, k].std() * eta_sim[k].std() for k in range(K)])
    order = np.argsort(-pol)   # descending
    return order.tolist(), pol


# ================================================================== #
# Topic alignment (by |cor(ideal)|) and sign alignment
# ================================================================== #

def hungarian_align(ideal_sim, ideal_hat):
    """Return matching gt_topic_k -> fit_topic_j that maximises sum
    of |cor(ideal_sim[:, k], ideal_hat[:, j])| over 25 topics."""
    K = ideal_sim.shape[1]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            r = np.corrcoef(ideal_sim[:, i], ideal_hat[:, j])[0, 1]
            cost[i, j] = abs(r) if not np.isnan(r) else 0.0
    row, col = linear_sum_assignment(-cost)
    return {int(r): int(c) for r, c in zip(row, col)}, cost


def sign_align(ideal_sim, ideal_hat_aligned):
    """Per-topic sign s_k = sign( cor(ideal_sim[:, k], ideal_hat_a[:, k]) )."""
    K = ideal_sim.shape[1]
    s = np.zeros(K)
    for k in range(K):
        r = np.corrcoef(ideal_sim[:, k], ideal_hat_aligned[:, k])[0, 1]
        s[k] = 1.0 if (np.isnan(r) or r >= 0) else -1.0
    return s


# ================================================================== #
# Main CCP table
# ================================================================== #

def build_ccp_table(iota_sim, pattern, iota_hat_signed, Sigma, alpha):
    K, L = iota_hat_signed.shape
    sigma_j = np.sqrt(np.diag(Sigma))            # (L,)
    z = chi2.ppf(1.0 - alpha, df=1) ** 0.5       # quantile for HPD_alpha

    rows = []
    for k in range(K):
        for j in range(L):
            mu = iota_hat_signed[k, j]
            s_j = sigma_j[j]
            p, c2 = ccp_single_coef(mu, s_j)
            code = ccp_code(p)
            lo = mu - z * s_j
            hi = mu + z * s_j
            sig = (lo > 0) or (hi < 0)
            gt_active = bool(pattern[k, j])
            if sig and gt_active:      cls = "TP"
            elif sig and not gt_active: cls = "FP"
            elif not sig and gt_active: cls = "FN"
            else:                      cls = "TN"
            rows.append(dict(
                k=k, j=j,
                iota_sim=float(iota_sim[k, j]),
                gt_active=int(gt_active),
                iota_hat=float(mu),
                sigma_j=float(s_j),
                chi2=float(c2),
                CCP=float(p),
                signif=code,
                hpd_lo=float(lo),
                hpd_hi=float(hi),
                classification=cls,
            ))
    return pd.DataFrame(rows), sigma_j, z


def bin_summary(df, cov_labels, alpha):
    L = len(cov_labels)
    K = df["k"].max() + 1
    rows = []
    for j in range(L):
        sub = df[df["j"] == j]
        counts = {c: int((sub["signif"] == c).sum()) for c in CCP_CODES}
        counts["ns"] = int((sub["signif"] == "").sum())
        rows.append(dict(
            j=j, covariate=cov_labels[j],
            gt_active=int(sub["gt_active"].sum()),
            **counts,
        ))
    return pd.DataFrame(rows)


# ================================================================== #
# Forest plot
# ================================================================== #

def forest_plot(df, sigma_j, alpha, cov_labels, gt, out_path,
                topic_order=None, pol=None):
    """Forest plot with topics ordered top-to-bottom by polarization
    (most polarizing at the top) when topic_order is provided.
    """
    K = int(df["k"].max()) + 1
    L = len(cov_labels)
    z = chi2.ppf(1.0 - alpha, df=1) ** 0.5

    if topic_order is None:
        topic_order = list(range(K))
    # row position of each topic k in the plot (0 = top)
    row_of_k = {int(k): i for i, k in enumerate(topic_order)}

    # Dynamic grid: up to 3 columns, as many rows as needed.
    # For L=5 covariates this gives 2x3 with one empty cell, which we
    # then hide so the figure shows exactly L panels.
    ncols = min(L, 3)
    nrows = int(np.ceil(L / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.3, nrows * 4.5))
    axes = np.atleast_1d(axes).ravel()

    for j in range(L):
        ax = axes[j]
        sub = df[df["j"] == j].reset_index(drop=True)
        for _, r in sub.iterrows():
            k = int(r["k"])
            y = row_of_k[k]            # reordered row position
            mu = float(r["iota_hat"])
            lo, hi = float(r["hpd_lo"]), float(r["hpd_hi"])
            gt_a = bool(r["gt_active"])
            color = {"TP": "tab:green", "FP": "tab:red",
                     "FN": "tab:orange", "TN": "lightgray"}[r["classification"]]
            face = color if gt_a else "white"
            ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]],
                        fmt='o', color=color, markerfacecolor=face,
                        markeredgecolor=color, markersize=6,
                        capsize=2, elinewidth=0.7)
            gt_val = float(r["iota_sim"])
            if gt_val != 0:
                ax.scatter(gt_val, y, marker='x', color='black', s=40, zorder=5)
            code = r["signif"]
            if code:
                ax.text(hi + 0.02, y, code, fontsize=8, color=color,
                        va='center', fontweight='bold')
        ax.axvline(0, color="black", ls="--", lw=0.6, alpha=0.5)
        ax.set_yticks(range(K)); ax.invert_yaxis()
        # Y-axis labels: topic index (optionally with rank prefix)
        if pol is not None:
            ax.set_yticklabels([f"{int(k)}  (pol={pol[int(k)]:.2f})"
                                for k in topic_order], fontsize=7)
        else:
            ax.set_yticklabels([str(int(k)) for k in topic_order], fontsize=7)
        ax.set_xlabel(r"$\hat\iota$ (point) $\pm z_{\alpha/2}\,\sigma_j$ (HPD$_{\alpha}$)")
        ax.set_title(f"{cov_labels[j]}  (GT act. on {int(gt['pattern'][:, j].sum())},"
                     f"  $\\sigma_j = {sigma_j[j]:.3f}$)", fontsize=10)
        vmax = max(1.5, np.abs(gt["iota_sim"][:, j]).max() * 1.25)
        ax.set_xlim(-vmax, vmax)

    # Hide any unused axes (e.g. 6th panel when L=5 in a 2x3 grid).
    # The TP / FP / FN / TN classification counts are NOT shown here;
    # they are aggregated across all 20 replicates by a separate
    # aggregator script (see 08d_ccp_classification_summary.py).
    for j in range(L, len(axes)):
        axes[j].set_visible(False)

    title_extra = ("  |  topics ordered by polarization "
                   r"$\mathrm{pol}_k = \sigma(x_k)\cdot\sigma(\eta_k)$, "
                   "most polarizing at top") if topic_order != list(range(K)) else ""
    fig.suptitle(f"CCP inference on $\\iota$ (alpha = {alpha}): "
                 f"HPD$_{{{alpha}}}$ intervals + significance codes "
                 "(*** < 0.001, ** < 0.01, * < 0.05, . < 0.1)" + title_extra,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================== #
# MAIN
# ================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True,
                    help="STBS result directory containing params/")
    ap.add_argument("--gt-dir", required=True,
                    help="Ground-truth directory with iota_sim.npy, "
                         "ideal_sim.npy (or ideal_points_sim.npy), "
                         "and effect_pattern.npy")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write outputs (default: --fit-dir)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level for HPD intervals and "
                         "TP/FP/FN/TN classification (default 0.05)")
    ap.add_argument("--cov-labels", default=None,
                    help="Optional comma-separated covariate labels")
    args = ap.parse_args()

    out_dir = args.out_dir or args.fit_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  CCP inference on iota")
    print("=" * 60)
    print(f"  fit-dir  : {args.fit_dir}")
    print(f"  gt-dir   : {args.gt_dir}")
    print(f"  out-dir  : {out_dir}")
    print(f"  alpha    : {args.alpha}")

    fit = load_fit(args.fit_dir)
    gt = load_ground_truth(args.gt_dir)

    K, L = fit["iota_loc"].shape
    if args.cov_labels is not None:
        cov_labels = [s.strip() for s in args.cov_labels.split(",")]
        assert len(cov_labels) == L
    else:
        cov_labels = [f"c{j}" for j in range(L)]

    # Alignment
    matching, _ = hungarian_align(gt["ideal_sim"], fit["ideal"])
    iota_hat_a = np.stack([fit["iota_loc"][matching[k]] for k in range(K)])
    ideal_hat_a = np.stack([fit["ideal"][:, matching[k]] for k in range(K)], axis=1)
    signs = sign_align(gt["ideal_sim"], ideal_hat_a)
    iota_hat_s = iota_hat_a * signs[:, None]

    # Posterior covariance (shared across k)
    T = fit["iota_tril"]
    Sigma = T @ T.T

    # Build CCP table
    df, sigma_j, z = build_ccp_table(gt["iota_sim"], gt["pattern"],
                                     iota_hat_s, Sigma, args.alpha)
    df.to_csv(os.path.join(out_dir, "iota_ccp_table.csv"), index=False)
    print(f"\n  marginal sigma_j = {np.round(sigma_j, 4).tolist()}")
    print(f"  total (K, L) entries : {len(df)}")

    # Classification counts
    cls_counts = df["classification"].value_counts().to_dict()
    for c in ("TP", "FP", "FN", "TN"):
        cls_counts.setdefault(c, 0)
    print(f"\n  Classification at alpha = {args.alpha}:")
    for c in ("TP", "FP", "FN", "TN"):
        print(f"    {c}: {cls_counts[c]}")

    # Summary by covariate
    summary = bin_summary(df, cov_labels, args.alpha)
    summary.to_csv(os.path.join(out_dir, "iota_ccp_summary.csv"), index=False)
    print("\n  CCP-bin summary per covariate:")
    print(summary.to_string(index=False))

    # Compute polarization for the log only; do NOT use it to sort.
    _, pol = polarization_order(gt["ideal_sim"], gt.get("eta_sim"))
    print("\n  Polarization per topic (pol_k = std(x_k)*std(eta_k); "
          "natural topic order):")
    for k in range(int(gt["iota_sim"].shape[0])):
        print(f"    k={k:2d}  pol={pol[k]:.3f}")

    # Plot — topics in natural order 0..K-1 on the y-axis
    out_png = os.path.join(out_dir, "iota_recovery_forest_ccp.png")
    forest_plot(df, sigma_j, args.alpha, cov_labels, gt, out_png,
                topic_order=None, pol=None)
    print(f"\n  [fig] {out_png}")

    # Meta
    meta = dict(
        alpha=args.alpha,
        sigma_j=sigma_j.tolist(),
        z_alpha_over_2=float(z),
        counts=cls_counts,
        fit_dir=args.fit_dir,
        gt_dir=args.gt_dir,
    )
    with open(os.path.join(out_dir, "iota_ccp_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  [meta] {out_dir}/iota_ccp_meta.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
