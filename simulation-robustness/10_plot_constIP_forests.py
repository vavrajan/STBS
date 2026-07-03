#!/usr/bin/env python3
"""
10_plot_constIP_forests.py
==========================
Per-replicate forest plot for the misspecified constant-IP fits, in
the SAME visual style as the topic-varying CCP forest plots produced
by 07c_iota_ccp.py (so the reader does not have to re-orient).

Layout (mirrors §A appendix plots):
  2x3 panel grid (one panel per covariate, 1 empty), figsize 16x9.
  y-axis = topic index 0..24 (25 rows per panel).
  Black 'x' = GT iota_sim_kj for each topic (one per row).
  Coloured horizontal HPD band = the SINGLE iota_hat_l, replicated
    across all 25 topic rows since the constant-iota model returns
    one estimate per cov, shared across topics. The band visually
    encodes "this is the same number for every topic".
  Colours:
    green  = detected (HPD_0.05 excludes 0)  - HPD band coloured
    grey   = not detected (HPD covers 0)
  Significance code at right edge: *** < 0.001, ** < 0.01, * < 0.05, . < 0.1.
  Vertical dashed line at iota = 0.
  Per-panel title: covariate label + count of GT-active topics.
  Suptitle: "CCP-style summary, constant-IP misspecified fit ..."

For each of the 20 sim_dgp_NN_constIP/ fits:
  -> writes results_simulation/sim_dgp_NN_constIP/iota_constIP_forest.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

REPO = os.path.dirname(os.path.abspath(__file__))
RES_BASE = os.path.join(REPO, "results_simulation")
DATA_BASE = os.path.join(REPO, "data_simulation")

COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]
N_SIMS = 20
ALPHA = 0.05
Z_CRIT = norm.ppf(1.0 - ALPHA / 2.0)   # 1.96


def signif_code(z_abs):
    """Two-sided p-value of |z| under N(0,1) -> code."""
    p = 2.0 * (1.0 - norm.cdf(z_abs))
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""


def plot_one(rep_idx):
    fit_dir = os.path.join(RES_BASE, f"sim_dgp_{rep_idx:02d}_constIP")
    gt_dir = os.path.join(DATA_BASE, f"sim_dgp_{rep_idx:02d}", "ground_truth")

    iota_hat = np.load(os.path.join(fit_dir, "params",
                                    "iota_location_final.npy")).ravel()  # (J,)
    tril = np.load(os.path.join(fit_dir, "params",
                                "iota_scale_tril_final.npy"))            # (J, J)
    cov = tril @ tril.T
    sd = np.sqrt(np.diag(cov))                                            # (J,)

    iota_sim = np.load(os.path.join(gt_dir, "iota_sim.npy"))             # (K, J)
    K, J = iota_sim.shape

    # Sign-align (x_hat, iota_hat) jointly via topic-mean alignment, so
    # the panels are visually consistent across reps.
    iota_sim_mean = iota_sim.mean(axis=0)
    # Cheap proxy: align sign of c1 (which has uniform truth +0.30); if
    # c1's truth is non-zero and the fit's c1 has the wrong sign, flip.
    if iota_sim_mean[1] != 0 and iota_hat[1] != 0:
        if np.sign(iota_hat[1]) != np.sign(iota_sim_mean[1]):
            iota_hat = -iota_hat

    # ------ figure setup, mirrors 07c_iota_ccp.forest_plot --------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for j in range(J):
        ax = axes[j]
        mu = float(iota_hat[j])
        sigma = float(sd[j])
        lo = mu - Z_CRIT * sigma
        hi = mu + Z_CRIT * sigma
        z_abs = abs(mu) / max(sigma, 1e-12)
        detected = (abs(mu) >= Z_CRIT * sigma)
        code = signif_code(z_abs)

        # Colour: green if detected, grey if not
        color = "tab:green" if detected else "lightgray"

        # Per-row visualisation: coloured horizontal HPD band spanning
        # all 25 topic rows. The mean is plotted as a marker per row
        # so the reader sees "it's the same value at every topic".
        rows = np.arange(K)
        # HPD shaded band
        ax.fill_betweenx(rows, lo, hi, color=color, alpha=0.18, zorder=1)
        # Vertical line at the point estimate
        ax.axvline(mu, color=color, lw=1.2, alpha=0.85, zorder=2)
        # Markers per row (faint), to mimic the per-cell forest style
        ax.scatter(np.full(K, mu), rows, marker='o', s=22,
                   color=color, edgecolor=color, zorder=3, alpha=0.85)
        # HPD whiskers per row (very light)
        for k in rows:
            ax.plot([lo, hi], [k, k], color=color, lw=0.8,
                    alpha=0.55, zorder=2)

        # GT topic-specific iota_sim_kj as black 'x'
        n_act = int((iota_sim[:, j] != 0).sum())
        for k in range(K):
            gt_v = float(iota_sim[k, j])
            if gt_v != 0:
                ax.scatter(gt_v, k, marker='x', color='black',
                           s=40, zorder=5)

        # Significance code at the right edge of the HPD band
        if code:
            ax.text(hi + 0.04, K * 0.5, code, fontsize=11, color=color,
                    va='center', fontweight='bold')

        ax.axvline(0, color="black", ls="--", lw=0.6, alpha=0.5)
        ax.set_yticks(range(K)); ax.invert_yaxis()
        ax.set_yticklabels([str(int(k)) for k in range(K)], fontsize=7)
        ax.set_xlabel(r"$\hat\iota_l$ (constant across $k$) $\pm z_{\alpha/2}\,\sigma_l$ (HPD$_{\alpha}$)",
                      fontsize=9)
        ax.set_title(f"{COV_LABELS[j]}  (GT act. on {n_act} topics,"
                     f" $\\hat\\iota_l = {mu:+.3f}$,"
                     f" $\\sigma_l = {sigma:.3f}$)",
                     fontsize=10)
        vmax = max(1.5, np.abs(iota_sim[:, j]).max() * 1.25, abs(hi) * 1.1)
        ax.set_xlim(-vmax, vmax)

    # Hide the empty 6th panel
    axes[J].axis("off")

    fig.suptitle("CCP-style summary, constant-IP misspecified fit "
                 f"(replicate {rep_idx:02d})  |  "
                 r"$\hat\iota_l$ HPD$_{0.05}$ band shown across all $k$ "
                 "(model returns one $\\iota$ per cov, shared)  |  "
                 "GT $\\iota^{sim}_{k,j}$ as black $\\times$  |  "
                 "(*** < 0.001, ** < 0.01, * < 0.05, . < 0.1)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(fit_dir, "iota_constIP_forest.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    made = []
    for r in range(1, N_SIMS + 1):
        fit_dir = os.path.join(RES_BASE, f"sim_dgp_{r:02d}_constIP")
        if not os.path.exists(os.path.join(fit_dir, "params",
                                           "iota_location_final.npy")):
            print(f"  skip rep {r:02d}: fit not yet complete")
            continue
        png = plot_one(r)
        print(f"  saved {png}")
        made.append(png)
    print(f"\nGenerated {len(made)}/{N_SIMS} forest plots.")


if __name__ == "__main__":
    main()
