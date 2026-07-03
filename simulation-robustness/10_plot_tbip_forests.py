#!/usr/bin/env python3
"""
10_plot_tbip_forests.py
========================
Two visualisations per TBIP-family replicate:

(A) Forest plot in the same 2x3 panel layout as 07c_iota_ccp.py and
    10_plot_constIP_forests.py: one panel per covariate, y-axis lists
    topic indices 0..24, with the topic-varying STBS estimates
    iota_hat[k, j] +/- HPD_0.05 plotted per-cell. The TBIP truth is a
    single horizontal vertical line at iota_sim_vec[j] (since the
    truth is constant across topics). This makes the over-specification
    pathology visible: any spread of iota_hat_kj around iota_sim_vec[j]
    is a measurement artefact of the topic-varying fit.

(B) Correlation scatter of (i) the polarisation-weighted aggregate
    x_hat^pol vs the constant truth x_sim_a, plus (ii) per-topic
    scatter of x_hat[:, k] vs x_sim_a in a small multiples grid.

For each of the 20 sim_tbip_NN/ fits:
  - results_simulation/sim_tbip_NN/iota_tbip_forest.png
  - results_simulation/sim_tbip_NN/ideal_tbip_scatter.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.abspath(__file__))
RES_BASE = os.path.join(REPO, "results_simulation")
DATA_BASE = os.path.join(REPO, "data_simulation")

COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]
N_SIMS = 20
ALPHA = 0.05
Z_CRIT = norm.ppf(1.0 - ALPHA / 2.0)


def signif_code(z_abs):
    p = 2.0 * (1.0 - norm.cdf(z_abs))
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""


def _hungarian_align(eta_sim, eta_hat):
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
        else 1.0 for k in range(K)
    ])
    return perm, signs


def plot_forest(rep_idx):
    fit_dir = os.path.join(RES_BASE, f"sim_tbip_{rep_idx:02d}")
    gt_dir = os.path.join(DATA_BASE, f"sim_tbip_{rep_idx:02d}", "ground_truth")

    iota_hat = np.load(os.path.join(fit_dir, "params",
                                    "iota_location_final.npy"))      # (K, J)
    eta_hat = np.load(os.path.join(fit_dir, "params",
                                   "eta_location_final.npy"))
    tril = np.load(os.path.join(fit_dir, "params",
                                "iota_scale_tril_final.npy"))         # (J, J)
    iota_sim_vec = np.load(os.path.join(gt_dir, "iota_sim_vec.npy"))  # (J,)
    eta_sim = np.load(os.path.join(gt_dir, "eta.npy"))
    K, J = iota_hat.shape

    perm, signs = _hungarian_align(eta_sim, eta_hat)
    iota_hat_aligned = np.stack(
        [iota_hat[perm[k]] * signs[k] for k in range(K)]
    )

    # Joint sign flip if needed -- align so c1 (truth +0.30) is positive
    if iota_sim_vec[1] != 0:
        if np.sign(iota_hat_aligned[:, 1].mean()) != np.sign(iota_sim_vec[1]):
            iota_hat_aligned = -iota_hat_aligned

    cov = tril @ tril.T
    sd = np.sqrt(np.diag(cov))   # (J,) shared across k

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for j in range(J):
        ax = axes[j]
        truth_const = float(iota_sim_vec[j])
        # Truth horizontal vertical line (constant across k)
        ax.axvline(truth_const, color='black', lw=1.4, ls='-',
                   zorder=2, label=f'TBIP truth = {truth_const:+.3f}')
        for k in range(K):
            mu = float(iota_hat_aligned[k, j])
            lo = mu - Z_CRIT * sd[j]
            hi = mu + Z_CRIT * sd[j]
            detected = (abs(mu) >= Z_CRIT * sd[j])
            color = "tab:green" if detected else "lightgray"
            face = color if truth_const != 0 else "white"
            ax.errorbar(mu, k, xerr=[[mu - lo], [hi - mu]],
                        fmt='o', color=color, markerfacecolor=face,
                        markeredgecolor=color, markersize=6,
                        capsize=2, elinewidth=0.7, zorder=3)
            z_abs = abs(mu) / max(sd[j], 1e-12)
            code = signif_code(z_abs)
            if code:
                ax.text(hi + 0.02, k, code, fontsize=7, color=color,
                        va='center', fontweight='bold')

        ax.axvline(0, color="black", ls="--", lw=0.6, alpha=0.5)
        ax.set_yticks(range(K)); ax.invert_yaxis()
        ax.set_yticklabels([str(k) for k in range(K)], fontsize=7)
        ax.set_xlabel(r"$\hat\iota_{k,j}$ $\pm z_{\alpha/2}\sigma_j$ "
                      "(HPD$_{0.05}$)", fontsize=9)
        n_active_truth = 0 if truth_const == 0 else K
        ax.set_title(f"{COV_LABELS[j]}  (TBIP truth: {truth_const:+.3f}, "
                     f"active on {n_active_truth} topics)",
                     fontsize=10)
        vmax = max(1.5, abs(truth_const) * 1.4,
                   np.abs(iota_hat_aligned[:, j]).max() * 1.25)
        ax.set_xlim(-vmax, vmax)
        ax.legend(fontsize=7, loc='lower right', framealpha=0.9)

    axes[J].axis("off")
    fig.suptitle(f"TBIP-family fit, replicate {rep_idx:02d}: "
                 r"$\hat\iota_{k,j}$ from STBS-with-$x_{a,k}$ vs. "
                 "topic-constant TBIP truth (black line)  |  "
                 "*** < 0.001, ** < 0.01, * < 0.05, . < 0.1",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(fit_dir, "iota_tbip_forest.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_scatter(rep_idx):
    fit_dir = os.path.join(RES_BASE, f"sim_tbip_{rep_idx:02d}")
    gt_dir = os.path.join(DATA_BASE, f"sim_tbip_{rep_idx:02d}", "ground_truth")

    x_hat = np.load(os.path.join(fit_dir, "params",
                                 "ideal_point_location_final.npy"))   # (N, K)
    x_sim = np.load(os.path.join(gt_dir, "ideal_sim_vec.npy"))         # (N,)
    eta_sim = np.load(os.path.join(gt_dir, "eta.npy"))
    eta_hat = np.load(os.path.join(fit_dir, "params",
                                   "eta_location_final.npy"))
    N, K = x_hat.shape

    perm, signs = _hungarian_align(eta_sim, eta_hat)
    x_hat_aligned = np.stack(
        [x_hat[:, perm[k]] * signs[k] for k in range(K)], axis=1
    )

    # Polarisation-weighted aggregate
    pol_w = x_hat_aligned.std(axis=0)
    pol_w = pol_w / pol_w.sum() if pol_w.sum() > 0 else np.ones(K) / K
    x_hat_polmean = (x_hat_aligned * pol_w[None, :]).sum(axis=1)

    # Joint sign flip if needed
    cor_raw = np.corrcoef(x_hat_polmean, x_sim)[0, 1]
    if cor_raw < 0:
        x_hat_polmean = -x_hat_polmean
        x_hat_aligned = -x_hat_aligned

    fig = plt.figure(figsize=(14, 8))
    # Big scatter on the left: polarisation-weighted aggregate vs truth
    ax_main = fig.add_axes([0.06, 0.10, 0.32, 0.80])
    ax_main.scatter(x_sim, x_hat_polmean, s=30, alpha=0.7,
                    color="tab:blue", edgecolor="black")
    lim = max(np.abs(x_sim).max(), np.abs(x_hat_polmean).max()) * 1.1
    ax_main.plot([-lim, lim], [-lim, lim], 'k--', lw=0.7, alpha=0.5)
    ax_main.set_xlim(-lim, lim); ax_main.set_ylim(-lim, lim)
    ax_main.set_xlabel(r"TBIP truth $x^{sim}_a$ (constant across $k$)")
    ax_main.set_ylabel(r"STBS aggregate $\hat x_a$ "
                       "(polarisation-weighted)")
    cor_pol = np.corrcoef(x_hat_polmean, x_sim)[0, 1]
    ax_main.set_title(f"Aggregate IP recovery   (cor = {cor_pol:+.3f})",
                      fontsize=11)
    ax_main.grid(alpha=0.3)

    # Small multiples on the right: per-topic scatter (5x5 = 25 panels)
    n_rows, n_cols = 5, 5
    grid_left, grid_bot = 0.43, 0.06
    grid_w, grid_h = 0.55, 0.86
    cell_w = grid_w / n_cols
    cell_h = grid_h / n_rows
    for k in range(K):
        rr = k // n_cols; cc = k % n_cols
        ax = fig.add_axes([
            grid_left + cc * cell_w + 0.005,
            grid_bot + (n_rows - 1 - rr) * cell_h + 0.005,
            cell_w - 0.012, cell_h - 0.012,
        ])
        ax.scatter(x_sim, x_hat_aligned[:, k], s=8, alpha=0.6,
                   color="tab:orange", edgecolor="none")
        lim_k = max(np.abs(x_sim).max(),
                    np.abs(x_hat_aligned[:, k]).max()) * 1.1
        ax.plot([-lim_k, lim_k], [-lim_k, lim_k], 'k--', lw=0.5, alpha=0.4)
        ax.set_xlim(-lim_k, lim_k); ax.set_ylim(-lim_k, lim_k)
        cor_k = np.corrcoef(x_hat_aligned[:, k], x_sim)[0, 1]
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"k={k}  r={cor_k:+.2f}", fontsize=7, pad=2)

    fig.suptitle(f"TBIP-family replicate {rep_idx:02d}: ideal-point "
                 "recovery   |   left: aggregate vs constant truth   "
                 "|   right: per-topic scatter",
                 fontsize=11)
    out_png = os.path.join(fit_dir, "ideal_tbip_scatter.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    forests, scatters = 0, 0
    for r in range(1, N_SIMS + 1):
        fit_dir = os.path.join(RES_BASE, f"sim_tbip_{r:02d}")
        if not os.path.exists(os.path.join(fit_dir, "params",
                                           "iota_location_final.npy")):
            print(f"  skip rep {r:02d}: fit not yet complete")
            continue
        print(f"  plotting rep {r:02d} ...")
        try:
            png = plot_forest(r); print(f"    forest:  {png}"); forests += 1
        except Exception as e:
            print(f"    forest FAILED: {e}")
        try:
            png = plot_scatter(r); print(f"    scatter: {png}"); scatters += 1
        except Exception as e:
            print(f"    scatter FAILED: {e}")
    print(f"\nGenerated {forests} forests + {scatters} scatters of "
          f"{N_SIMS} possible.")


if __name__ == "__main__":
    main()
