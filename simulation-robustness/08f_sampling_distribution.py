#!/usr/bin/env python3
"""
08f_sampling_distribution.py
============================
Empirical sampling distribution of the variational posterior mean
hat_iota_{k,j} across the 20 Fixed-truth Monte Carlo replicates
(sim_01 ... sim_20). All 20 runs share an identical data-generating
process (same iota_sim, X, active patterns, theta, beta, eta);
only the Poisson seed varies. The cell-wise spread of hat_iota across
the 20 replicates is therefore a Monte-Carlo estimate of the true
sampling distribution of the STBS-CAVI point estimator under Poisson
noise --- a frequentist alternative to the variational HPD width that
we showed in 08e_coverage_summary.py to be miscalibrated.

For every cell (k, j) we compute:
  - the 20 hat_iota values (one per replicate, Hungarian + sign-fixed
    upstream by 07c_iota_ccp.py)
  - the empirical mean and standard deviation across replicates
  - the empirical 2.5%/97.5% quantiles (the empirical 95% sampling
    interval)
  - whether the truth iota_sim_{k,j} lies inside [Q_2.5, Q_97.5]

We then aggregate over the K*L = 25*5 = 125 cells.

Outputs:
  results_simulation/centered_replicate_summary/
    sampling_distribution.csv               (one row per (k,j))
    sampling_coverage_table.tex             (headline table)
    sampling_boxplots.pdf, .png             (5-panel forest-style boxplot)

Run:
    python3 08f_sampling_distribution.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(REPO, "results_simulation")
DATA = os.path.join(REPO, "data_simulation")
GT_DIR  = os.path.join(DATA, "simdata_centered_design", "ground_truth")
OUT_DIR = os.path.join(RES,  "centered_replicate_summary")
os.makedirs(OUT_DIR, exist_ok=True)

N_REPS = 20
COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]


def main():
    # Load truth (K, L) -- same across all 20 reps
    iota_sim   = np.load(os.path.join(GT_DIR, "iota_sim.npy"))
    pattern    = np.load(os.path.join(GT_DIR, "effect_pattern.npy"))
    K, L = iota_sim.shape

    # Stack hat_iota across 20 replicates from the iota_ccp_table.csv
    # (which already has Hungarian + sign alignment applied upstream).
    hat = np.full((N_REPS, K, L), np.nan, dtype=np.float64)
    for r in range(1, N_REPS + 1):
        csv = os.path.join(RES, f"sim_{r:02d}", "iota_ccp_table.csv")
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            hat[r - 1, int(row["k"]), int(row["j"])] = float(row["iota_hat"])
    assert not np.isnan(hat).any(), "some hat_iota cells missing"

    # Per-cell summary statistics
    rows = []
    for k in range(K):
        for j in range(L):
            vals = hat[:, k, j]
            truth = float(iota_sim[k, j])
            q25, q975 = np.quantile(vals, [0.025, 0.975])
            covered = (q25 <= truth <= q975)
            rows.append(dict(
                k=k, j=j, covariate=COV_LABELS[j],
                gt_active=int(pattern[k, j]),
                iota_sim=truth,
                hat_mean=float(vals.mean()),
                hat_sd=float(vals.std(ddof=1)),
                hat_min=float(vals.min()),
                hat_q25=float(q25),
                hat_median=float(np.median(vals)),
                hat_q975=float(q975),
                hat_max=float(vals.max()),
                empirical_95_covered=int(covered),
            ))
    df_cells = pd.DataFrame(rows)
    df_cells.to_csv(os.path.join(OUT_DIR, "sampling_distribution.csv"),
                    index=False)
    print(f"-> {OUT_DIR}/sampling_distribution.csv  ({len(df_cells)} cells)")

    # ---------------- coverage summary ----------------
    df_act  = df_cells[df_cells["gt_active"] == 1]
    df_null = df_cells[df_cells["gt_active"] == 0]
    cov_act  = float(df_act["empirical_95_covered"].mean())
    cov_null = float(df_null["empirical_95_covered"].mean())
    n_act, n_null = len(df_act), len(df_null)

    print()
    print(f"Active cells  ({n_act}):  empirical 95% sampling-interval contains "
          f"the truth in {cov_act*100:.1f}% of cells")
    print(f"Null  cells   ({n_null}):  empirical 95% sampling-interval contains "
          f"the truth in {cov_null*100:.1f}% of cells")
    print()

    # Mean empirical SD vs. mean variational SD (for comparison)
    # Variational SD: take mean sigma_j from the first replicate's CSV
    sd_var = np.zeros(L)
    df0 = pd.read_csv(os.path.join(RES, "sim_01", "iota_ccp_table.csv"))
    for j in range(L):
        sd_var[j] = df0[df0["j"] == j]["sigma_j"].iloc[0]
    sd_emp_per_cov = df_cells.groupby("j")["hat_sd"].mean().to_numpy()
    inflation = sd_emp_per_cov / sd_var
    print(f"{'covariate':<16} {'emp.SD':>8} {'var.SD':>8} {'ratio':>7}")
    for j in range(L):
        print(f"  {COV_LABELS[j]:<14} {sd_emp_per_cov[j]:>8.4f} "
              f"{sd_var[j]:>8.4f} {inflation[j]:>7.2f}x")
    print()

    # ---------------- LaTeX table ----------------
    tex = []
    tex.append("% Auto-generated by 08f_sampling_distribution.py")
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering\small")
    tex.append(
        r"\caption{Empirical sampling distribution of the variational "
        r"posterior mean $\hat\iota_{k,j}$ across the $N=20$ "
        r"replicates of the Fixed-truth Monte Carlo (identical "
        r"data-generating process, only the Poisson seed varies). For "
        r"every covariate $j$ we report (i) the mean (across the "
        r"$K=25$ topics) of the empirical standard deviation of "
        r"$\hat\iota_{k,j}$ across the $20$ replicates "
        r"(`emp.\ SD'); (ii) the variational posterior SD that STBS "
        r"itself reports (`var.\ SD'); (iii) their ratio --- a value "
        r"$>1$ means STBS underestimates the true sampling spread by "
        r"that factor; (iv) the fraction of cells in which the "
        r"empirical $95\%$ sampling interval "
        r"$[Q_{2.5\%}(\hat\iota), Q_{97.5\%}(\hat\iota)]$ contains the "
        r"simulator's truth $\iota_{k,j}^{\mathrm{sim}}$, separately for "
        r"truly-active cells (`act') and truly-null cells (`null'). "
        r"The empirical sampling interval is a frequentist alternative "
        r"to the variational HPD: it directly captures the spread of "
        r"the point estimator under Poisson resampling without relying "
        r"on the variational posterior's calibration.}")
    tex.append(r"\label{tab:sampling_distribution}")
    tex.append(r"\begin{tabular}{l r r r r r}")
    tex.append(r"\toprule")
    tex.append(r"covariate & emp.\ SD & var.\ SD & ratio & cov.\ "
               r"emp.\ 95\,\% act. & cov.\ emp.\ 95\,\% null \\")
    tex.append(r"\midrule")
    for j in range(L):
        sub = df_cells[df_cells["j"] == j]
        sub_act  = sub[sub["gt_active"] == 1]
        sub_null = sub[sub["gt_active"] == 0]
        ca = (sub_act["empirical_95_covered"].mean()
              if len(sub_act) else float("nan"))
        cn = (sub_null["empirical_95_covered"].mean()
              if len(sub_null) else float("nan"))
        cov_esc = COV_LABELS[j].replace("_", r"\_")
        ca_s = f"{100*ca:.1f}\\,\\%" if not np.isnan(ca) else "--"
        cn_s = f"{100*cn:.1f}\\,\\%" if not np.isnan(cn) else "--"
        tex.append(
            f"{cov_esc} & {sd_emp_per_cov[j]:.4f} & {sd_var[j]:.4f} & "
            f"{inflation[j]:.2f}$\\times$ & {ca_s} & {cn_s} \\\\"
        )
    tex.append(r"\midrule")
    tex.append(
        rf"\textbf{{pooled}} & "
        rf"{sd_emp_per_cov.mean():.4f} & {sd_var.mean():.4f} & "
        rf"{(sd_emp_per_cov/sd_var).mean():.2f}$\times$ & "
        rf"\textbf{{{100*cov_act:.1f}\,\%}} ($n={n_act}$) & "
        rf"\textbf{{{100*cov_null:.1f}\,\%}} ($n={n_null}$) \\"
    )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(os.path.join(OUT_DIR, "sampling_coverage_table.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"-> {OUT_DIR}/sampling_coverage_table.tex")

    # ---------------- 5-panel boxplot (forest-style) ----------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for j in range(L):
        ax = axes[j]
        # Sort topics by truth value for readability
        order = np.argsort(iota_sim[:, j])
        positions = np.arange(K)
        # Box per topic
        data = [hat[:, k, j] for k in order]
        bp = ax.boxplot(data, positions=positions, vert=False, widths=0.6,
                         patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cfe2f3"); patch.set_edgecolor("#3a6ea5")
        # Overlay the truth as a red dot per topic
        truth_sorted = iota_sim[order, j]
        active_sorted = pattern[order, j]
        for i, t in enumerate(truth_sorted):
            color = "red" if active_sorted[i] == 1 else "gray"
            ax.scatter(t, positions[i], color=color, s=40, zorder=5,
                        edgecolor="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.6, ls="--")
        ax.set_xlim(-2.0, 2.0)
        ax.set_yticks(positions)
        ax.set_yticklabels([f"k={int(k)}" for k in order], fontsize=7)
        ax.set_title(f"{COV_LABELS[j]}", fontsize=11)
        ax.set_xlabel(r"$\hat\iota_{k,j}$ across 20 Poisson seeds",
                       fontsize=9)
        ax.grid(alpha=0.3, axis="x")
    axes[L].set_visible(False)
    fig.suptitle("Sampling distribution of $\\hat\\iota_{k,j}$ over the "
                  "20 Fixed-truth replicates (boxplot) versus the "
                  "simulator truth (red = active, gray = null)",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_pdf = os.path.join(OUT_DIR, "sampling_boxplots.pdf")
    out_png = os.path.join(OUT_DIR, "sampling_boxplots.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"-> {out_pdf}")
    print(f"-> {out_png}")


if __name__ == "__main__":
    main()
