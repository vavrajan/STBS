#!/usr/bin/env python3
"""
08e_coverage_summary.py
=======================
Empirical HPD-coverage and bias diagnostics for STBS-CAVI variational
inference, across both Monte Carlo studies in Section 3 of the
simulation reply:

  * Fixed-truth MC  (sim_01 ... sim_20)
  * DGP-family MC   (sim_dgp_01 ... sim_dgp_20)

For every replicate we compute:

  iota_{k,j}:
    - empirical coverage of the variational HPD_{0.95} of the simulator
      truth iota_sim_{k,j}, stratified by covariate and by ground-truth
      activity (active = simulator truth != 0, null = simulator truth = 0)
    - mean HPD width on active and on null cells
    - magnitude-bias slope alpha* = OLS no-intercept slope of
      iota_hat on iota_sim over truly active cells

  x_{a,k}:
    - empirical coverage of the variational Normal HPD_{0.95}
      mu +- 1.96 sigma of x_sim_{a,k}, by topic k and pooled
    - mean HPD width
    - bias slope alpha*_x = OLS no-intercept slope of x_hat on x_sim
      (computed per topic after Hungarian + sign alignment, see
      08_aggregate_centered_replicates.py)

The Hungarian alignment for x_{a,k} reuses the same |corr| objective on
ideal as the existing aggregators (08_aggregate_centered_replicates.py,
08_aggregate_dgp_replicates.py). For iota no alignment is needed (the
columns of iota are indexed by covariate, which is canonical across the
simulator and the fit).

Outputs (per simulation type):
  centered_replicate_summary/coverage_summary.csv     (per-replicate rows)
  centered_replicate_summary/coverage_table.tex       (LaTeX table)
  dgp_replicate_summary/coverage_summary.csv
  dgp_replicate_summary/coverage_table.tex
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

REPO  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(REPO, "results_simulation")
DATA  = os.path.join(REPO, "data_simulation")

# Nominal HPD level for variational normal posteriors on x and iota.
# The same HPD_{0.95} = mu +- 1.96 sigma is used by CCP in 07c_iota_ccp.py.
ZCRIT = norm.ppf(0.975)
# Additional nominal levels for the multi-level coverage sweep
LEVELS_SWEEP = [0.90, 0.95, 0.99, 0.999, 0.9999]
COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]

SIM_TYPES = {
    "fixed_truth": dict(
        prefix      = "sim_",
        n           = 20,
        master_gt   = os.path.join(DATA, "simdata_centered_design",
                                    "ground_truth"),
        per_rep_gt  = False,    # shared GT
        out_dir     = os.path.join(RES, "centered_replicate_summary"),
        pretty_name = "Fixed-truth MC (20 Poisson seeds, shared truth)",
    ),
    "dgp_family": dict(
        prefix      = "sim_dgp_",
        n           = 20,
        master_gt   = None,
        per_rep_gt  = True,     # per-replicate GT
        out_dir     = os.path.join(RES, "dgp_replicate_summary"),
        pretty_name = "DGP-family MC (20 fresh truths)",
    ),
}


def hungarian_align_ideal(ideal_sim, ideal_hat):
    """Return (perm, signs) so that
       ideal_hat_aligned[:,k] = signs[k] * ideal_hat[:, perm[k]]
       is the best |corr|-aligned match to ideal_sim[:, k]."""
    K = ideal_sim.shape[1]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            r = np.corrcoef(ideal_sim[:, i], ideal_hat[:, j])[0, 1]
            C[i, j] = abs(r) if not np.isnan(r) else 0.0
    row, col = linear_sum_assignment(-C)
    perm = {int(r): int(c) for r, c in zip(row, col)}
    signs = np.zeros(K)
    for k in range(K):
        r = np.corrcoef(ideal_sim[:, k], ideal_hat[:, perm[k]])[0, 1]
        signs[k] = np.sign(r) if not np.isnan(r) else 1.0
    return perm, signs


def coverage_at_level(rep_dir, gt_dir, L):
    """Empirical coverage at nominal HPD level L (0 < L < 1).
    Returns (cov_iota_active, cov_iota_null, cov_x) for one replicate."""
    z = norm.ppf(0.5 + L / 2)
    df = pd.read_csv(os.path.join(rep_dir, "iota_ccp_table.csv"))
    df_act  = df[df["gt_active"] == 1]
    df_null = df[df["gt_active"] == 0]
    lo_a = df_act["iota_hat"]  - z * df_act["sigma_j"]
    hi_a = df_act["iota_hat"]  + z * df_act["sigma_j"]
    cov_act = float(((lo_a <= df_act["iota_sim"]) &
                     (df_act["iota_sim"] <= hi_a)).mean()) if len(df_act) else np.nan
    lo_n = df_null["iota_hat"] - z * df_null["sigma_j"]
    hi_n = df_null["iota_hat"] + z * df_null["sigma_j"]
    cov_null = float(((lo_n <= df_null["iota_sim"]) &
                      (df_null["iota_sim"] <= hi_n)).mean()) if len(df_null) else np.nan
    # x_{a,k}: Hungarian + sign-flip
    ideal_sim = np.load(os.path.join(gt_dir, "ideal_sim.npy"))
    ideal_hat = np.load(os.path.join(rep_dir, "params",
                                       "ideal_point_location_final.npy"))
    ideal_scl = np.load(os.path.join(rep_dir, "params",
                                       "ideal_point_scale_final.npy"))
    perm, signs = hungarian_align_ideal(ideal_sim, ideal_hat)
    K = ideal_sim.shape[1]
    ihat_a = np.stack([signs[k] * ideal_hat[:, perm[k]] for k in range(K)], axis=1)
    iscl_a = np.stack([ideal_scl[:, perm[k]] for k in range(K)], axis=1)
    lox = ihat_a - z * iscl_a; hix = ihat_a + z * iscl_a
    cov_x = float(((lox <= ideal_sim) & (ideal_sim <= hix)).mean())
    return cov_act, cov_null, cov_x


def write_levels_latex(rows_fixed, rows_dgp, out_tex):
    """Joint table: nominal x {Fixed-truth, DGP-family} x {ι_active, ι_null, x_{a,k}}."""
    def fmt(v):
        return f"{100*v:.1f}" if not np.isnan(v) else "--"
    lines = []
    lines.append("% Auto-generated by 08e_coverage_summary.py — multi-level coverage")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Empirical HPD coverage as a function of the nominal "
        r"level, for both Monte-Carlo studies of \S\ref{sec:dgp_fixed} "
        r"and \S\ref{sec:dgp_family}. Entries are the mean empirical "
        r"coverage (in $\%$) over the $20$ replicates of each MC, "
        r"stratified into truly-active $\iota$ cells (`$\iota$ act.'), "
        r"truly-null $\iota$ cells (`$\iota$ null') and the pooled "
        r"per-author per-topic ideal points (`$\ideal_{a,k}$'). The nominal "
        r"level is shown as the percentage on the left; a perfectly "
        r"calibrated posterior would have empirical coverage equal to "
        r"the nominal level (diagonal). Widening from HPD$_{0.95}$ to "
        r"HPD$_{0.9999}$ multiplies the interval width by "
        r"$z_{0.99995}/z_{0.975}\approx 1.99$.}")
    lines.append(r"\label{tab:coverage_levels}")
    lines.append(r"\begin{tabular}{l c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{Fixed-truth MC} & "
                 r"\multicolumn{3}{c}{DGP-family MC} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    lines.append(r"nominal & $\iota$ act. & $\iota$ null & $\ideal_{a,k}$ "
                 r"& $\iota$ act. & $\iota$ null & $\ideal_{a,k}$ \\")
    lines.append(r"\midrule")
    for L in LEVELS_SWEEP:
        fa = fmt(rows_fixed[L][0]); fn = fmt(rows_fixed[L][1]); fx = fmt(rows_fixed[L][2])
        da = fmt(rows_dgp[L][0]);   dn = fmt(rows_dgp[L][1]);   dx = fmt(rows_dgp[L][2])
        lines.append(
            f"{100*L:.2f}\\% & {fa} & {fn} & {fx} & {da} & {dn} & {dx} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def process_replicate(rep_dir, gt_dir):
    """Read iota_ccp_table.csv, ideal_sim, ideal_hat + scale; compute the
    coverage / width / bias diagnostics for this replicate."""
    df_iota = pd.read_csv(os.path.join(rep_dir, "iota_ccp_table.csv"))
    # iota coverage: truth lies in [hpd_lo, hpd_hi]?
    df_iota = df_iota.assign(
        covered = (df_iota["hpd_lo"] <= df_iota["iota_sim"])
                  & (df_iota["iota_sim"] <= df_iota["hpd_hi"]),
        width   = df_iota["hpd_hi"] - df_iota["hpd_lo"],
    )
    out = {}

    # Per-covariate coverage on truly active cells (gt_active == 1)
    for j, cov in enumerate(COV_LABELS):
        sub_all  = df_iota[df_iota["j"] == j]
        sub_act  = sub_all[sub_all["gt_active"] == 1]
        sub_null = sub_all[sub_all["gt_active"] == 0]
        if len(sub_act) > 0:
            out[f"iota_cov_active_{cov}"]   = float(sub_act["covered"].mean())
            out[f"iota_width_active_{cov}"] = float(sub_act["width"].mean())
        else:
            out[f"iota_cov_active_{cov}"]   = np.nan
            out[f"iota_width_active_{cov}"] = np.nan
        if len(sub_null) > 0:
            out[f"iota_cov_null_{cov}"]     = float(sub_null["covered"].mean())
            out[f"iota_width_null_{cov}"]   = float(sub_null["width"].mean())
        else:
            out[f"iota_cov_null_{cov}"]     = np.nan
            out[f"iota_width_null_{cov}"]   = np.nan

    # Pooled iota coverage
    df_act  = df_iota[df_iota["gt_active"] == 1]
    df_null = df_iota[df_iota["gt_active"] == 0]
    out["iota_cov_active_overall"] = float(df_act["covered"].mean()) if len(df_act) else np.nan
    out["iota_cov_null_overall"]   = float(df_null["covered"].mean()) if len(df_null) else np.nan
    out["iota_width_active_overall"] = float(df_act["width"].mean()) if len(df_act) else np.nan
    out["iota_width_null_overall"]   = float(df_null["width"].mean()) if len(df_null) else np.nan

    # Bias slope alpha* (no-intercept OLS) on truly active cells
    x = df_act["iota_sim"].to_numpy()
    y = df_act["iota_hat"].to_numpy()
    out["iota_alpha_star"] = float((x @ y) / (x @ x)) if len(x) > 0 else np.nan

    # ----- ideal points x_{a,k} -----
    ideal_sim = np.load(os.path.join(gt_dir, "ideal_sim.npy"))             # (A, K)
    ideal_hat = np.load(os.path.join(rep_dir, "params",
                                       "ideal_point_location_final.npy"))  # (A, K)
    ideal_scl = np.load(os.path.join(rep_dir, "params",
                                       "ideal_point_scale_final.npy"))     # (A, K)
    A, K = ideal_sim.shape
    perm, signs = hungarian_align_ideal(ideal_sim, ideal_hat)
    ideal_hat_a = np.stack([signs[k] * ideal_hat[:, perm[k]] for k in range(K)], axis=1)
    ideal_scl_a = np.stack([ideal_scl[:, perm[k]] for k in range(K)], axis=1)
    # HPD_{0.95} = ideal_hat_a +- ZCRIT * ideal_scl_a
    lo = ideal_hat_a - ZCRIT * ideal_scl_a
    hi = ideal_hat_a + ZCRIT * ideal_scl_a
    cov_x = (lo <= ideal_sim) & (ideal_sim <= hi)
    width_x = hi - lo
    out["x_cov_overall"]    = float(cov_x.mean())
    out["x_width_overall"]  = float(width_x.mean())
    # Bias for x: no-intercept slope per topic, then averaged
    alpha_x_per_k = []
    for k in range(K):
        xs = ideal_sim[:, k]; ys = ideal_hat_a[:, k]
        if (xs @ xs) > 0:
            alpha_x_per_k.append((xs @ ys) / (xs @ xs))
    out["x_alpha_star_meanK"] = float(np.mean(alpha_x_per_k)) if alpha_x_per_k else np.nan
    return out


def aggregate(sim_type, cfg):
    rows = []
    for r in range(1, cfg["n"] + 1):
        rep_dir = os.path.join(RES, f"{cfg['prefix']}{r:02d}")
        gt_dir  = (os.path.join(DATA, f"{cfg['prefix']}{r:02d}", "ground_truth")
                   if cfg["per_rep_gt"] else cfg["master_gt"])
        if not os.path.exists(os.path.join(rep_dir, "iota_ccp_table.csv")):
            print(f"  WARN: missing {rep_dir}/iota_ccp_table.csv")
            continue
        d = process_replicate(rep_dir, gt_dir)
        d["replicate"] = r
        rows.append(d)
    df = pd.DataFrame(rows)
    cols = ["replicate"] + [c for c in df.columns if c != "replicate"]
    return df[cols]


def write_latex(df, sim_type, cfg, out_tex):
    cov_pct = lambda c: 100 * df[c].mean()
    cov_sd  = lambda c: 100 * df[c].std(ddof=1)
    wd_mean = lambda c: df[c].mean()
    wd_sd   = lambda c: df[c].std(ddof=1)

    lines = []
    lines.append("% Auto-generated by 08e_coverage_summary.py — do not edit by hand.")
    lines.append(f"% {cfg['pretty_name']}, N = {len(df)} replicates")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Empirical HPD$_{0.95}$ coverage and mean interval width "
        r"for the variational posterior on the regression coefficients "
        r"$\iota_{k,j}$ (per covariate $j$, stratified by whether the "
        r"simulator's truth is non-zero (`active') or exactly zero "
        r"(`null')) and on the per-author per-topic ideal points "
        r"$x_{a,k}$ (pooled across all $A\cdot K$ cells, after Hungarian "
        r"topic-alignment and per-topic sign-flip against the simulator's "
        r"$x_{a,k}^{\mathrm{sim}}$). "
        r"Bias is summarised by the no-intercept OLS slope "
        r"$\alpha^\star = \sum t\,\hat t / \sum t^2$ (with $t$ the simulator truth and $\hat t$ the variational posterior mean) (a value of $1$ "
        r"denotes unbiased magnitude recovery; $\alpha^\star>1$ "
        r"indicates an upward magnitude bias). Mean $\pm$ standard "
        r"deviation over " + f"{len(df)}" + r" Monte-Carlo replicates of the " +
        cfg["pretty_name"].replace("&", r"\&") + r".}")
    lines.append(r"\label{tab:coverage_" + sim_type + r"}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")
    lines.append(r" & active cells (\%) & active width & "
                 r"null cells (\%) & null width \\")
    lines.append(r"\midrule")
    # Per-covariate rows
    for cov in COV_LABELS:
        ca = f"iota_cov_active_{cov}";   wa = f"iota_width_active_{cov}"
        cn = f"iota_cov_null_{cov}";     wn = f"iota_width_null_{cov}"
        active_n = df[ca].notna().sum()
        null_n   = df[cn].notna().sum()
        s_ca = f"{cov_pct(ca):.1f}\\,$\\pm$\\,{cov_sd(ca):.1f}" if active_n else "--"
        s_wa = f"{wd_mean(wa):.3f}" if active_n else "--"
        s_cn = f"{cov_pct(cn):.1f}\\,$\\pm$\\,{cov_sd(cn):.1f}" if null_n else "--"
        s_wn = f"{wd_mean(wn):.3f}" if null_n else "--"
        cov_esc = cov.replace("_", r"\_")
        lines.append(
            f"{cov_esc} & {s_ca} & {s_wa} & {s_cn} & {s_wn} \\\\"
        )
    lines.append(r"\midrule")
    # Pooled iota
    lines.append(
        rf"\textbf{{$\iota_{{k,j}}$, pooled}} & "
        rf"\textbf{{{cov_pct('iota_cov_active_overall'):.1f}\,$\pm$\,{cov_sd('iota_cov_active_overall'):.1f}}} & "
        rf"{wd_mean('iota_width_active_overall'):.3f} & "
        rf"\textbf{{{cov_pct('iota_cov_null_overall'):.1f}\,$\pm$\,{cov_sd('iota_cov_null_overall'):.1f}}} & "
        rf"{wd_mean('iota_width_null_overall'):.3f} \\"
    )
    lines.append(r"\midrule")
    # Pooled ideal point
    lines.append(
        rf"\textbf{{$x_{{a,k}}$, pooled}} (overall) & "
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{cov_pct('x_cov_overall'):.1f}\,$\pm$\,{cov_sd('x_cov_overall'):.1f}}}\;\;(width {wd_mean('x_width_overall'):.3f})}} & "
        rf"\multicolumn{{2}}{{l}}{{}} \\"
    )
    lines.append(r"\midrule")
    # Bias rows
    lines.append(
        rf"$\alpha^\star_\iota$ (active $\iota$ cells) & "
        rf"\multicolumn{{4}}{{l}}{{{df['iota_alpha_star'].mean():.3f}\,$\pm$\,{df['iota_alpha_star'].std(ddof=1):.3f}}} \\"
    )
    lines.append(
        rf"$\alpha^\star_\ideal$ (per-topic mean) & "
        rf"\multicolumn{{4}}{{l}}{{{df['x_alpha_star_meanK'].mean():.3f}\,$\pm$\,{df['x_alpha_star_meanK'].std(ddof=1):.3f}}} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    # Step 1: per-MC summary tables (active/null per covariate)
    levels_by_mc = {}
    for name, cfg in SIM_TYPES.items():
        os.makedirs(cfg["out_dir"], exist_ok=True)
        print("=" * 70)
        print(f"  {cfg['pretty_name']}")
        print("=" * 70)
        df = aggregate(name, cfg)
        if len(df) == 0:
            print("  no replicates found, skipping"); continue
        out_csv = os.path.join(cfg["out_dir"], "coverage_summary.csv")
        df.to_csv(out_csv, index=False)
        print(f"  -> {out_csv}")
        out_tex = os.path.join(cfg["out_dir"], "coverage_table.tex")
        write_latex(df, name, cfg, out_tex)
        print(f"  -> {out_tex}")

        # Multi-level sweep for this MC
        per_rep = {L: [[], [], []] for L in LEVELS_SWEEP}
        for r in range(1, cfg["n"] + 1):
            rep_dir = os.path.join(RES, f"{cfg['prefix']}{r:02d}")
            gt_dir  = (os.path.join(DATA, f"{cfg['prefix']}{r:02d}", "ground_truth")
                       if cfg["per_rep_gt"] else cfg["master_gt"])
            if not os.path.exists(os.path.join(rep_dir, "iota_ccp_table.csv")):
                continue
            for L in LEVELS_SWEEP:
                a, b, c = coverage_at_level(rep_dir, gt_dir, L)
                per_rep[L][0].append(a); per_rep[L][1].append(b); per_rep[L][2].append(c)
        levels_by_mc[name] = {L: (np.mean(per_rep[L][0]),
                                  np.mean(per_rep[L][1]),
                                  np.mean(per_rep[L][2])) for L in LEVELS_SWEEP}
        # show headline
        print(f"\n  n_replicates = {len(df)}")
        keys = [
            ("iota_cov_active_overall", "iota cov, active (%)"),
            ("iota_cov_null_overall",   "iota cov, null   (%)"),
            ("x_cov_overall",           "x_{a,k} cov      (%)"),
            ("iota_alpha_star",         "alpha*_iota (active)"),
            ("x_alpha_star_meanK",      "alpha*_x (mean K)"),
        ]
        for k, lbl in keys:
            print(f"    {lbl:<26}  {100*df[k].mean():6.2f}  +- "
                  f"{100*df[k].std(ddof=1):5.2f}" if "cov" in k
                  else f"    {lbl:<26}  {df[k].mean():6.3f}  +- "
                  f"{df[k].std(ddof=1):5.3f}")
        print()

    # Cross-MC multi-level coverage table
    if "fixed_truth" in levels_by_mc and "dgp_family" in levels_by_mc:
        out_tex = os.path.join(RES, "coverage_levels_table.tex")
        write_levels_latex(levels_by_mc["fixed_truth"],
                            levels_by_mc["dgp_family"], out_tex)
        print(f"-> {out_tex}")


if __name__ == "__main__":
    main()
