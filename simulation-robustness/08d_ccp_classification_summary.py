#!/usr/bin/env python3
"""
08d_ccp_classification_summary.py
=================================
Aggregate the CCP classification (TP / FP / FN / TN) across the 20
replicates of each simulation type. For every replicate we read
results_simulation/sim_NN/iota_ccp_table.csv (or
results_simulation/sim_dgp_NN/iota_ccp_table.csv) and sum counts per
covariate. Two simulation types are handled:

  * fixed-truth Monte Carlo  (sim_01 ... sim_20)
  * DGP-family Monte Carlo   (sim_dgp_01 ... sim_dgp_20)

For each simulation type we produce three artefacts in the
corresponding summary directory:

  classification_counts_summary.csv
        Per-covariate totals (TP, FP, FN, TN summed across the 20
        replicates) plus precision, recall, F1.
  classification_counts_per_replicate.csv
        One row per replicate-and-covariate (long form), so plots
        downstream are easy.
  classification_counts_table.tex
        LaTeX booktabs table with per-covariate counts (mean per
        replicate, sum, and headline P / R / F1). Ready to
        \\input{} from any LaTeX document.

Run:
    python3 08d_ccp_classification_summary.py
"""
import os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(REPO, "results_simulation")

COV_LABELS = ["c0_zero", "c1_uniform_all", "c2_topic4",
              "c3_topic16", "c4_topic25"]

SIM_TYPES = {
    "fixed_truth": dict(
        prefix="sim_",
        n=20,
        out_dir=os.path.join(RES, "centered_replicate_summary"),
        pretty_name="Fixed-truth MC (20 Poisson seeds, shared truth)",
    ),
    "dgp_family": dict(
        prefix="sim_dgp_",
        n=20,
        out_dir=os.path.join(RES, "dgp_replicate_summary"),
        pretty_name="DGP-family MC (20 fresh truths)",
    ),
}


def aggregate(sim_type, cfg):
    """Read the per-replicate iota_ccp_table.csv files for one
    simulation type and produce per-covariate summaries."""
    rows_long = []
    rows_per_rep = []
    n_seen = 0
    for r in range(1, cfg["n"] + 1):
        rep_dir = os.path.join(RES, f"{cfg['prefix']}{r:02d}")
        csv_p = os.path.join(rep_dir, "iota_ccp_table.csv")
        if not os.path.exists(csv_p):
            print(f"  WARN: missing {csv_p}")
            continue
        df = pd.read_csv(csv_p)
        n_seen += 1
        # per (replicate, covariate)
        for j, cov in enumerate(COV_LABELS):
            sub = df[df["j"] == j]
            row = dict(
                replicate=r,
                j=j,
                covariate=cov,
                gt_active=int(sub["gt_active"].sum()),
                TP=int((sub["classification"] == "TP").sum()),
                FP=int((sub["classification"] == "FP").sum()),
                FN=int((sub["classification"] == "FN").sum()),
                TN=int((sub["classification"] == "TN").sum()),
            )
            rows_long.append(row)
        # per-replicate totals (across covariates)
        rows_per_rep.append(dict(
            replicate=r,
            TP=int((df["classification"] == "TP").sum()),
            FP=int((df["classification"] == "FP").sum()),
            FN=int((df["classification"] == "FN").sum()),
            TN=int((df["classification"] == "TN").sum()),
        ))

    long_df = pd.DataFrame(rows_long)
    rep_df = pd.DataFrame(rows_per_rep)

    # Per-covariate aggregate: sum + mean across replicates
    grp = long_df.groupby(["j", "covariate"], sort=False)
    summary = grp.agg(
        gt_active_per_rep=("gt_active", "mean"),
        TP_sum=("TP", "sum"),  FP_sum=("FP", "sum"),
        FN_sum=("FN", "sum"),  TN_sum=("TN", "sum"),
        TP_mean=("TP", "mean"), FP_mean=("FP", "mean"),
        FN_mean=("FN", "mean"), TN_mean=("TN", "mean"),
    ).reset_index()
    # precision / recall / F1 from summed counts
    summary["precision"] = summary["TP_sum"] / \
        (summary["TP_sum"] + summary["FP_sum"]).replace(0, np.nan)
    summary["recall"]    = summary["TP_sum"] / \
        (summary["TP_sum"] + summary["FN_sum"]).replace(0, np.nan)
    summary["F1"] = 2 * summary["precision"] * summary["recall"] / \
        (summary["precision"] + summary["recall"]).replace(0, np.nan)

    # TOTAL row
    tot_TP = int(summary["TP_sum"].sum())
    tot_FP = int(summary["FP_sum"].sum())
    tot_FN = int(summary["FN_sum"].sum())
    tot_TN = int(summary["TN_sum"].sum())
    prec = tot_TP / (tot_TP + tot_FP) if (tot_TP + tot_FP) else float("nan")
    rec  = tot_TP / (tot_TP + tot_FN) if (tot_TP + tot_FN) else float("nan")
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else float("nan")
    total_row = dict(
        j=-1, covariate="TOTAL",
        gt_active_per_rep=summary["gt_active_per_rep"].sum(),
        TP_sum=tot_TP, FP_sum=tot_FP, FN_sum=tot_FN, TN_sum=tot_TN,
        TP_mean=summary["TP_mean"].sum(), FP_mean=summary["FP_mean"].sum(),
        FN_mean=summary["FN_mean"].sum(), TN_mean=summary["TN_mean"].sum(),
        precision=prec, recall=rec, F1=f1,
    )
    summary = pd.concat([summary, pd.DataFrame([total_row])],
                        ignore_index=True)

    return n_seen, long_df, rep_df, summary, dict(
        precision=prec, recall=rec, F1=f1,
        TP=tot_TP, FP=tot_FP, FN=tot_FN, TN=tot_TN,
    )


def write_latex(summary, headline, n_reps, pretty_name, out_tex):
    """Write a self-contained booktabs LaTeX table."""
    cov_rows = summary[summary["covariate"] != "TOTAL"]
    tot = summary[summary["covariate"] == "TOTAL"].iloc[0]

    def esc(s):
        return str(s).replace("_", r"\_")

    def fmt(x, nd=3):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "--"
        if np.isnan(v):
            return "--"
        return f"{v:.{nd}f}"

    lines = []
    lines.append("% Auto-generated by 08d_ccp_classification_summary.py")
    lines.append("% Do not edit by hand.")
    lines.append(f"% {pretty_name}, N = {n_reps} replicates")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{CCP classification ($\alpha=0.05$) summed over " +
                 f"{n_reps} replicates of the " + pretty_name.replace("&", r"\&") +
                 r". Each row sums the classifications across all 25 topics of all "
                 f"{n_reps} replicates, so the \\texttt{{entries}} column equals "
                 f"$25 \\cdot {n_reps} = {25*n_reps}$ and the four classification "
                 r"columns (TP+FP+FN+TN) add up to this value. "
                 r"TP = true positive (CCP-significant and ground-truth active), "
                 r"FP = false positive (significant but inactive), "
                 r"FN = false negative (active but not flagged), "
                 r"TN = true negative.}")
    lines.append(r"\label{tab:ccp_class_" +
                 pretty_name.split()[0].lower().replace("-", "_") + r"}")
    lines.append(r"\begin{tabular}{l r r r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"covariate & GT act.\ (mean/rep) & "
                 r"TP & FP & FN & TN & entries & precision & recall & F1 \\")
    lines.append(r"\midrule")
    for _, r in cov_rows.iterrows():
        entries = int(r['TP_sum'] + r['FP_sum'] + r['FN_sum'] + r['TN_sum'])
        lines.append(
            f"{esc(r['covariate'])} & "
            f"{r['gt_active_per_rep']:.1f} & "
            f"{int(r['TP_sum'])} & {int(r['FP_sum'])} & "
            f"{int(r['FN_sum'])} & {int(r['TN_sum'])} & "
            f"{entries} & "
            f"{fmt(r['precision'])} & {fmt(r['recall'])} & {fmt(r['F1'])} \\\\"
        )
    lines.append(r"\midrule")
    tot_entries = int(tot['TP_sum'] + tot['FP_sum'] + tot['FN_sum'] + tot['TN_sum'])
    lines.append(
        f"\\textbf{{TOTAL}} & "
        f"{tot['gt_active_per_rep']:.1f} & "
        f"\\textbf{{{int(tot['TP_sum'])}}} & "
        f"\\textbf{{{int(tot['FP_sum'])}}} & "
        f"\\textbf{{{int(tot['FN_sum'])}}} & "
        f"\\textbf{{{int(tot['TN_sum'])}}} & "
        f"\\textbf{{{tot_entries}}} & "
        f"\\textbf{{{fmt(tot['precision'])}}} & "
        f"\\textbf{{{fmt(tot['recall'])}}} & "
        f"\\textbf{{{fmt(tot['F1'])}}} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    for name, cfg in SIM_TYPES.items():
        os.makedirs(cfg["out_dir"], exist_ok=True)
        print("=" * 70)
        print(f"  {cfg['pretty_name']}")
        print("=" * 70)
        n, long_df, rep_df, summary, headline = aggregate(name, cfg)
        if n == 0:
            print(f"  no replicates found, skipping")
            continue

        # Per-covariate aggregate
        out_csv = os.path.join(cfg["out_dir"],
                                "classification_counts_summary.csv")
        summary.to_csv(out_csv, index=False)
        print(f"  -> {out_csv}")

        # Per-replicate per-covariate long form
        out_long = os.path.join(cfg["out_dir"],
                                 "classification_counts_per_replicate.csv")
        long_df.to_csv(out_long, index=False)
        print(f"  -> {out_long}")

        # LaTeX snippet
        out_tex = os.path.join(cfg["out_dir"],
                                "classification_counts_table.tex")
        write_latex(summary, headline, n, cfg["pretty_name"], out_tex)
        print(f"  -> {out_tex}")

        # Print to stdout
        print(f"\n  n_replicates = {n}")
        print("  Headline (counts summed across all 20 replicates):")
        print(f"    TP = {headline['TP']}   FP = {headline['FP']}   "
              f"FN = {headline['FN']}   TN = {headline['TN']}")
        print(f"    precision = {headline['precision']:.3f}  "
              f"recall = {headline['recall']:.3f}  "
              f"F1 = {headline['F1']:.3f}\n")
        show_cols = ["covariate", "gt_active_per_rep",
                     "TP_sum", "FP_sum", "FN_sum", "TN_sum",
                     "precision", "recall", "F1"]
        print(summary[show_cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
