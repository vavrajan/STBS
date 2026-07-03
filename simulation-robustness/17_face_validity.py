#!/usr/bin/env python3
"""
17_face_validity.py
===================
Build the construct-/face-validity material for the simulation_reply
based on the ORIGINAL PolAn variational fit
(originalPolAn_results/fits/TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25/).

Outputs (all into results_simulation/face_validity/):

  * per_topic_top3_table.tex
        25-row LaTeX table; one row per topic with the topic label,
        the three most-conservative and three most-liberal senators
        according to the PolAn posterior-mean ideal point hat_x_{a,k},
        plus a per-topic R/D separation diagnostic (Cohen's d).

  * dw_nominate_scatter.{pdf,png}
        Scatter of the PolAn aggregated ideal point (the `avg` column
        of params/ideal_data.csv, which is the theta-weighted
        per-senator aggregate from supplement S.1.5) against
        DW-NOMINATE 1st dimension. Headline correlation in the title.

  * wrong_side_outliers.tex
        Senators whose per-topic ideal point sits on the OPPOSITE side
        of zero from their party (z-score > 2 against the party
        mean) -- a face-validity stress test.

  * face_validity_summary.csv
        Per-topic summary (mean R, mean D, gap, Cohen's d, AUC,
        topic label, top-/bottom-3 names) -- machine-readable.

Run:
    python3 17_face_validity.py
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO  = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.normpath(os.path.join(REPO, ".."))
POLAN = os.path.join(PROJ, "originalPolAn_results", "fits",
                      "TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25",
                      "params")
CLEAN = os.path.join(PROJ, "STBS_CAVI", "data", "hein-daily", "clean")
CAVI  = os.path.join(REPO, "stbs_cavi_results")
OUT   = os.path.join(REPO, "results_simulation", "face_validity")
os.makedirs(OUT, exist_ok=True)

# Published-paper topic labels (from 05_run_R_plots.R)
TOPIC_LABELS = [
    "National Security",                              # 0
    "Supreme Court",                                  # 1
    "Coast Guard",                                    # 2
    "Human Trafficking",                              # 3
    "Commemoration and Anniversaries",                # 4
    "Gun Violence",                                   # 5
    "Middle Class and Small Businesses",              # 6
    "Health Care",                                    # 7
    "Public Health (Zika)",                           # 8
    "Veterans and Health Care",                       # 9
    "Drugs and Addiction",                            # 10
    "Climate Change",                                 # 11
    "Natural Resources",                              # 12
    "Planned Parenthood and Abortion",                # 13
    "Institutes and Research",                        # 14
    "Middle East and Nuclear Weapons",                # 15
    "Immigration and DHS",                            # 16
    "Social Security and Taxes",                      # 17
    "Rhetorics and Discussion",                       # 18
    "Clean Water Act",                                # 19
    "Law Enforcement",                                # 20
    "Wars and Human Rights",                          # 21
    "Education for Children",                         # 22
    "Cyber Security",                                 # 23
    "Export, Import and Business",                    # 24
]


def load_polan():
    """Read ideal_data.csv from the PolAn fit. Returns a DataFrame
    with columns 0..24 (per-topic IPs), avg, tbip, surname, party."""
    return pd.read_csv(os.path.join(POLAN, "ideal_data.csv"), index_col=0)


def load_author_map():
    """Read author_map114.txt -> list of canonical 'First Last (Party)'."""
    names = []
    with open(os.path.join(CLEAN, "author_map114.txt")) as f:
        for line in f:
            s = line.strip()
            if s:
                names.append(s)
    return names


def cohen_d(x_R, x_D):
    """Cohen's d effect size between two samples."""
    nR, nD = len(x_R), len(x_D)
    sR2, sD2 = np.var(x_R, ddof=1), np.var(x_D, ddof=1)
    sp = np.sqrt(((nR - 1) * sR2 + (nD - 1) * sD2) / (nR + nD - 2))
    return (np.mean(x_R) - np.mean(x_D)) / sp if sp > 0 else 0.0


def auc_RvsD(x, party):
    """AUC of treating x as a score for classifying R vs D
    (R = positive class). Uses the rank-sum identity."""
    is_R = (party == "R").to_numpy()
    nR = is_R.sum(); nD = (~is_R).sum()
    if nR == 0 or nD == 0:
        return float("nan")
    ranks = pd.Series(x).rank().to_numpy()
    return float((ranks[is_R].sum() - nR * (nR + 1) / 2) / (nR * nD))


def latex_escape(s):
    return s.replace("&", r"\&").replace("_", r"\_")


# =====================================================================
def main():
    df = load_polan()             # 99 rows
    names_map = load_author_map()
    print(f"loaded PolAn fit: {df.shape} (99 senators, 25 topics + avg + tbip)")
    print(f"loaded author_map: {len(names_map)} names")

    # Build first-name index by surname; surname column is uppercase
    # Build full 'First Last (Party)' lookup keyed by surname
    full_name_by_surname = {}
    for nm in names_map:
        # nm = 'Alan Franken (D)'
        m = re.match(r"(.+)\s+\(([A-Z])\)$", nm)
        if not m:
            continue
        first_last = m.group(1).strip()
        surname = first_last.split()[-1].upper()
        full_name_by_surname[surname] = nm
    df["full_name"] = df["surname"].map(
        lambda s: full_name_by_surname.get(s, s.title()))

    # Restrict to R/D for the separation diagnostics (drop 2 Independents)
    df_RD = df[df["party"].isin(["R", "D"])].copy()

    # ---------------- Per-topic summary ----------------
    print("\n=== Per-topic top-3 + separation ===")
    rows = []
    for k in range(25):
        col = str(k)
        x_R = df_RD[df_RD["party"] == "R"][col].to_numpy()
        x_D = df_RD[df_RD["party"] == "D"][col].to_numpy()
        d   = cohen_d(x_R, x_D)
        auc = auc_RvsD(df_RD[col], df_RD["party"])
        # Top-3 conservative (largest IP), Top-3 liberal (smallest IP)
        sub = df.sort_values(col)
        bot3 = sub.iloc[:3][["full_name", "party", col]].values.tolist()
        top3 = sub.iloc[-3:][::-1][["full_name", "party", col]].values.tolist()
        rows.append(dict(
            k=k, label=TOPIC_LABELS[k],
            R_mean=float(x_R.mean()), D_mean=float(x_D.mean()),
            gap=float(x_R.mean() - x_D.mean()),
            cohen_d=float(d), auc_RvsD=float(auc),
            # n already contains '(R)' or '(D)', so do NOT re-append (p)
            top3_conservative=";".join(
                f"{n} {v:+.2f}" for n, p, v in top3),
            top3_liberal=";".join(
                f"{n} {v:+.2f}" for n, p, v in bot3),
        ))
        print(f"  k={k:2d} {TOPIC_LABELS[k]:<45} "
              f"R={x_R.mean():+.2f} D={x_D.mean():+.2f} "
              f"d={d:+.2f} AUC={auc:.2f}")
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT, "face_validity_summary.csv"), index=False)

    # ---------------- LaTeX master table ----------------
    print("\n=== writing per_topic_top3_table.tex ===")
    tex = []
    tex.append("% Auto-generated by 17_face_validity.py")
    tex.append(r"\begin{sidewaystable}")
    tex.append(r"\centering\scriptsize")
    tex.append(r"\caption{Face validity of the per-topic ideal points "
               r"from the main fit reported in the paper "
               r"($K{=}25$). For each topic $k$ we list the three "
               r"senators with the most-positive (`conservative')  "
               r"and most-negative (`liberal') topic-specific ideal "
               r"points $\hat\ideal_{a,k}$, with each name labelled by "
               r"party and ideal-point value. The AUC column reports "
               r"the area under the ROC curve of using $\hat\ideal_{a,k}$ "
               r"to classify the $97$ R/D senators (the two Independents "
               r"are excluded). $\mathrm{AUC}=1$ means perfect linear "
               r"separability of the two parties along the topic "
               r"dimension; $\mathrm{AUC}=0.5$ means no separation. "
               r"Rows are sorted by AUC in descending order, so the "
               r"cleanest-separating topics appear at the top of the "
               r"table.}")
    tex.append(r"\label{tab:face_validity_top3}")
    tex.append(r"\setlength{\tabcolsep}{3pt}")
    tex.append(r"\rowcolors{2}{gray!12}{white}")
    tex.append(r"\begin{tabular}{r@{\hskip 6pt} p{4.5cm}@{\hskip 2pt} c@{\hskip 8pt} p{8.0cm} p{8.0cm}}")
    tex.append(r"\toprule")
    tex.append(r"$k$ & Topic & AUC & Top-3 conservative ($\hat\ideal_{a,k}$) & "
               r"Top-3 liberal ($\hat\ideal_{a,k}$) \\")
    tex.append(r"\midrule")
    # Sort rows by AUC desc (tie-break: k asc) for the typeset table
    rows_sorted = sorted(rows, key=lambda r: (-r["auc_RvsD"], r["k"]))
    for r in rows_sorted:
        cons = "; ".join(s.strip() for s in r["top3_conservative"].split(";"))
        libe = "; ".join(s.strip() for s in r["top3_liberal"].split(";"))
        tex.append(
            f"{r['k']} & {latex_escape(r['label'])} & "
            f"{r['auc_RvsD']:.2f} & "
            f"{latex_escape(cons)} & {latex_escape(libe)} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{sidewaystable}")
    with open(os.path.join(OUT, "per_topic_top3_table.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")

    # ---------------- DW-NOMINATE scatter ----------------
    print("\n=== DW-NOMINATE convergent-validity scatter ===")
    dw = pd.read_csv(os.path.join(CAVI, "comparison_dw_nominate.csv"))
    dw["surname"] = dw["author"].apply(lambda s: s.split()[-1].upper())
    merged = df.merge(dw[["surname", "dw_nominate"]], on="surname", how="inner")
    merged = merged.dropna(subset=["dw_nominate", "avg"])
    print(f"  merged: {len(merged)} senators")
    r_pearson = float(np.corrcoef(merged["avg"], merged["dw_nominate"])[0, 1])
    print(f"  Pearson r(PolAn-avg, DW-NOMINATE) = {r_pearson:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    colors = {"R": "#c0392b", "D": "#2874a6", "I": "#7f8c8d"}
    for p in ["R", "D", "I"]:
        sub = merged[merged["party"] == p]
        ax.scatter(sub["dw_nominate"], sub["avg"], s=40,
                    c=colors[p], label=f"{p} (n={len(sub)})", alpha=0.7,
                    edgecolor="black", lw=0.4)
    # Diagonal reference (linear regression)
    z = np.polyfit(merged["dw_nominate"], merged["avg"], 1)
    xs = np.array([merged["dw_nominate"].min(), merged["dw_nominate"].max()])
    ax.plot(xs, z[0] * xs + z[1], "k--", lw=0.8, alpha=0.6,
             label=f"OLS fit  ({z[0]:+.2f}x + {z[1]:+.2f})")
    # Identify a few notable outliers / endpoints to annotate
    merged["resid"] = merged["avg"] - (z[0] * merged["dw_nominate"] + z[1])
    notable_idx = (merged["avg"].abs().rank(ascending=False) <= 5) | \
                  (merged["dw_nominate"].abs().rank(ascending=False) <= 5) | \
                  (merged["resid"].abs().rank(ascending=False) <= 5)
    for _, row in merged[notable_idx].iterrows():
        ax.annotate(row["surname"].title(),
                     (row["dw_nominate"], row["avg"]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.4, ls=":")
    ax.axvline(0, color="black", lw=0.4, ls=":")
    ax.set_xlabel("DW-NOMINATE 1st dimension", fontsize=11)
    ax.set_ylabel(r"Aggregate ideal point $\bar{\hat{\mathrm{i}}}_a$ ($\theta$-weighted, supplement S.1.5)",
                   fontsize=11)
    ax.set_title(f"Convergent validity: STBS aggregate IP vs. DW-NOMINATE\n"
                  f"Pearson $r = {r_pearson:.3f}$  (n={len(merged)} senators)",
                  fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dw_nominate_scatter.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "dw_nominate_scatter.png"), dpi=160, bbox_inches="tight")
    print(f"  -> {OUT}/dw_nominate_scatter.{{pdf,png}}")

    # ---------------- Wrong-side outliers ----------------
    print("\n=== wrong-side outliers ===")
    # Build a quick AUC lookup by topic k from the per-topic summary
    auc_by_k = {r["k"]: float(r["auc_RvsD"]) for r in rows}
    # Criterion (simple distances):
    #   flag:    the senator's ideal point lies BEYOND the opposite
    #            party's mean (R: below the D mean; D: above the R mean)
    #   report:  absolute distance to the own-party mean (computed
    #            EXCLUDING the senator) and to the opposite-party mean
    #   select:  top-25 by distance-to-own-party
    outliers = []
    for k in range(25):
        col = str(k)
        muR = df_RD[df_RD["party"] == "R"][col].mean()
        muD = df_RD[df_RD["party"] == "D"][col].mean()
        for _, row in df_RD.iterrows():
            v = float(row[col])
            party = row["party"]
            mu_opp = muD if party == "R" else muR
            beyond = (party == "R" and v < mu_opp) or \
                     (party == "D" and v > mu_opp)
            if not beyond:
                continue
            own = df_RD[(df_RD["party"] == party)
                        & (df_RD["full_name"] != row["full_name"])][col]
            outliers.append(dict(
                k=k, label=TOPIC_LABELS[k],
                senator=row["full_name"], party=party,
                ideal=v,
                dist_own=abs(v - own.mean()),
                dist_opp=abs(v - mu_opp),
                auc=auc_by_k[k],
            ))
    df_out = pd.DataFrame(outliers)
    # Select the 25 cases farthest from their OWN party; display sorted
    # by per-topic AUC desc (matching Table 3), tiebreak dist_own desc.
    df_out_top = df_out.nlargest(25, "dist_own").sort_values(
        by=["auc", "dist_own"], ascending=[False, False])
    # Full CSV: ordered by distance-to-own-party for analysis convenience
    df_out_full = df_out.sort_values("dist_own", ascending=False)
    df_out_full.to_csv(os.path.join(OUT, "wrong_side_outliers.csv"), index=False)
    print(f"  found {len(df_out)} wrong-side cases over {25} topics")
    print(df_out_top.head(15).to_string(index=False))

    # LaTeX for the top wrong-side cases
    tex = []
    tex.append("% Auto-generated by 17_face_validity.py")
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering\small")
    tex.append(r"\caption{Wrong-side outliers: senators whose per-topic "
               r"ideal point $\hat\ideal_{a,k}$ lies \emph{beyond} the "
               r"opposite party's mean on that topic (Republicans below "
               r"the Democratic mean, Democrats above the Republican "
               r"mean). `dist.\ own' is the absolute distance to the "
               r"senator's own-party mean (computed excluding the "
               r"senator), `dist.\ opp.' the absolute distance to the "
               r"opposite party's mean. The $25$ cases farthest from "
               r"their own party are listed; the full table is in the "
               r"replication archive. Rows are sorted by per-topic AUC "
               r"in descending order (matching "
               r"Table~\ref{tab:face_validity_top3}), so flags on "
               r"cleanly partisan topics --- where deviation is "
               r"substantively meaningful --- appear first.}")
    tex.append(r"\label{tab:wrong_side_outliers}")
    tex.append(r"\rowcolors{2}{gray!12}{white}")
    tex.append(r"\begin{tabular}{r l c l c r r r}")
    tex.append(r"\toprule")
    tex.append(r"$k$ & Topic & AUC & Senator & Party & "
               r"$\hat\ideal_{a,k}$ & dist.\ own & dist.\ opp. \\")
    tex.append(r"\midrule")
    for _, row in df_out_top.iterrows():
        tex.append(
            f"{int(row['k'])} & {latex_escape(row['label'])} & "
            f"{row['auc']:.2f} & "
            f"{latex_escape(str(row['senator']))} & {row['party']} & "
            f"{row['ideal']:+.2f} & {row['dist_own']:.2f} & "
            f"{row['dist_opp']:.2f} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(os.path.join(OUT, "wrong_side_outliers.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"  -> {OUT}/wrong_side_outliers.{{tex,csv}}")

    print("\nDone.")


if __name__ == "__main__":
    main()
