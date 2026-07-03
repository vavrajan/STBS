#!/usr/bin/env python3
"""
24_no_party_auc.py
==================
Party-classification AUC (Republican vs. Democrat) of the theta-weighted
aggregate STBS ideal point, for the no-party ablation runs.

Complements the DW-NOMINATE Pearson correlation in tab:no_party_ablation:
the AUC measures how well the aggregate ideal point *separates* the two
parties, on a bounded [0,1] scale, under the three ideal-point
initialisations (party / zero / random) with the party dummies removed
from X. The DW-NOMINATE benchmark AUC is reported as a ceiling.

The AUC routine is the SAME rank-sum (Mann-Whitney) identity used for the
per-topic face-validity table (17_face_validity.py, auc_RvsD), so the
numbers are on the identical convention as the rest of the paper.

Outputs:
  results_simulation/no_party/no_party_auc_summary.csv
"""
import os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(REPO, "..", "STBS_CAVI", "data", "hein-daily",
                    "clean", "author_detailed_info114.csv")
NP_DIR = os.path.join(REPO, "stbs_cavi_results_no_party")
OUT_DIR = os.path.join(REPO, "results_simulation", "no_party")
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = {
    "party_init":  "seed_314159_K25_init-party_pm1",
    "zero_init":   "seed_314159_K25_init-zero_OLD",
    "random_init": "seed_314159_K25_init-random_pm1",
}


def auc_RvsD(x, party):
    """AUC of treating x as a score for classifying R vs D
    (R = positive class). Rank-sum identity --- identical to
    17_face_validity.py:auc_RvsD."""
    is_R = (party == "R").to_numpy()
    nR = is_R.sum(); nD = (~is_R).sum()
    if nR == 0 or nD == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(x)).rank().to_numpy()
    return float((ranks[is_R].sum() - nR * (nR + 1) / 2) / (nR * nD))


def load_party():
    meta = pd.read_csv(META)
    scol = "surname_x" if "surname_x" in meta.columns else "surname"
    meta = meta[[scol, "party"]].rename(columns={scol: "surname"})
    meta["surname"] = meta["surname"].astype(str).str.upper().str.strip()
    return meta


def main():
    meta = load_party()
    rows = []
    from scipy.stats import pearsonr

    # ---- Headline / baseline full model (party in X), from the main fit ----
    cmp_f = os.path.join(REPO, "stbs_cavi_results", "comparison_dw_nominate.csv")
    if os.path.exists(cmp_f):
        cmp = pd.read_csv(cmp_f)
        cmp["surname"] = cmp["author"].astype(str).str.split().str[-1].str.upper().str.strip()
        mh = cmp.merge(meta, on="surname", how="inner")
        mh = mh[mh["party"].isin(["D", "R"])]
        col = "ip_mean"   # multi-seed mean, matching the +0.856 headline number
        rows.append(dict(run="headline_full_model", N=len(mh),
                         nD=int((mh["party"] == "D").sum()),
                         nR=int((mh["party"] == "R").sum()),
                         auc_RvsD=round(auc_RvsD(mh[col], mh["party"]), 3),
                         sep_auc=round(max(auc_RvsD(mh[col], mh["party"]),
                                           1 - auc_RvsD(mh[col], mh["party"])), 3),
                         pearson_dw=round(pearsonr(mh[col], mh["dw_nominate"])[0], 3),
                         auc_dw_benchmark=round(auc_RvsD(mh["dw_nominate"], mh["party"]), 3)))

    for name, sub in RUNS.items():
        f = os.path.join(NP_DIR, sub, "dw_nominate_no_party_correlation.csv")
        if not os.path.exists(f):
            print(f"  WARNING: missing {f}"); continue
        df = pd.read_csv(f)
        df["surname"] = df["surname"].astype(str).str.upper().str.strip()
        m = df.merge(meta, on="surname", how="inner")
        m = m[m["party"].isin(["D", "R"])]            # binary AUC: drop Independents
        auc = auc_RvsD(m["ip_no_party"], m["party"])
        sep = max(auc, 1.0 - auc)
        pr = pearsonr(m["ip_no_party"], m["dw_nominate"])[0]
        # DW-NOMINATE benchmark on the same matched set
        auc_dw = auc_RvsD(m["dw_nominate"], m["party"])
        rows.append(dict(run=name, N=len(m),
                         nD=int((m["party"] == "D").sum()),
                         nR=int((m["party"] == "R").sum()),
                         auc_RvsD=round(auc, 3), sep_auc=round(sep, 3),
                         pearson_dw=round(pr, 3),
                         auc_dw_benchmark=round(auc_dw, 3)))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "no_party_auc_summary.csv"), index=False)
    print(out.to_string(index=False))
    print(f"\n-> {OUT_DIR}/no_party_auc_summary.csv")


if __name__ == "__main__":
    main()
