#!/usr/bin/env python3
"""
12_results_simulation_reply.py
==============================
Single source of truth for all numerical results that appear in
docs/simulation_reply.tex (the 11-page summary of the six robustness
and simulation exercises).

For every table in simulation_reply.tex this script:
  1. reads the underlying CSV(s) directly,
  2. exposes the contents as a nested dict `RESULTS`, and
  3. emits ready-to-paste LaTeX table-body snippets.

Run as a module (`from 12_results_simulation_reply import RESULTS`)
to get the dict, or run directly to print all summaries + LaTeX
snippets to stdout for cross-checking against the tex file.

Sources (all live under Revision_code_CAVI/):

  Robustness on the real-data fit
  -------------------------------
    seed sensitivity:
      stbs_cavi_results/topic_comparison/topic_comparison_summary.csv
      stbs_cavi_results/correlation_summary.csv

    number of topics K:
      stbs_cavi_results/K_comparison/K_comparison_summary.csv
      stbs_cavi_results/K_comparison/dw_nominate_by_K.csv

    hyperparameters: hard-coded from
      Revision_code/simulation_hyperparams/results_hyperparams.tex
      (no machine-readable CSV; numbers transcribed once below)

  DGP simulations
  ---------------
    fixed-truth MC (20 Poisson seeds):
      results_simulation/centered_replicate_summary/global_replicate_summary.csv

    DGP-family MC (20 fresh truths):
      results_simulation/dgp_replicate_summary/global_replicate_summary.csv

    misspec down (TBIP fit on STBS truth):
      results_simulation/dgp_constIP_summary/global_replicate_summary.csv
      results_simulation/dgp_constIP_summary/per_covariate_summary.csv

    misspec up (STBS fit on TBIP truth):
      results_simulation/tbip_replicate_summary/global_replicate_summary.csv
      results_simulation/tbip_replicate_summary/per_covariate_summary.csv
"""
from __future__ import annotations
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd


def _rhu(x: float, n: int) -> float:
    """Round half up to n decimal places (the conventional academic
    convention; Python's default banker's rounding would give 11.25 -> 11.2,
    which surprises readers)."""
    q = Decimal(10) ** (-n)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def _fmt(x: float, n: int) -> str:
    return f"{_rhu(x, n):.{n}f}"

REPO = Path(__file__).resolve().parent
RES = REPO / "results_simulation"
CAVI = REPO / "stbs_cavi_results"


def _read_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        print(f"  WARNING: {p} does not exist, returning None")
        return None
    return pd.read_csv(p)


def _ms(df: pd.DataFrame, metric: str, dec_mean: int = 3,
        dec_std: int = 3) -> str:
    """Format a single mean-std cell from a global_replicate_summary.csv
    table that has columns metric, mean, std, q025, q500, q975, min, max.
    Uses round-half-up rounding so e.g. 0.005 -> 0.01, not 0.00.
    """
    row = df[df["metric"] == metric]
    if row.empty:
        return "n/a"
    m = float(row["mean"].iloc[0])
    s = float(row["std"].iloc[0])
    return f"${_fmt(m, dec_mean)}\\pm {_fmt(s, dec_std)}$"


def _ms_int(df: pd.DataFrame, metric: str) -> str:
    row = df[df["metric"] == metric]
    if row.empty:
        return "n/a"
    m = float(row["mean"].iloc[0])
    s = float(row["std"].iloc[0])
    return f"${_fmt(m, 1)}\\pm {_fmt(s, 1)}$"


# ============================================================== #
# Load every result file once
# ============================================================== #
print("Loading source CSVs ...")
SEEDS_PAIRS = _read_csv(CAVI / "topic_comparison" / "topic_comparison_summary.csv")
SEEDS_DW    = _read_csv(CAVI / "correlation_summary.csv")
K_SUMMARY   = _read_csv(CAVI / "K_comparison" / "K_comparison_summary.csv")
K_DW        = _read_csv(CAVI / "K_comparison" / "dw_nominate_by_K.csv")
MC_FIXED    = _read_csv(RES / "centered_replicate_summary" / "global_replicate_summary.csv")
MC_DGP      = _read_csv(RES / "dgp_replicate_summary" / "global_replicate_summary.csv")
MS_DOWN_G   = _read_csv(RES / "dgp_constIP_summary" / "global_replicate_summary.csv")
MS_DOWN_PC  = _read_csv(RES / "dgp_constIP_summary" / "per_covariate_summary.csv")
MS_UP_G     = _read_csv(RES / "tbip_replicate_summary" / "global_replicate_summary.csv")
MS_UP_PC    = _read_csv(RES / "tbip_replicate_summary" / "per_covariate_summary.csv")


# ============================================================== #
# Hyperparameter sensitivity: read from the CURRENT CAVI run.
# Source: stbs_cavi_results/hyperparameter_sensitivity.csv produced
# by 02d_simulation_hyperparameters.py.
# Old PolAn numbers in Revision_code/simulation_hyperparams/
# results_hyperparams.tex are STALE and must not be used.
# ============================================================== #
HP_RAW = _read_csv(CAVI / "hyperparameter_sensitivity.csv")
if HP_RAW is not None:
    # Map CSV columns -> our table columns
    # CSV has: ip_topic_corr_mean, ip_agg_corr, iota_corr,
    # eta_topic_corr_mean, beta_topic_corr_mean
    HP_SENSITIVITY = pd.DataFrame({
        "group":    HP_RAW["group"].values,
        "value":    HP_RAW["value"].values,
        "IP_agg":   HP_RAW["ip_agg_corr"].values,
        "IP_topic": HP_RAW["ip_topic_corr_mean"].values,
        "iota":     HP_RAW["iota_corr"].values,
        "eta":      HP_RAW["eta_topic_corr_mean"].values,
        "beta":     HP_RAW["beta_topic_corr_mean"].values,
    })
else:
    HP_SENSITIVITY = None


# ============================================================== #
# Build the master RESULTS dict
# ============================================================== #
def build_results() -> dict:
    return {
        "seeds": {
            "between_seed_pairs": SEEDS_PAIRS.to_dict("records") if SEEDS_PAIRS is not None else None,
            "vs_dw_per_seed": SEEDS_DW.to_dict("records") if SEEDS_DW is not None else None,
        },
        "K": {
            "pairwise_summary": K_SUMMARY.to_dict("records") if K_SUMMARY is not None else None,
            "vs_dw_by_K": K_DW.to_dict("records") if K_DW is not None else None,
        },
        "hyperparameters": HP_SENSITIVITY.to_dict("records"),
        "mc_fixed": MC_FIXED.to_dict("records") if MC_FIXED is not None else None,
        "mc_dgp":   MC_DGP.to_dict("records") if MC_DGP is not None else None,
        "misspec_down": {
            "global":   MS_DOWN_G.to_dict("records") if MS_DOWN_G is not None else None,
            "per_cov":  MS_DOWN_PC.to_dict("records") if MS_DOWN_PC is not None else None,
        },
        "misspec_up": {
            "global":   MS_UP_G.to_dict("records") if MS_UP_G is not None else None,
            "per_cov":  MS_UP_PC.to_dict("records") if MS_UP_PC is not None else None,
        },
    }


RESULTS = build_results()


# ============================================================== #
# LaTeX snippet emitters — one per table in simulation_reply.tex
# ============================================================== #
def latex_seeds():
    if SEEDS_PAIRS is None or SEEDS_DW is None:
        return "% seeds CSVs missing"
    sp = SEEDS_PAIRS.copy().rename(columns=str.lower)
    rows = []
    for _, r in sp.iterrows():
        rows.append(
            f"seed {int(r['seed_a'])} vs.\\ seed {int(r['seed_b'])} & "
            f"${r['beta_cosine_mean']:.3f}$ & ${r['beta_cosine_min']:.3f}$ & "
            f"${r['eta_x_ideal_corr_mean']:.3f}$ & ${r['iota_corr_mean']:.3f}$ & "
            f"$\\mathbf{{{r['agg_ip_corr']:.3f}}}$ \\\\"
        )
    dw_block = []
    for _, r in SEEDS_DW.iterrows():
        if r["seed"] in ("mean", "std"):
            continue
        dw_block.append(f"seed {int(float(r['seed']))} & & & & & ${float(r['pearson_r']):.3f}$ \\\\")
    mean_row = SEEDS_DW[SEEDS_DW["seed"] == "mean"].iloc[0]
    std_row  = SEEDS_DW[SEEDS_DW["seed"] == "std"].iloc[0]
    return (
        "% --- seed sensitivity table body ---\n" +
        "\n".join(rows) +
        "\n\\midrule\n" +
        " & \\multicolumn{4}{r}{\\emph{vs.\\ DW-NOMINATE, Pearson $r$:}} & \\\\\n" +
        "\n".join(dw_block) +
        "\n\\midrule\n" +
        f"mean (3 seeds) & & & & & ${float(mean_row['pearson_r']):.3f}\\pm {float(std_row['pearson_r']):.3f}$ \\\\"
    )


def latex_K():
    if K_SUMMARY is None or K_DW is None:
        return "% K CSVs missing"
    rows1 = []
    for _, r in K_SUMMARY.iterrows():
        rows1.append(
            f"{int(r['K_A'])} & {int(r['K_B'])} & {int(r['n_matched'])} & "
            f"${r['beta_cosine_mean']:.3f}$ & {int(r['n_well_matched_05'])} & "
            f"{int(r['n_highly_matched_08'])} & ${r['iota_corr_mean']:.3f}$ & "
            f"${r['agg_ip_corr']:.3f}$ \\\\"
        )
    rows2 = []
    for _, r in K_DW.iterrows():
        rows2.append(f"{int(r['K'])} & ${r['dw_corr']:.3f}$ \\\\")
    return ("% --- K-pairwise body ---\n" + "\n".join(rows1) +
            "\n% --- DW-by-K body ---\n" + "\n".join(rows2))


def latex_hyperparameters():
    """Body of the hyperparameter sensitivity table. The CSV organises
    rows as: omega 0.1, omega 0.3 (baseline), omega 1.0, beta 0.1, beta 0.3,
    beta 1.0, ... where every group has its own 0.3 baseline row. We
    print the baseline ONCE (using the first 0.3 row) and then perturbed
    rows grouped by hyperparameter. Note: the CAVI CSV has no ELBO
    column, so the table has 7 columns (vs 8 in the old PolAn-era table).
    """
    if HP_SENSITIVITY is None:
        return "% hyperparameter_sensitivity.csv missing"
    out = ["% --- hyperparameter sensitivity body (CAVI) ---"]
    # Print baseline once
    base_rows = HP_SENSITIVITY[HP_SENSITIVITY["value"] == 0.3]
    if len(base_rows):
        b = base_rows.iloc[0]
        out.append("\\multicolumn{7}{l}{\\emph{Baseline ($a=0.3$ for all groups)}} \\\\")
        out.append(f"& ${b['value']:.1f}$ & ${b['IP_agg']:.3f}$ & "
                   f"${b['IP_topic']:.3f}$ & ${b['iota']:.3f}$ & "
                   f"${b['eta']:.3f}$ & ${b['beta']:.3f}$ \\\\")
    titles = {
        "omega":  r"$a_\omega$ -- regression-coefficient shrinkage",
        "beta":   r"$a_\beta$ -- topic--word sparsity",
        "rho":    r"$a_\rho$ -- polarity-loading shrinkage",
        "theta":  r"$a_\theta$ -- document--topic sparsity",
        "ideal":  r"$a_I, b_I$ -- ideal-point precision",
    }
    last_group = None
    for _, r in HP_SENSITIVITY.iterrows():
        if r["value"] == 0.3:
            continue   # baseline row already printed
        if r["group"] != last_group:
            out.append("\\addlinespace[0.3em]")
            out.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{titles[r['group']]}}}}} \\\\")
            last_group = r["group"]
        out.append(f"& ${r['value']:.1f}$ & ${r['IP_agg']:.3f}$ & "
                   f"${r['IP_topic']:.3f}$ & ${r['iota']:.3f}$ & "
                   f"${r['eta']:.3f}$ & ${r['beta']:.3f}$ \\\\")
    return "\n".join(out)


def latex_mc_fixed():
    if MC_FIXED is None:
        return "% MC_FIXED missing"
    out = ["% --- mc_fixed body ---"]
    out.append(f"TP & {_ms_int(MC_FIXED, 'TP')} \\\\")
    out.append(f"FP & {_ms_int(MC_FIXED, 'FP')} \\\\")
    out.append(f"FN & {_ms_int(MC_FIXED, 'FN')} \\\\")
    out.append(f"TN & {_ms_int(MC_FIXED, 'TN')} \\\\")
    out.append(f"Precision & {_ms(MC_FIXED, 'precision')} \\\\")
    out.append(f"Recall & $\\mathbf{{{float(MC_FIXED.set_index('metric').loc['recall', 'mean']):.3f}\\pm {float(MC_FIXED.set_index('metric').loc['recall', 'std']):.3f}}}$ \\\\")
    out.append(f"Specificity & {_ms(MC_FIXED, 'specificity')} \\\\")
    out.append(f"$\\alpha^*$ (active cells of c2--c4) & {_ms(MC_FIXED, 'alpha_star')} \\\\")
    out.append(f"flat-cor$(\\hat\\iota,\\iota^{{sim}})$ & {_ms(MC_FIXED, 'cor_iota_flat')} \\\\")
    out.append(f"$|\\mathrm{{cor}}|(\\text{{ideal}}_k^{{sim}},\\hat{{\\text{{ideal}}}}_k)$ topic-mean & "
               f"{_ms(MC_FIXED, 'mean_abs_cor_ideal_topic')} \\\\")
    return "\n".join(out)


def latex_mc_dgp():
    if MC_DGP is None:
        return "% MC_DGP missing"
    out = ["% --- mc_dgp body ---"]
    out.append(f"TP & {_ms_int(MC_DGP, 'TP')} \\\\")
    out.append(f"FP & {_ms_int(MC_DGP, 'FP')} \\\\")
    out.append(f"FN & {_ms_int(MC_DGP, 'FN')} \\\\")
    out.append(f"TN & {_ms_int(MC_DGP, 'TN')} \\\\")
    out.append(f"Precision & {_ms(MC_DGP, 'precision')} \\\\")
    out.append(f"Recall & $\\mathbf{{{float(MC_DGP.set_index('metric').loc['recall', 'mean']):.3f}\\pm {float(MC_DGP.set_index('metric').loc['recall', 'std']):.3f}}}$ \\\\")
    out.append(f"Specificity & {_ms(MC_DGP, 'specificity')} \\\\")
    out.append(f"$\\alpha^*$ (active cells of c2--c4) & {_ms(MC_DGP, 'alpha_star')} \\\\")
    out.append(f"flat-cor$(\\hat\\iota,\\iota^{{sim}})$ & {_ms(MC_DGP, 'cor_iota_flat')} \\\\")
    out.append(f"$|\\mathrm{{cor}}|(\\text{{ideal}}_k^{{sim}},\\hat{{\\text{{ideal}}}}_k)$ topic-mean & "
               f"{_ms(MC_DGP, 'mean_abs_cor_ideal_topic')} \\\\")
    return "\n".join(out)


def latex_misspec_down():
    if MS_DOWN_G is None or MS_DOWN_PC is None:
        return "% misspec_down missing"
    g = MS_DOWN_G
    out = ["% --- misspec_down global body ---"]
    out.append(f"$\\mathrm{{cor}}(\\hat x_a, \\bar x^{{sim}}_a)$ Pearson & {_ms(g, 'cor_x_pearson_mean')} \\\\")
    out.append(f"$\\mathrm{{cor}}(\\hat x_a, x^{{sim,\\text{{polmean}}}}_a)$ & {_ms(g, 'cor_x_pearson_polmean')} \\\\")
    out.append(f"$|\\mathrm{{cor}}(\\hat x_a, \\text{{party}}_a)|$ & "
               f"$\\mathbf{{{float(g.set_index('metric').loc['abs_cor_x_party', 'mean']):.3f}\\pm {float(g.set_index('metric').loc['abs_cor_x_party', 'std']):.3f}}}$ \\\\")
    out.append(f"$\\mathrm{{cor}}(\\hat\\iota_l, \\bar\\iota^{{sim}}_j)$ & {_ms(g, 'cor_iota_l')} \\\\")
    out.append(f"mean$_k\\,|\\mathrm{{cor}}(\\eta^{{sim}}_k,\\hat\\eta_k)|$ & {_ms(g, 'mean_abs_cor_eta_topic')} \\\\")
    out.append("\n% --- misspec_down per_covariate body ---")
    for _, r in MS_DOWN_PC.iterrows():
        cov_tex = str(r["covariate"]).replace("_", r"\_")
        out.append(
            f"\\texttt{{{cov_tex}}} & "
            f"${float(r['iota_sim_mean_mean']):+.3f}\\pm {float(r['iota_sim_mean_std']):.3f}$ & "
            f"${float(r['iota_hat_mean']):+.3f}\\pm {float(r['iota_hat_std']):.3f}$ & "
            f"${float(r['iota_sim_std_over_k_mean']):.3f}$ & "
            f"${float(r['detect_rate']):.2f}$ \\\\"
        )
    return "\n".join(out)


def latex_misspec_up():
    if MS_UP_G is None or MS_UP_PC is None:
        return "% misspec_up missing"
    g = MS_UP_G
    out = ["% --- misspec_up global body ---"]
    out.append(f"$\\mathrm{{cor}}(\\hat x^{{\\text{{pol}}}}_a, x^{{sim}}_a)$ Pearson & "
               f"$\\mathbf{{{float(g.set_index('metric').loc['cor_x_polmean_pearson', 'mean']):.3f}\\pm {float(g.set_index('metric').loc['cor_x_polmean_pearson', 'std']):.3f}}}$ \\\\")
    out.append(f"$\\mathrm{{cor}}(\\hat x^{{\\text{{pol}}}}_a, x^{{sim}}_a)$ Spearman & {_ms(g, 'cor_x_polmean_spearman')} \\\\")
    out.append(f"$\\overline{{\\mathrm{{cor}}_k}}(\\hat x_{{:,k}}, x^{{sim}})$ & "
               f"$\\mathbf{{{float(g.set_index('metric').loc['mean_cor_per_topic', 'mean']):.3f}\\pm {float(g.set_index('metric').loc['mean_cor_per_topic', 'std']):.3f}}}$ \\\\")
    out.append(f"$\\mathrm{{std}}_k\\,\\mathrm{{cor}}(\\hat x_{{:,k}}, x^{{sim}})$ & {_ms(g, 'std_cor_per_topic')} \\\\")
    out.append(f"flat-cor$(\\hat\\iota,\\iota^{{sim}})$ & {_ms(g, 'cor_iota_flat')} \\\\")
    out.append(f"TP & {_ms_int(g, 'TP')} \\\\")
    out.append(f"FP & {_ms_int(g, 'FP')} \\\\")
    out.append(f"Recall & {_ms(g, 'recall')} \\\\")
    out.append(f"Precision & {_ms(g, 'precision')} \\\\")
    out.append(f"Specificity & "
               f"$\\mathbf{{{float(g.set_index('metric').loc['specificity', 'mean']):.3f}\\pm {float(g.set_index('metric').loc['specificity', 'std']):.3f}}}$ \\\\")
    return "\n".join(out)


# ============================================================== #
# Print a verification report
# ============================================================== #
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" RESULTS dictionary keys")
    print("=" * 70)
    for k, v in RESULTS.items():
        print(f"  {k}")
        if isinstance(v, dict):
            for sub_k in v:
                payload = v[sub_k]
                n = len(payload) if payload is not None else 0
                print(f"    .{sub_k}: {n} record(s)")
        else:
            n = len(v) if v is not None else 0
            print(f"    : {n} record(s)")

    print("\n" + "=" * 70)
    print(" LATEX SNIPPETS (paste into docs/simulation_reply.tex)")
    print("=" * 70)

    print("\n>>> Table 1: seed sensitivity")
    print(latex_seeds())

    print("\n>>> Table 2: K sensitivity")
    print(latex_K())

    print("\n>>> Table 3: hyperparameter sensitivity")
    print(latex_hyperparameters())

    print("\n>>> Table 4: mc_fixed")
    print(latex_mc_fixed())

    print("\n>>> Table 5: mc_dgp")
    print(latex_mc_dgp())

    print("\n>>> Table 6+7: misspec_down")
    print(latex_misspec_down())

    print("\n>>> Table 8: misspec_up")
    print(latex_misspec_up())

    print("\n" + "=" * 70)
    print(" Done. Cross-check the numbers above against simulation_reply.tex.")
    print("=" * 70)
