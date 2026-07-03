#!/usr/bin/env python3
"""
26_format_regression_tables.py
==============================
Re-format the two appendix regression-coefficient tables that the referee
flagged as "poorly formatted" (and: "Table 2 should clearly be split into
six tables").

It PARSES the existing input tabulars (so every number comes straight from
the source -- no re-typing) and emits cleaned-up versions:

  docs/STBS_a_regression_coefs_R1.tex
        one tidy booktabs table (fixed ideological positions, no interaction)

  docs/STBS_ak_regression_coefs_k_4_9_11_13_16_24_R1.tex
        SIX separate per-topic tables (4, 9, 11, 13, 16, 24), interaction
        model -- one table each, portrait, no sidewaystable.

Improvements over the originals:
  * \\texttt{} code labels -> readable names; (1,10] -> "1--10 yrs" etc.
  * significance stars from the CCP, with a legend + baselines in the caption
  * main effects and party-interaction effects separated by a labelled rule
  * Joint CCP (the covariate-level test) kept, once per covariate

The originals are left untouched (both versions kept).
"""
import os
import re

REPO = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(REPO, "docs")
SRC1 = os.path.join(DOCS, "STBS_a_regression_coefs.tex")
SRC2 = os.path.join(DOCS, "STBS_ak_regression_coefs_k_4_9_11_13_16_24.tex")
OUT1 = os.path.join(DOCS, "STBS_a_regression_coefs_R1.tex")
OUT2 = os.path.join(DOCS, "STBS_ak_regression_coefs_k_4_9_11_13_16_24_R1.tex")

TOPICS = [(4,  "Commemoration and Anniversaries"),
          (9,  "Veterans and Health Care"),
          (11, "Climate Change"),
          (13, "Planned Parenthood and Abortion"),
          (16, "Immigration and Department of Homeland Security"),
          (24, "Export, Import and Business")]

COV = {
    "intercept":  "Intercept",
    "party":      "Party",
    "gender":     "Gender",
    "region":     "Region",
    "generation": "Generation",
    "experience": "Experience",
    "religion":   "Religion",
    "party_Republican:gender":     r"Party\,(R) $\times$ Gender",
    "party_Republican:region":     r"Party\,(R) $\times$ Region",
    "party_Republican:generation": r"Party\,(R) $\times$ Generation",
    "party_Republican:experience": r"Party\,(R) $\times$ Experience",
    "party_Republican:religion":   r"Party\,(R) $\times$ Religion",
}
CAT = {"(1,10]": "1--10 yrs", "(0,1]": r"$<$1 yr"}

CAPTION_NOTE = (r"Baselines: Party~=~Democrat, Gender~=~Male, Region~=~Northeast, "
                r"Generation~=~Silent, Experience~=~10+~yrs, Religion~=~Other. "
                r"Significance codes (complementary coverage probability, CCP): "
                r"$^{***}\,\text{CCP}<0.001$, $^{**}<0.01$, $^{*}<0.05$, "
                r"$^{\cdot}<0.1$.")


def clean_label(s):
    s = s.strip()
    m = re.search(r"\\multirow\{\d+\}\{\*\}\{(.*)\}\s*$", s)
    if m:
        s = m.group(1)
    m = re.search(r"\\texttt\{(.*)\}\s*$", s)
    if m:
        s = m.group(1)
    return s.replace(r"\_", "_").strip()


def val(s):
    m = re.search(r"\$([^$]*)\$", s)
    return m.group(1).strip() if m else ""


def parse(path):
    """-> list of groups: {'name', 'joint':[per-topic], 'rows':[(cat,[(est,se,ccp)])]}"""
    groups = []
    for raw in open(path):
        if "$" not in raw or raw.lstrip().startswith("%"):
            continue
        line = raw.rstrip()
        line = re.sub(r"\\\\\s*$", "", line)          # drop trailing \\
        f = line.split("&")
        coef, cat = clean_label(f[0]), clean_label(f[1])
        data = f[2:]
        nt = len(data) // 4
        per = []
        joint = []
        for t in range(nt):
            est, se, ccp, jt = (val(data[4 * t + j]) for j in range(4))
            per.append((est, se, ccp))
            joint.append(jt)
        if coef:
            groups.append({"name": coef, "joint": joint, "rows": [(cat, per)]})
        else:
            groups[-1]["rows"].append((cat, per))
    return groups


def stars(ccp):
    if ccp == "":
        return ""
    v = 0.0005 if ccp.startswith("<") else float(ccp)
    if v < 0.001:
        return "***"
    if v < 0.01:
        return "**"
    if v < 0.05:
        return "*"
    if v < 0.1:
        return r"\cdot"
    return ""


def ccp_cell(ccp):
    if ccp == "":
        return ""
    core = "<0.001" if ccp.startswith("<") else ccp
    s = stars(ccp)
    return f"${core}^{{{s}}}$" if s else f"${core}$"


def num(x):
    return f"${x}$" if x else ""


def cov_name(raw):
    return COV.get(raw, raw)


def cat_name(raw):
    return CAT.get(raw, raw)


def emit_table(groups, t, caption, label):
    """Emit one booktabs table for topic-column index t."""
    L = []
    L.append(r"\begin{table}[!htbp]")
    L.append(r"\centering\footnotesize")
    L.append(r"\setlength{\tabcolsep}{5pt}")
    L.append(r"\caption{" + caption + r"}")
    L.append(r"\label{" + label + r"}")
    L.append(r"\begin{tabular}{@{}ll rrr r@{}}")
    L.append(r"\toprule")
    L.append(r"Covariate & Category & Estimate & SE & CCP & Joint CCP\\")
    L.append(r"\midrule")
    seen_int = False
    for g in groups:
        is_int = ":" in g["name"]
        if is_int and not seen_int:
            seen_int = True
            L.append(r"\midrule")
            L.append(r"\multicolumn{6}{@{}l}{\itshape Interactions with Party "
                     r"(Republican)}\\")
            L.append(r"\addlinespace[1pt]")
        jcell = ccp_cell(g["joint"][t])
        for i, (cat, per) in enumerate(g["rows"]):
            est, se, ccp = per[t]
            cov = r"\textbf{%s}" % cov_name(g["name"]) if i == 0 else ""
            jc = jcell if i == 0 else ""
            L.append(f"{cov} & {cat_name(cat)} & {num(est)} & {num(se)} & "
                     f"{ccp_cell(ccp)} & {jc}\\\\")
        L.append(r"\addlinespace[2pt]")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    # ---- Table 1: fixed positions, single column block ----
    g1 = parse(SRC1)
    cap1 = (r"Posterior estimates of the regression coefficients for the model "
            r"with fixed ideological positions across topics (no interactions). "
            + CAPTION_NOTE)
    with open(OUT1, "w") as f:
        f.write("% Auto-generated by 26_format_regression_tables.py\n")
        f.write(emit_table(g1, 0, cap1, "tab:regression_coefs") + "\n")
    print(f"-> {OUT1}  ({len(g1)} covariate blocks)")

    # ---- Table 2: interaction model, split into six per-topic tables ----
    g2 = parse(SRC2)
    parts = ["% Auto-generated by 26_format_regression_tables.py",
             "% Table 2 split into six per-topic tables (referee request)."]
    for t, (knum, kname) in enumerate(TOPICS):
        cap = (f"Topic~{knum} ({kname}): posterior estimates of the regression "
               r"coefficients for the model with topic-specific ideological "
               r"positions (interaction model). " + CAPTION_NOTE +
               r" The Party\,(R)$\times$Religion interaction for Jewish is not "
               r"identified (no Republican Jewish Senators) and is fixed at zero.")
        parts.append(emit_table(g2, t, cap, f"tab:reg_coefs_k{knum}"))
        parts.append("")
    with open(OUT2, "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"-> {OUT2}  ({len(g2)} covariate blocks x {len(TOPICS)} topics)")


if __name__ == "__main__":
    main()
