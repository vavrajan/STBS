#!/usr/bin/env python3
"""
19_distinctive_words.py
=======================
Most ideologically distinctive words per topic, computed on the
ORIGINAL PolAn fit (R2: "tables of top words or polarity terms are
usually clearer [than word clouds]"; R3: colour-code words red /
blue if clouds are kept).

Ranking follows the convention of the published word clouds: words
are ranked by the ideological loading eta_kv itself, restricted to
relevant words with E[log beta_kv] > -1 (digamma(shp) - log(rte)).
eta_kv > 0  -> word amplified at conservative ideal points (red),
eta_kv < 0  -> word amplified at liberal ideal points (blue).
This is the "eta contribution conditional on E log beta_kv > -1"
of the original supplement, presented as a table (R2) and as a
single red/blue cloud per topic (R3) instead of three separate
clouds per topic.

Outputs (results_simulation/distinctive_words/):
    distinctive_words_table.tex   top-10 R / top-10 D words per topic
    distinctive_words.csv         full ranking (all filtered words)
    diff_wordcloud_topic_KK.png   one red/blue cloud per topic
    diff_wordclouds_topics_A_B.png  5-topic sheets (drop-in for the
                                    old 3-column cloud pages)
"""
import os
import numpy as np
import pandas as pd
from scipy.special import digamma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

REPO = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.normpath(os.path.join(REPO, ".."))
POLAN = os.path.join(PROJ, "originalPolAn_results", "fits",
                     "TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25", "params")
CLEAN = os.path.join(PROJ, "STBS_CAVI", "data", "hein-daily", "clean")
OUT = os.path.join(REPO, "results_simulation", "distinctive_words")
os.makedirs(OUT, exist_ok=True)

TOPIC_LABELS = [
    "National Security", "Supreme Court", "Coast Guard", "Human Trafficking",
    "Commemoration and Anniversaries", "Gun Violence",
    "Middle Class and Small Businesses", "Health Care", "Public Health (Zika)",
    "Veterans and Health Care", "Drugs and Addiction", "Climate Change",
    "Natural Resources", "Planned Parenthood and Abortion",
    "Institutes and Research", "Middle East and Nuclear Weapons",
    "Immigration and DHS", "Social Security and Taxes",
    "Rhetorics and Discussion", "Clean Water Act", "Law Enforcement",
    "Wars and Human Rights", "Education for Children", "Cyber Security",
    "Export, Import and Business"]

RED, BLUE = "#c0392b", "#2874a6"
N_TABLE = 10          # words per direction in the LaTeX table
N_CLOUD = 40          # words per cloud
ELOGB_MIN = -1.0      # relevance filter, as in the published clouds


def latex_escape(s):
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


def main():
    bshp = pd.read_csv(os.path.join(POLAN, "beta_shp.csv"), index_col=0).to_numpy()
    brte = pd.read_csv(os.path.join(POLAN, "beta_rte.csv"), index_col=0).to_numpy()
    eta = pd.read_csv(os.path.join(POLAN, "eta_loc.csv"), index_col=0).to_numpy()
    beta = bshp / brte
    ElogB = digamma(bshp) - np.log(brte)
    vocab = np.array([l.strip() for l in
                      open(os.path.join(CLEAN, "vocabulary114.txt"))])
    K, V = beta.shape
    print(f"PolAn fit: beta {beta.shape}, eta {eta.shape}, vocab {len(vocab)}")

    rows, csv_rows = [], []
    for k in range(K):
        score = eta[k]                      # paper convention: rank by eta
        mask = ElogB[k] > ELOGB_MIN
        idx = np.where(mask)[0]
        order = idx[np.argsort(score[idx])]
        top_D = order[:N_TABLE]                  # most negative
        top_R = order[::-1][:N_TABLE]            # most positive
        rows.append(dict(k=k, label=TOPIC_LABELS[k],
                         words_R=", ".join(vocab[top_R]),
                         words_D=", ".join(vocab[top_D])))
        for i in idx:
            csv_rows.append(dict(k=k, label=TOPIC_LABELS[k], word=vocab[i],
                                 eta=eta[k, i], beta=beta[k, i],
                                 ElogB=ElogB[k, i]))

        # ---- one red/blue cloud per topic ----
        cloud_idx = idx[np.argsort(-np.abs(score[idx]))][:N_CLOUD]
        freqs = {vocab[i]: float(abs(score[i])) for i in cloud_idx
                 if abs(score[i]) > 0}
        colors = {vocab[i]: (RED if score[i] > 0 else BLUE)
                  for i in cloud_idx}
        if freqs:
            wc = WordCloud(width=900, height=450, background_color="white",
                           prefer_horizontal=0.95,
                           color_func=lambda word, **kw: colors.get(word, "grey"))
            wc.generate_from_frequencies(freqs)
            wc.to_file(os.path.join(OUT, f"diff_wordcloud_topic_{k:02d}.png"))
        print(f"  k={k:2d} {TOPIC_LABELS[k]:<36} "
              f"topR: {vocab[top_R[0]]}, {vocab[top_R[1]]} | "
              f"topD: {vocab[top_D[0]]}, {vocab[top_D[1]]}")

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT, "distinctive_words.csv"),
                                  index=False)

    # ---- 5-topic sheets (drop-in replacement for the old cloud pages) ----
    for lo in range(0, K, 5):
        hi = min(lo + 5, K) - 1
        fig, axes = plt.subplots(5, 1, figsize=(10, 24))
        for ax, k in zip(axes, range(lo, hi + 1)):
            img = plt.imread(os.path.join(OUT, f"diff_wordcloud_topic_{k:02d}.png"))
            ax.imshow(img); ax.axis("off")
            ax.set_title(f"Topic {k}: {TOPIC_LABELS[k]}", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, f"diff_wordclouds_topics_{lo}_{hi}.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ---- LaTeX table (two sideways pages: topics 0-13 and 14-24) ----
    def table_part(part_rows, caption, label):
        t = []
        t.append(r"\begin{sidewaystable}")
        t.append(r"\centering\scriptsize")
        t.append(r"\caption{" + caption + "}")
        t.append(r"\label{" + label + "}")
        t.append(r"\setlength{\tabcolsep}{3pt}")
        t.append(r"\rowcolors{2}{gray!12}{white}")
        t.append(r"\begin{tabular}{r p{3.2cm} p{8.4cm} p{8.4cm}}")
        t.append(r"\toprule")
        t.append(r"$k$ & Topic & "
                 r"\textcolor{red2}{Conservative-leaning words ($\eta_{kv}>0$)} & "
                 r"\textcolor{mediumblue}{Liberal-leaning words ($\eta_{kv}<0$)} \\")
        t.append(r"\midrule")
        for r in part_rows:
            t.append(f"{r['k']} & {latex_escape(r['label'])} & "
                     f"{latex_escape(r['words_R'])} & "
                     f"{latex_escape(r['words_D'])} \\\\")
        t.append(r"\bottomrule")
        t.append(r"\end{tabular}")
        t.append(r"\end{sidewaystable}")
        return t

    cap_main = (r"Most ideologically distinctive bigrams per topic "
                r"Topics 0--13. Following the "
                r"convention of the word clouds, bigrams are ranked by "
                r"the ideological loading $\eta_{kv}$, restricted to "
                r"relevant terms with $\mathbb{E}\log\beta_{kv}>-1$: "
                r"listed are the ten bigrams with the largest positive "
                r"$\eta_{kv}$ (amplified at conservative ideal points) "
                r"and the ten with the most negative $\eta_{kv}$ "
                r"(amplified at liberal ideal points).")
    cap_cont = (r"Most ideologically distinctive bigrams per topic "
                r"continued: Topics 14--24. "
                r"Construction as in "
                r"Table~\ref{tab:distinctive_words}.")

    tex = ["% Auto-generated by 19_distinctive_words.py"]
    tex += table_part([r for r in rows if r["k"] <= 13],
                      cap_main, "tab:distinctive_words")
    tex.append("")
    tex += table_part([r for r in rows if r["k"] >= 14],
                      cap_cont, "tab:distinctive_words_cont")
    with open(os.path.join(OUT, "distinctive_words_table.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"\n-> {OUT}/distinctive_words_table.tex")
    print(f"-> {OUT}/distinctive_words.csv")
    print(f"-> {OUT}/diff_wordcloud_topic_KK.png + 5-topic sheets")


if __name__ == "__main__":
    main()
