#!/usr/bin/env python3
"""
03c_wordclouds_colored_R1.py
============================
Recolour the topic word clouds with a CONSISTENT ideological colour code
(referee request: "words should be blue/grey/red for left-wing, neutral, and
right-wing words").

Every word is coloured by its topic-specific polarity value eta_kv on a single
GLOBAL diverging scale:

    eta < 0  ->  blue   (#2c6fbb, used more by liberal/Democratic speakers)
    eta ~ 0  ->  grey   (#7f7f7f, neutral)
    eta > 0  ->  red    (#c0392b, used more by conservative/Republican speakers)

These are exactly the party colours used for the ideal points in Figure 4
(Democrats blue, Republicans red), so the colour code is now consistent across
all plots. The scale is global (same eta -> same colour in every panel and
every topic).

Layout per topic matches 03b_generate_wordclouds_logscale.py (6 panels):
    top:    E[log b] - eta | E[log b] | E[log b] + eta   (size = frequency at ideal -1/0/+1)
    bottom: -eta           | <label>  | +eta             (size = polarity, beta>thr)

Reads the PAPER fit (CSV params) by default:
    originalPolAn_results/fits/TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25/params/
        beta_shp.csv, beta_rte.csv, eta_loc.csv          (25 x 5031)
(falls back to *_final.npy if CSVs are absent, so it also works on .npy runs).

Outputs (all tagged _R1):
    <fit>/figs_R1/wordcloud_logscale_k_<k>_R1.png        (all 25 topics)
    <fit>/figs_R1/wordclouds_selected_topics_R1.png      (composite, paper topics)

Usage:
    python3 03c_wordclouds_colored_R1.py            # paper fit, all outputs
    python3 03c_wordclouds_colored_R1.py --results-dir <dir>
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.special import digamma
from wordcloud import WordCloud

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIT = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "originalPolAn_results", "fits",
    "TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25"))
DEFAULT_VOCAB = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "STBS_CAVI", "data", "hein-daily", "clean",
    "vocabulary114.txt"))

# Party / ideology colours -- identical to the ideal-point plots.
BLUE, GREY, RED = "#2c6fbb", "#7f7f7f", "#c0392b"
CMAP = LinearSegmentedColormap.from_list("blue_grey_red", [BLUE, GREY, RED])

# Selected topics shown in the main-text figure (paper numbering).
SELECTED = [4, 9, 11, 13, 16, 24]
_TOPIC_LABELS = [
    "National Security", "Supreme Court", "National Parks & Coast Guard",
    "Human Trafficking", "Commemoration & Anniversaries", "Gun Violence",
    "Middle Class & Small Business", "Health Care (ACA)", "Public Health (Zika)",
    "Veterans & Health Care", "Drugs & Addiction", "Climate Change",
    "Natural Resources & Energy", "Planned Parenthood & Abortion",
    "National Institutes of Health", "Middle East & Nuclear Weapons",
    "Immigration & Homeland Security", "Social Security & Taxes",
    "Bipartisan Rhetoric", "Clean Water Act", "Law Enforcement",
    "Wars & Civil Rights", "Education for Children", "Cyber Security",
    "Export, Import & Business",
]
LABELS = {k: lab for k, lab in enumerate(_TOPIC_LABELS)}


def load_params(results_dir):
    p = os.path.join(results_dir, "params")
    csv = os.path.join(p, "beta_shp.csv")
    if os.path.exists(csv):
        beta_shp = pd.read_csv(csv, index_col=0).to_numpy()
        beta_rte = pd.read_csv(os.path.join(p, "beta_rte.csv"), index_col=0).to_numpy()
        eta_loc = pd.read_csv(os.path.join(p, "eta_loc.csv"), index_col=0).to_numpy()
    else:                                              # .npy fallback
        beta_shp = np.load(os.path.join(p, "beta_shape_final.npy"))
        beta_rte = np.load(os.path.join(p, "beta_rate_final.npy"))
        eta_loc = np.load(os.path.join(p, "eta_location_final.npy"))
    return beta_shp, beta_rte, eta_loc


def load_vocab(path):
    return np.array([l.strip() for l in open(path) if l.strip()])


def make_color_func(eta_by_word, norm):
    def f(word, **kwargs):
        r, g, b, _ = CMAP(norm(eta_by_word.get(word, 0.0)))
        return f"rgb({int(255 * r)}, {int(255 * g)}, {int(255 * b)})"
    return f


def shift_positive(y):
    """WordCloud needs positive weights; same linear shift as the original."""
    return y - 1.05 * y.min() + 0.05 * y.max()


def cloud(ax, freqs, color_func, title=None, nwords=50):
    wc = WordCloud(collocations=False, background_color="white",
                   prefer_horizontal=1.0, max_words=nwords, width=400,
                   height=200, color_func=color_func)
    wc.generate_from_frequencies(freqs)
    ax.imshow(wc, interpolation="bilinear")
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def topic_panels(k, beta_shp, beta_rte, eta_loc, vocab, norm, thr=0.01):
    """Return the five frequency dicts + colour func for topic k.

    neg/neu/pos use a JOINT positive shift over the three columns and the
    eta panels a joint shift over the pair -- exactly the convention of the
    paper's plot_wordclouds / plot_wordclouds_slides, so word sizes match
    the published figures (only the colour code is changed)."""
    Elog = digamma(beta_shp[k]) - np.log(beta_rte[k])
    eta = eta_loc[k]
    Ebeta = beta_shp[k] / beta_rte[k]
    mask = (Ebeta > thr).astype(float)
    cf = make_color_func({vocab[v]: float(eta[v]) for v in range(len(vocab))}, norm)

    raw = np.stack([Elog - eta, Elog, Elog + eta], axis=1)       # joint shift
    raw = raw - 1.05 * raw.min() + 0.05 * raw.max()
    neg, neu, pos = raw[:, 0], raw[:, 1], raw[:, 2]

    pair = np.stack([-eta * mask, +eta * mask], axis=1)          # joint shift (pair)
    pair = pair - 1.05 * pair.min() + 0.05 * pair.max()
    yeta_neg, yeta_pos = pair[:, 0], pair[:, 1]

    d = lambda y: {vocab[v]: float(y[v]) for v in range(len(vocab))}
    return cf, d(neg), d(neu), d(pos), d(yeta_neg), d(yeta_pos)


def plot_topic(k, parts, out_path):
    cf, neg, neu, pos, en, ep = parts
    fig, ax = plt.subplots(2, 3, figsize=(11, 5))
    cloud(ax[0, 0], neg, cf, r"$\mathbb{E}\,\log\beta - \eta$  (liberal)")
    cloud(ax[0, 1], neu, cf, r"$\mathbb{E}\,\log\beta$  (neutral)")
    cloud(ax[0, 2], pos, cf, r"$\mathbb{E}\,\log\beta + \eta$  (conservative)")
    cloud(ax[1, 0], en, cf, r"$-\eta$  (most liberal terms)")
    ax[1, 1].text(0.5, 0.5, f"Topic {k}\n{LABELS.get(k, '')}", fontsize=14,
                  ha="center", va="center", transform=ax[1, 1].transAxes)
    ax[1, 1].axis("off")
    cloud(ax[1, 2], ep, cf, r"$+\eta$  (most conservative terms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_composite(rows, out_path):
    """rows: list of (k, cf, neg, neu, pos). 3 columns: liberal/neutral/conservative.
    The topic name is written centred BELOW each row of word clouds."""
    n = len(rows)
    fig, ax = plt.subplots(n, 3, figsize=(11, 2.4 * n))
    cols = ["Liberal (ideal $-1$)", "Neutral (ideal $0$)", "Conservative (ideal $+1$)"]
    for r, (k, cf, neg, neu, pos) in enumerate(rows):
        for c, freqs in enumerate([neg, neu, pos]):
            cloud(ax[r, c], freqs, cf, cols[c] if r == 0 else None, nwords=40)
        # topic name centred under the row (xlabel of the middle column)
        ax[r, 1].set_xlabel(f"Topic {k}: {LABELS.get(k, '')}", fontsize=12,
                            fontweight="bold", labelpad=6)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(rows, out_path, headers=("Negative", "Neutral", "Positive")):
    """Supplement-style grid: n topics (one per row) x 3 columns
    (Negative/Neutral/Positive = ideal -1/0/+1). Topic name + number centred
    BELOW each row. Same blue/grey/red colour code."""
    n = len(rows)
    fig, ax = plt.subplots(n, 3, figsize=(11, 2.4 * n), squeeze=False)
    for r, (k, cf, neg, neu, pos) in enumerate(rows):
        for c, freqs in enumerate([neg, neu, pos]):
            cloud(ax[r, c], freqs, cf, headers[c] if r == 0 else None, nwords=40)
        ax[r, 1].set_xlabel(f"Topic {k}: {LABELS.get(k, '')}", fontsize=12,
                            fontweight="bold", labelpad=6)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_FIT)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--nwords", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()

    beta_shp, beta_rte, eta_loc = load_params(args.results_dir)
    vocab = load_vocab(args.vocab)
    K, V = beta_shp.shape
    assert V == len(vocab), f"vocab mismatch: beta V={V} vs {len(vocab)}"

    # GLOBAL diverging scale: same eta -> same colour everywhere.
    M = float(np.quantile(np.abs(eta_loc), 0.99))
    norm = Normalize(vmin=-M, vmax=+M, clip=True)
    print(f"Global eta colour scale: +/-{M:.3f} (99th pct of |eta|)")

    out_dir = os.path.join(args.results_dir, "figs_R1")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for k in range(K):
        parts = topic_panels(k, beta_shp, beta_rte, eta_loc, vocab, norm,
                             thr=args.threshold)
        plot_topic(k, parts, os.path.join(out_dir, f"wordcloud_logscale_k_{k}_R1.png"))
        cf, neg, neu, pos, _, _ = parts
        all_rows.append((k, cf, neg, neu, pos))
        print(f"  [{k:2d}/{K-1}] wordcloud_logscale_k_{k}_R1.png")

    # main-text composite of the selected topics (Liberal/Neutral/Conservative)
    comp = [r for r in all_rows if r[0] in SELECTED]
    comp.sort(key=lambda t: SELECTED.index(t[0]))
    plot_composite(comp, os.path.join(out_dir, "wordclouds_selected_topics_R1.png"))
    print(f"\n-> wordclouds_selected_topics_R1.png (main-text composite)")

    # supplement grids: ALL topics, 5 per figure, Negative/Neutral/Positive
    for i in range(0, K, 5):
        chunk = all_rows[i:i + 5]
        a, b = chunk[0][0], chunk[-1][0]
        out = os.path.join(out_dir, f"wordclouds_topics_{a}-{b}_R1.png")
        plot_grid(chunk, out)
        print(f"-> {os.path.basename(out)}")
    print(f"\nAll outputs in {out_dir}/ (25 per-topic + 1 composite + "
          f"{(K + 4) // 5} grids, all _R1)")


if __name__ == "__main__":
    main()
