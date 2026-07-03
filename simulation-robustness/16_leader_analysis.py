#!/usr/bin/env python3
"""
16_leader_analysis.py
=====================
Three-part analysis of party-leadership ideological positions in the
114th Senate, on the basis of the original PolAn variational fit.

Part A -- "Do party leaders drive the ideal points?"
    For each leader and each topic, compute the z-score of the
    leader's ideal point with respect to the distribution of the
    remaining members of their own party:
        z_{leader,k} = (ideal_{leader,k} - mean_{a in party \ leader} ideal_{a,k})
                       / std_{a in party \ leader} ideal_{a,k}
    A large positive z means the leader is more extreme on the right
    end of the party than the rest of their party; a large negative
    z means more extreme on the left end. Output: a single CSV with
    one row per (leader, topic) plus a summary that flags topics
    where the leader's |z| > 1 (i.e. the leader is more than one
    party-standard-deviation more extreme than the typical party
    member on that topic).

Part B -- Plot of leader ideal points across all 25 topics
    A line plot with the six party leaders on a common axis:
    x = topic index 0..24, y = ideal point.

Part C -- Speech excerpts with wordcloud-direction bold words
    For the two top party-polarizing topics (PolAn Topic 5 = Gun
    Violence, PolAn Topic 16 = Immigration / DHS), retrieve one
    illustrative left-leaning and one right-leaning speech (from the
    CAVI fit, since the PolAn fit lacks the per-document speech-id permutation),
    apply the canonical PolAn-to-CAVI topic mapping to pick the matching
    CAVI topic, and bold-print all words/phrases that appear in the
    top-30 "right-direction" or "left-direction" wordcloud for that
    PolAn topic.

Outputs in results_simulation/leader_analysis/
    leader_zscores.csv
    leader_ips_plot.pdf / .png
    leader_topics_with_extreme_z.csv
    topic_KK_excerpts.tex   (LaTeX-formatted excerpts with \textbf{})
    polan_to_cavi_topic_map.csv
"""
import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm

REPO = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.normpath(os.path.join(REPO, ".."))

POLAN = os.path.join(PROJ, "originalPolAn_results", "fits",
                    "TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25", "params")
CAVI = os.path.join(REPO, "stbs_cavi_results", "seed_123456_K25", "params")
CLEAN = os.path.join(PROJ, "STBS_CAVI", "data", "hein-daily", "clean")
HEIN = os.path.join(PROJ, "hein_daily")
SPEAKER_MAP = "/Users/paul.hofmarcher/Documents/svn/baR/Projects/Congress_Speeches/hein-daily/114_SpeakerMap.txt"
INFLU_DIR = os.path.join(REPO, "results_simulation", "influential_speeches")
OUT = os.path.join(REPO, "results_simulation", "leader_analysis")

# Same six leaders as 15_influential_polan.py
PARTY_LEADERS = [
    {"name": "MITCH MCCONNELL", "party": "R", "color": "#cc0000",
     "role": "Majority Leader"},
    {"name": "JOHN CORNYN",     "party": "R", "color": "#e85a3a",
     "role": "Majority Whip"},
    {"name": "ORRIN HATCH",     "party": "R", "color": "#f0a070",
     "role": "President pro tempore"},
    {"name": "HARRY REID",      "party": "D", "color": "#1f3a93",
     "role": "Minority Leader (until Jan 2017)"},
    {"name": "RICHARD DURBIN",  "party": "D", "color": "#3066be",
     "role": "Minority Whip"},
    {"name": "CHARLES SCHUMER", "party": "D", "color": "#5599e0",
     "role": "Minority Leader (from Jan 2017)"},
]


# ============================================================== #
def load_polan():
    bshp = pd.read_csv(os.path.join(POLAN, "beta_shp.csv"), index_col=0).to_numpy()
    brte = pd.read_csv(os.path.join(POLAN, "beta_rte.csv"), index_col=0).to_numpy()
    beta = bshp / brte
    eta = pd.read_csv(os.path.join(POLAN, "eta_loc.csv"), index_col=0).to_numpy()
    ideal = np.load(os.path.join(POLAN, "ideal_point_location.npy"))
    return beta, eta, ideal


def load_CAVI_beta():
    bshp = np.load(os.path.join(CAVI, "beta_shape_final.npy"))
    brte = np.load(os.path.join(CAVI, "beta_rate_final.npy"))
    return bshp / brte


def load_author_map():
    with open(os.path.join(CLEAN, "author_map114.txt")) as f:
        names = [l.strip() for l in f if l.strip()]
    parties = np.array([re.search(r"\((\w)\)\s*$", n).group(1) for n in names])
    return names, parties


def load_vocabulary():
    with open(os.path.join(CLEAN, "vocabulary114.txt")) as f:
        return [w.strip() for w in f if w.strip()]


def find_idx(name, names):
    target = name.strip().upper()
    for i, nm in enumerate(names):
        if nm.split("(")[0].strip().upper() == target:
            return i
    return None


# ============================================================== #
# Part A: Z-score against own party (excluding the leader)
# ============================================================== #
def compute_leader_zscores(ideal, names, parties):
    K = ideal.shape[1]
    rows = []
    for leader in PARTY_LEADERS:
        a = find_idx(leader["name"], names)
        if a is None:
            print(f"  WARN: {leader['name']} not in author_map")
            continue
        party_mask = (parties == leader["party"])
        party_mask[a] = False  # exclude leader from party-rest distribution
        for k in range(K):
            ip_leader = float(ideal[a, k])
            party_rest = ideal[party_mask, k]
            mu = float(party_rest.mean())
            sd = float(party_rest.std(ddof=1))
            z = (ip_leader - mu) / sd if sd > 0 else 0.0
            rows.append(dict(
                leader=leader["name"], party=leader["party"],
                role=leader["role"],
                topic=k,
                ideal_leader=ip_leader,
                party_rest_mean=mu, party_rest_std=sd,
                z=z, abs_z=abs(z),
            ))
    return pd.DataFrame(rows)


# ============================================================== #
# Part B: Plot leader ideal points across all 25 topics
# ============================================================== #
def plot_leader_ips(ideal, names, K, party_means, out_pdf):
    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(K)
    # Mean ideal for R / D for context (shaded bands)
    R_mean = party_means["R_mean"].values
    D_mean = party_means["D_mean"].values
    ax.plot(x, R_mean, color="#cc0000", lw=1.5, alpha=0.35,
             linestyle="--", label="party mean (R)")
    ax.plot(x, D_mean, color="#1f3a93", lw=1.5, alpha=0.35,
             linestyle="--", label="party mean (D)")
    ax.fill_between(x, D_mean, R_mean, color="grey", alpha=0.05)

    for leader in PARTY_LEADERS:
        a = find_idx(leader["name"], names)
        if a is None:
            continue
        ax.plot(x, ideal[a], marker="o", markersize=5, lw=1.8,
                 color=leader["color"],
                 label=f"{leader['name'].title()} ({leader['party']})")

    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("Topic $k$")
    ax.set_ylabel(r"Ideal point $\hat{\imath}_{a,k}$  (PolAn fit)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x], fontsize=8)
    ax.set_title("Per-topic ideal points of 114th-Senate party leaders "
                 "(dashed = party mean across non-leadership senators)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    # Highlight the two top-polarising topics
    for k_hl, lbl in [(5, "5: Gun Violence"), (16, "16: Immigration/DHS")]:
        ax.axvspan(k_hl - 0.4, k_hl + 0.4, color="yellow", alpha=0.18, zorder=-1)
        ax.annotate(lbl, xy=(k_hl, 2.6), xytext=(k_hl, 2.8),
                     fontsize=7, ha="center", color="#705500")

    ax.set_ylim(-3.0, 3.0)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=160, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out_pdf}")


# ============================================================== #
# Part C: Speech excerpts with bold top-words
# ============================================================== #
def polan_top_words(beta_k, eta_k, vocab, direction, top_n=30):
    """Wordcloud-equivalent top words for PolAn topic k in a given
    direction. Convention from the supplement: positive col uses
    ideal=+1, negative col uses ideal=-1, so the word score is
    beta_k * exp(eta_k * sign).
    """
    sign = +1.0 if direction == "right" else -1.0
    score = beta_k * np.exp(eta_k * sign)
    idx = np.argsort(-score)[:top_n]
    return [vocab[v] for v in idx]


def bold_words_in_text(text, top_words):
    """Wrap every case-insensitive occurrence of each top-word phrase
    in `top_words` with \\textbf{...}. Phrases come from the vocab,
    e.g. 'gun violence' or 'border security' (bigrams).
    """
    # Sort longest first so that bigrams are matched before unigrams
    sorted_words = sorted(set(top_words), key=lambda s: -len(s))
    out = text
    for w in sorted_words:
        w_clean = w.replace("_", " ").strip()
        if not w_clean:
            continue
        pattern = re.compile(r"\b(" + re.escape(w_clean) + r")\b",
                              flags=re.IGNORECASE)
        out = pattern.sub(r"\\textbf{\1}", out)
    return out


def latex_escape(s):
    """Minimal LaTeX-escape for text inside a body paragraph."""
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
            "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\~{}", "^": r"\^{}"}
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def latex_escape_keep_textbf(s):
    """Escape special chars but preserve \\textbf{...} that we
    inserted ourselves. Placeholders use ASCII letters only so they
    survive latex_escape without modification.
    """
    placeholder_open = "ZZTBFOPENZZ"
    placeholder_close = "ZZTBFCLOSEZZ"
    # Replace each \textbf{...} occurrence pair-wise so that the
    # matching close brace is the one that closes \textbf, not some
    # other brace later in the speech.
    out = []
    i = 0
    while i < len(s):
        if s[i:].startswith("\\textbf{"):
            out.append(placeholder_open)
            i += len("\\textbf{")
            # find matching closing brace at depth 0 (no nested braces here)
            depth = 1
            while i < len(s) and depth > 0:
                ch = s[i]
                if ch == "{":
                    depth += 1
                    out.append(ch)
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(placeholder_close)
                    else:
                        out.append(ch)
                else:
                    out.append(ch)
                i += 1
        else:
            out.append(s[i])
            i += 1
    s2 = "".join(out)
    # Now escape LaTeX special chars (placeholders survive)
    s2 = latex_escape(s2)
    s2 = s2.replace(placeholder_open, r"\textbf{")
    s2 = s2.replace(placeholder_close, "}")
    return s2


def make_excerpt_block(speech_text, top_words, max_chars=1400):
    # Take the first chunk of the speech to keep it short
    text = speech_text[:max_chars]
    if len(speech_text) > max_chars:
        text += " ..."
    # Bold the top-words first (case-insensitive)
    text_with_bold = bold_words_in_text(text, top_words)
    # Then LaTeX-escape but keep the \textbf{...} markers
    text_escaped = latex_escape_keep_textbf(text_with_bold)
    return text_escaped


def load_topic_speeches_from_influ(cavi_topic, direction, n=2):
    """Pull the top-N speeches for a given CAVI topic + direction from
    the existing 14_influential_speeches.py output.
    """
    txt_path = os.path.join(INFLU_DIR, f"topic_{cavi_topic:02d}_{direction}.txt")
    if not os.path.exists(txt_path):
        return []
    with open(txt_path) as f:
        content = f.read()
    # Parse speech blocks separated by '# Speech rank X'
    blocks = re.split(r"^# Speech rank \d+\s*$", content,
                       flags=re.MULTILINE)
    speeches = []
    for b in blocks[1:1 + n + 5]:  # take a few extra in case of malformed
        m_speaker = re.search(r"Speaker \(descr\)\s*:\s*(.*)", b)
        m_author = re.search(r"Author \(model\)\s*:\s*(.*)", b)
        m_date = re.search(r"Date\s*:\s*(.*)", b)
        m_ideal = re.search(r"Ideal_\{a,k\}\s*:\s*(\S+)", b)
        m_theta = re.search(r"Theta_\{d,k\}\s*:\s*(\S+)", b)
        m_speech = re.search(r"Speech\s*:\s*(.*?)(?:\n\s*\.\.\.|\n\n)",
                              b, flags=re.DOTALL)
        if m_speech is None:
            continue
        speeches.append(dict(
            speaker=(m_speaker.group(1).strip() if m_speaker else "?"),
            author=(m_author.group(1).strip() if m_author else "?"),
            date=(m_date.group(1).strip() if m_date else "?"),
            ideal=(m_ideal.group(1).strip() if m_ideal else "?"),
            theta=(m_theta.group(1).strip() if m_theta else "?"),
            text=m_speech.group(1).strip(),
        ))
        if len(speeches) >= n:
            break
    return speeches


def write_topic_excerpt_tex(polan_topic, cavi_topic, topic_label,
                             beta_polan, eta_polan, vocab,
                             out_path, top_n_words=30,
                             n_speeches_per_dir=2):
    """Write a LaTeX snippet with bold top-word speech excerpts for
    one topic, both directions.
    """
    right_words = polan_top_words(beta_polan[polan_topic], eta_polan[polan_topic],
                                  vocab, "right", top_n_words)
    left_words = polan_top_words(beta_polan[polan_topic], eta_polan[polan_topic],
                                 vocab, "left", top_n_words)

    right_sp = load_topic_speeches_from_influ(cavi_topic, "right",
                                                n_speeches_per_dir)
    left_sp = load_topic_speeches_from_influ(cavi_topic, "left",
                                               n_speeches_per_dir)

    with open(out_path, "w") as fh:
        fh.write(f"% LaTeX snippet for PolAn Topic {polan_topic} "
                 f"= CAVI Topic {cavi_topic}\n")
        fh.write(f"% Bold words are the top-{top_n_words} "
                 f"wordcloud-direction words from the PolAn fit.\n")
        fh.write(f"\\paragraph{{Top {top_n_words} right-direction "
                 f"words ({topic_label}):}}\n")
        fh.write(", ".join(f"\\emph{{{latex_escape(w)}}}"
                            for w in right_words) + ".\n\n")
        fh.write(f"\\paragraph{{Top {top_n_words} left-direction "
                 f"words ({topic_label}):}}\n")
        fh.write(", ".join(f"\\emph{{{latex_escape(w)}}}"
                            for w in left_words) + ".\n\n")

        for sp_list, dir_label, words in [(right_sp, "right", right_words),
                                            (left_sp, "left", left_words)]:
            fh.write(f"\\paragraph{{Illustrative {dir_label}-leaning "
                     f"excerpts ({topic_label}):}}\n")
            for i, sp in enumerate(sp_list, 1):
                excerpt = make_excerpt_block(sp["text"], words)
                fh.write(f"\n\\smallskip\\noindent\\textbf{{Excerpt {i}}} "
                         f"-- {latex_escape(sp['author'])}, "
                         f"{latex_escape(sp['date'])}, "
                         f"$\\hat{{\\imath}}={latex_escape(sp['ideal'])}$, "
                         f"$\\theta={latex_escape(sp['theta'])}$.\n\n"
                         f"\\begingroup\\small\n{excerpt}\n\\endgroup\n")
            fh.write("\n")
    print(f"  -> wrote {out_path}")


# ============================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n-words", type=int, default=30)
    ap.add_argument("--excerpt-chars", type=int, default=1400)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("Loading PolAn + CAVI ...")
    beta_polan, eta_polan, ideal_polan = load_polan()
    beta_CAVI = load_CAVI_beta()
    names, parties = load_author_map()
    vocab = load_vocabulary()
    A, K = ideal_polan.shape
    print(f"  PolAn: beta {beta_polan.shape}, ideal {ideal_polan.shape}")
    print(f"  vocab: {len(vocab)}")

    # =================================================
    # Topic mapping PolAn -> CAVI (for excerpt retrieval)
    # =================================================
    sim = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            sim[i, j] = (beta_polan[i] @ beta_CAVI[j]) / (
                norm(beta_polan[i]) * norm(beta_CAVI[j]) + 1e-12)
    row, col = linear_sum_assignment(-sim)
    polan_to_cavi = {int(r): int(c) for r, c in zip(row, col)}
    map_df = pd.DataFrame([
        dict(PolAn_topic=k, CAVI_topic=polan_to_cavi[k],
             cosine=float(sim[k, polan_to_cavi[k]])) for k in range(K)
    ])
    map_df.to_csv(os.path.join(OUT, "polan_to_cavi_topic_map.csv"), index=False)
    print(f"  PolAn->CAVI map saved")

    # =================================================
    # Part A: leader z-scores
    # =================================================
    print("\nPart A: leader z-scores vs.\\ party-rest distribution")
    z_df = compute_leader_zscores(ideal_polan, names, parties)
    z_df.to_csv(os.path.join(OUT, "leader_zscores.csv"), index=False)
    extreme = z_df[z_df["abs_z"] > 1.0].sort_values(
        ["leader", "abs_z"], ascending=[True, False])
    extreme.to_csv(os.path.join(OUT, "leader_topics_with_extreme_z.csv"),
                   index=False)
    print(f"  wrote leader_zscores.csv ({len(z_df)} rows)")
    print(f"  wrote leader_topics_with_extreme_z.csv ({len(extreme)} rows)")
    print("\n  Leader: avg |z| and topics with |z|>1:")
    summary = z_df.groupby(["leader", "party"]).agg(
        avg_abs_z=("abs_z", "mean"),
        n_topics_z_gt_1=("abs_z", lambda x: int((x > 1).sum())),
        max_abs_z=("abs_z", "max")).round(3).reset_index()
    print(summary.to_string(index=False))

    # ----------------------------------------------------------------
    # Permutation test:  is the leader's avg|z| larger than what we
    # would get by treating an arbitrary non-leader same-party senator
    # as the 'leader' and computing the same avg|z| against the
    # remaining party-rest?
    #
    # We run the test twice:
    #   (i)  on all K=25 topics
    #   (ii) on the subset of POLARIZING topics (per-topic AUC > 0.9),
    #        where the partisan signal is cleanest
    # ----------------------------------------------------------------
    print("\n  Permutation test on avg|z| (leader-vs-non-leader):")
    K = ideal_polan.shape[1]
    leader_idx = []
    for L in PARTY_LEADERS:
        a = find_idx(L["name"], names)
        if a is not None:
            leader_idx.append(a)
    leader_idx = set(leader_idx)

    # --- per-topic AUC (R vs D, IPs as scores) --------------------
    try:
        from sklearn.metrics import roc_auc_score
        y = (parties == "R").astype(int)
        keep = (parties == "R") | (parties == "D")
        per_topic_auc = np.array([
            roc_auc_score(y[keep], ideal_polan[keep, k]) for k in range(K)
        ])
    except Exception as e:
        print(f"  WARN: sklearn unavailable ({e}); falling back to manual AUC")
        per_topic_auc = np.zeros(K)
        for k in range(K):
            xR = ideal_polan[parties == "R", k]
            xD = ideal_polan[parties == "D", k]
            wins = sum(r > d for r in xR for d in xD)
            ties = sum(r == d for r in xR for d in xD)
            per_topic_auc[k] = (wins + 0.5 * ties) / (len(xR) * len(xD))
    POLAR_TOPICS = np.where(per_topic_auc > 0.9)[0].tolist()
    print(f"  Polarizing topics (AUC > 0.9): {POLAR_TOPICS}")
    print(f"    -> {len(POLAR_TOPICS)} of {K} topics")

    def avg_abs_z_for(a_idx, party_char, topic_subset=None):
        """avg|z| of senator a_idx, scored against the rest of their party.
        topic_subset=None -> all K topics; else -> the given list of k's."""
        party_mask = (parties == party_char)
        party_mask[a_idx] = False
        ks = range(K) if topic_subset is None else topic_subset
        zs = []
        for k in ks:
            v = float(ideal_polan[a_idx, k])
            rest = ideal_polan[party_mask, k]
            mu, sd = float(rest.mean()), float(rest.std(ddof=1))
            zs.append(abs((v - mu) / sd) if sd > 0 else 0.0)
        return float(np.mean(zs))

    def run_perm(topic_subset, label):
        rows = []
        for L in PARTY_LEADERS:
            a = find_idx(L["name"], names)
            if a is None:
                continue
            actual = avg_abs_z_for(a, L["party"], topic_subset)
            null_dist = []
            for j in range(len(names)):
                if parties[j] != L["party"]:
                    continue
                if j == a or j in leader_idx:
                    continue
                null_dist.append(avg_abs_z_for(j, L["party"], topic_subset))
            null_dist = np.array(null_dist)
            p_one = float(((null_dist >= actual).sum() + 1)
                          / (len(null_dist) + 1))
            rank = int((null_dist >= actual).sum() + 1)
            rows.append(dict(
                subset=label,
                leader=L["name"], party=L["party"], role=L["role"],
                avg_abs_z=round(actual, 3),
                null_mean=round(float(null_dist.mean()), 3),
                null_max=round(float(null_dist.max()), 3),
                null_n=len(null_dist),
                rank=rank,
                p_perm=round(p_one, 4),
            ))
        return pd.DataFrame(rows)

    perm_all = run_perm(None,          "all_K25")
    perm_pol = run_perm(POLAR_TOPICS,  f"polarizing_AUC_gt_0.9_n{len(POLAR_TOPICS)}")
    perm_df = pd.concat([perm_all, perm_pol], ignore_index=True)
    perm_df.to_csv(os.path.join(OUT, "leader_permutation_test.csv"),
                   index=False)
    print("\n  All 25 topics:")
    print(perm_all.drop(columns=["role"]).to_string(index=False))
    print(f"\n  Polarizing topics only ({POLAR_TOPICS}):")
    print(perm_pol.drop(columns=["role"]).to_string(index=False))
    print(f"\n  -> leader_permutation_test.csv")

    # ---- LaTeX table: side-by-side all-K25 vs polarizing-only ----
    tex = []
    tex.append("% Auto-generated by 16_leader_analysis.py")
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering\small")
    cap = (r"\caption{Permutation test of leader extremity. "
           r"For each leader, the statistic $\mathrm{avg}\,|z|$ "
           r"(mean absolute z-score of the leader's per-topic ideal "
           r"point against the same-party rank-and-file distribution) "
           r"is compared against the empirical null obtained by "
           r"treating every non-leader same-party senator as a "
           r"pseudo-leader and re-computing the same statistic "
           r"against the remaining party-rest. The test is run on "
           r"(i) all $K{=}25$ topics and (ii) the subset of "
           r"\emph{polarizing} topics with per-topic AUC $>0.9$ "
           rf"($\{{{','.join(str(t) for t in POLAR_TOPICS)}\}}$ "
           rf"-- $n{{=}}{len(POLAR_TOPICS)}$ topics; see "
           r"Table~\ref{tab:face_validity_top3}). `Rank' is the "
           r"leader's position in the pooled list of "
           r"$(n_{\text{null}}{+}1)$ same-party senators (rank $1$ = "
           r"strictly more extreme than every non-leader). "
           r"$p_{\text{perm}}$ is the one-sided permutation $p$-value "
           r"with the usual $(n_{\geq}+1)/(n_{\text{null}}+1)$ "
           r"correction.}")
    tex.append(cap)
    tex.append(r"\label{tab:polan_leader_perm}")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\rowcolors{3}{gray!12}{white}")
    tex.append(r"\begin{tabular}{l c rcc rcc}")
    tex.append(r"\toprule")
    tex.append(r"& & \multicolumn{3}{c}{All $K{=}25$ topics} & "
               r"\multicolumn{3}{c}{Polarizing only "
               rf"(AUC$>$0.9, $n{{=}}{len(POLAR_TOPICS)}$)" + r"}\\")
    tex.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
    tex.append(r"Leader & Party & "
               r"$\mathrm{avg}\,|z|$ & rank & $p_{\text{perm}}$ & "
               r"$\mathrm{avg}\,|z|$ & rank & $p_{\text{perm}}$ \\")
    tex.append(r"\midrule")
    # Use perm_all order (sorted by p_perm asc) and look up matching perm_pol
    perm_all_sorted = perm_all.sort_values("p_perm")
    pol_by_leader = perm_pol.set_index("leader")
    for _, ra in perm_all_sorted.iterrows():
        rp = pol_by_leader.loc[ra["leader"]]
        tex.append(
            f"{latex_escape(ra['leader'].title())} & {ra['party']} & "
            f"${ra['avg_abs_z']:.2f}$ & "
            f"${int(ra['rank'])}/{int(ra['null_n'])+1}$ & "
            f"${ra['p_perm']:.3f}$ & "
            f"${rp['avg_abs_z']:.2f}$ & "
            f"${int(rp['rank'])}/{int(rp['null_n'])+1}$ & "
            f"${rp['p_perm']:.3f}$ \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(os.path.join(OUT, "leader_permutation_table.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print("  -> leader_permutation_table.tex")

    # =================================================
    # Part B: plot leader ideal points
    # =================================================
    print("\nPart B: line plot of leader IPs over topics")
    R = parties == "R"
    D_ = parties == "D"
    party_means = pd.DataFrame({
        "topic": np.arange(K),
        "R_mean": ideal_polan[R].mean(axis=0),
        "D_mean": ideal_polan[D_].mean(axis=0),
    })
    plot_leader_ips(ideal_polan, names, K, party_means,
                     out_pdf=os.path.join(OUT, "leader_ips_plot.pdf"))

    # =================================================
    # Part C: speech excerpts with bold top-words
    # =================================================
    print("\nPart C: speech excerpts with bold wordcloud-direction words")
    for polan_k, lbl in [(5, "Gun Violence"),
                       (16, "Immigration/DHS")]:
        cavi_k = polan_to_cavi[polan_k]
        print(f"  PolAn topic {polan_k} ({lbl}) -> CAVI topic {cavi_k}")
        out_path = os.path.join(OUT, f"topic_{polan_k:02d}_excerpts.tex")
        write_topic_excerpt_tex(polan_k, cavi_k, lbl, beta_polan, eta_polan,
                                  vocab, out_path,
                                  top_n_words=args.top_n_words,
                                  n_speeches_per_dir=2)

    print(f"\nAll outputs in: {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
