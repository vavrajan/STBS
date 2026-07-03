#!/usr/bin/env python3
"""
14_influential_speeches.py
==========================
Identify the most "left-leaning" and most "right-leaning" influential
speeches for each topic of the STBS fit, and report per-topic ideal
points + most representative speeches for the 114th Senate party
leadership. The script is a numpy-only adaptation of
STBS_CAVI/code/influential_speeches.py and the
"theta_then_loglik_ratio_test" strategy described in the paper
\S{}S.3, but it reads the saved variational posterior summaries from
the headline original-PolAn fit rather than re-instantiating the
TensorFlow model.

Methodology -- "left" / "right" per topic
-----------------------------------------
For each topic $k$:
  1. Compute the per-document log-likelihood-ratio contribution
        chi_{d,k} = sum_v [ y_{d,v} * log(rate_true / rate_null)
                            - (rate_true - rate_null) ]
     where
        rate_null_{d,k,v} = theta_{d,k} * beta_{k,v}                    (ideal = 0)
        rate_true_{d,k,v} = theta_{d,k} * beta_{k,v}
                            * exp(eta_{k,v} * ideal_{a_d, k})
     i.e. how much topic-k counts gain when we allow the speaker's
     ideal point to be non-zero.
  2. Split by the sign of the speaker's per-topic ideal:
        - "Right-leaning influential" = top-N by chi_{d,k} among
          documents whose a_d has ideal_{a_d, k} > 0
        - "Left-leaning influential"  = top-N by chi_{d,k} among
          documents with ideal_{a_d, k} < 0
  3. Write out one .txt file per topic and direction, containing
     the senator, party, state, date, and raw speech text.

Party leaders (114th Senate, 2015-Jan 2017)
-------------------------------------------
We hard-code a small leadership panel:
  - Mitch McConnell  (R, KY)  Majority Leader
  - John Cornyn      (R, TX)  Majority Whip
  - Harry Reid       (D, NV)  Minority Leader
  - Dick Durbin      (D, IL)  Minority Whip
  - Charles Schumer  (D, NY)  Minority Leader (Jan 2017-)
  - Orrin Hatch      (R, UT)  President pro tempore
For each leader the script reports:
  * the K-dimensional vector of per-topic ideal points;
  * the top-5 most representative speeches (highest theta_{d,k} *
    |ideal_{leader, k}| across all topics).

Output
------
results_simulation/influential_speeches/
    topic_KK_left.txt
    topic_KK_right.txt
    party_leaders_ideal_points.csv
    party_leader_NAME_speeches.txt
    summary.csv            (one row per (topic, direction, speech))
"""
import os, sys, argparse, re
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.normpath(os.path.join(REPO, ".."))

# Note on the fit source:
# The original PolAn fit (originalPolAn_results/fits/TBIPhier_..._K25/)
# saves its parameters in *shuffled training order* (its
# all_author_indices.npy does NOT match the canonical
# author_indices114.npy that is aligned with speech_id_indices114.npy),
# and no permutation key is stored, so we cannot map PolAn doc-index
# back to original speech_id without re-running the shuffle.
# We therefore use the CAVI fit at seed 123456 / K=25 instead, whose
# theta and ideal posterior means are computed against the canonical
# author / speech-id mapping. Substantively the two fits agree at
# Pearson r > 0.83 on aggregate ideal points (see Table 1 of
# simulation_reply.tex) and r > 0.86 on theta after Hungarian topic
# alignment, so the choice of fit only affects very fine-grained
# per-speech rankings.
CAVI_FIT = os.path.join(REPO, "stbs_cavi_results", "seed_123456_K25",
                         "params")
CLEAN = os.path.join(PROJ, "STBS_CAVI", "data", "hein-daily", "clean")
HEIN_ORIG = os.path.join(PROJ, "hein_daily")
SPEAKER_MAP = os.path.join(
    "/Users/paul.hofmarcher/Documents/svn/baR/Projects/Congress_Speeches",
    "hein-daily", "114_SpeakerMap.txt")
OUT = os.path.join(REPO, "results_simulation", "influential_speeches")

PARTY_LEADERS = [
    {"name": "MITCH MCCONNELL",     "party": "R", "state": "KY",
     "role": "Majority Leader"},
    {"name": "JOHN CORNYN",         "party": "R", "state": "TX",
     "role": "Majority Whip"},
    {"name": "HARRY REID",          "party": "D", "state": "NV",
     "role": "Minority Leader (until Jan 2017)"},
    {"name": "RICHARD DURBIN",      "party": "D", "state": "IL",
     "role": "Minority Whip"},
    {"name": "CHARLES SCHUMER",     "party": "D", "state": "NY",
     "role": "Minority Leader (from Jan 2017)"},
    {"name": "ORRIN HATCH",         "party": "R", "state": "UT",
     "role": "President pro tempore"},
]


# ============================================================== #
def load_cavi_fit():
    """Load posterior-mean parameters of the CAVI fit, aligned with the
    canonical author_indices114.npy / speech_id_indices114.npy mapping.
    """
    theta_shp = np.load(os.path.join(CAVI_FIT, "theta_shape_final.npy"))
    theta_rte = np.load(os.path.join(CAVI_FIT, "theta_rate_final.npy"))
    theta = theta_shp / theta_rte                                # (D, K)
    beta_shp = np.load(os.path.join(CAVI_FIT, "beta_shape_final.npy"))
    beta_rte = np.load(os.path.join(CAVI_FIT, "beta_rate_final.npy"))
    beta = beta_shp / beta_rte                                   # (K, V)
    eta = np.load(os.path.join(CAVI_FIT, "eta_location_final.npy"))  # (K, V)
    ideal = np.load(os.path.join(CAVI_FIT,
                                  "ideal_point_location_final.npy"))  # (A, K)
    # author_indices: canonical mapping (doc -> author)
    author_indices = np.load(os.path.join(CLEAN, "author_indices114.npy"))
    return theta, beta, eta, ideal, author_indices.astype(np.int32)


def load_metadata():
    """Load the raw Hein-Daily speeches, descriptions, speaker map,
    plus the speech_id <-> doc_index mapping used during training.
    """
    speeches = pd.read_csv(os.path.join(HEIN_ORIG, "speeches_114.txt"),
                            encoding="ISO-8859-1", sep="|",
                            quoting=3, on_bad_lines="warn")
    descr = pd.read_csv(os.path.join(HEIN_ORIG, "descr_114.txt"),
                         encoding="ISO-8859-1", sep="|",
                         on_bad_lines="warn")
    speaker_map = pd.read_csv(SPEAKER_MAP, encoding="ISO-8859-1",
                                sep="|", on_bad_lines="warn")
    merged = speeches.merge(descr, on="speech_id", how="left")
    merged = merged.merge(speaker_map, on="speech_id", how="left",
                            suffixes=("", "_sm"))
    speech_id_indices = np.load(os.path.join(CLEAN,
                                              "speech_id_indices114.npy"))
    # author_map: human-readable name per author index
    with open(os.path.join(CLEAN, "author_map114.txt")) as fh:
        author_names = [line.rstrip("\n") for line in fh if line.strip()]
    return merged, speech_id_indices, author_names


def compute_loglik_contribution(theta, beta, eta, ideal_doc, counts):
    """Per-(d, k) log-likelihood gain from non-zero ideal point.
       chi_{d,k} = sum_v [ y_{dv} * log(lambda_true/lambda_null)
                           - (lambda_true - lambda_null) ]_k
       computed topic-by-topic (vectorised by k loop to keep memory
       manageable on D=14672 x V=5031).
    """
    D, K = theta.shape
    _, V = beta.shape
    chi = np.zeros((D, K), dtype=np.float64)
    for k in range(K):
        rate_null = theta[:, k:k+1] * beta[k:k+1, :]             # (D, V)
        ideo = np.exp(eta[k:k+1, :] * ideal_doc[:, k:k+1])       # (D, V)
        rate_true = rate_null * ideo
        diff = rate_true - rate_null                              # (D, V)
        # log ratio, avoid divide-by-zero
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(rate_true) - np.log(rate_null)
        lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
        # sum over v: count-weighted log ratio - rate diff
        cy = counts.toarray() if hasattr(counts, "toarray") else counts
        chi[:, k] = (cy * lr).sum(axis=1) - diff.sum(axis=1)
    return chi


# ============================================================== #
def write_topic_extreme_speeches(direction, k, top_docs, df, ids,
                                  author_names, ideal_doc, theta,
                                  n_to_write):
    """Write the top-N most extreme speeches for one (topic, direction)
    combination to a text file. direction in {'left', 'right'}.
    Returns a list of dicts for the summary CSV.
    """
    out_path = os.path.join(OUT, f"topic_{k:02d}_{direction}.txt")
    rows = []
    with open(out_path, "w") as fh:
        fh.write("=" * 70 + "\n")
        fh.write(f"Topic {k:2d} -- {n_to_write} most {direction}-leaning "
                 f"influential speeches\n")
        fh.write(f"Influence: log-likelihood gain from non-zero ideal "
                 f"point, restricted to speakers with ideal {('<' if direction=='left' else '>')} 0.\n")
        fh.write("=" * 70 + "\n\n")
        for rank, d_idx in enumerate(top_docs[:n_to_write], 1):
            sid = int(ids[d_idx])
            row = df.loc[df["speech_id"] == sid]
            if row.empty:
                fh.write(f"Speech rank {rank}: speech_id {sid} not found in metadata.\n\n")
                continue
            row = row.iloc[0]
            speaker = row.get("speaker", "?")
            party = row.get("party", row.get("party_sm", "?"))
            state = row.get("state", row.get("state_sm", "?"))
            date = row.get("date", "?")
            speech = str(row.get("speech", ""))
            a_idx = int(np.load(os.path.join(CLEAN,
                                              "author_indices114.npy"))[d_idx])
            speaker_known = author_names[a_idx] if 0 <= a_idx < len(author_names) else "?"
            fh.write(f"# Speech rank {rank}\n")
            fh.write(f"  Speaker (descr) : {speaker}\n")
            fh.write(f"  Author (model)  : {speaker_known}\n")
            fh.write(f"  Party / State   : {party} / {state}\n")
            fh.write(f"  Date            : {date}\n")
            fh.write(f"  Ideal_{{a,k}}   : {ideal_doc[d_idx, k]:+.3f}\n")
            fh.write(f"  Theta_{{d,k}}   : {theta[d_idx, k]:.4f}\n")
            fh.write(f"  Speech         : {speech.strip()[:2500]}\n")
            if len(speech) > 2500:
                fh.write(f"  ... [truncated; full length {len(speech)} chars]\n")
            fh.write("\n" + "-" * 70 + "\n\n")
            rows.append(dict(topic=k, direction=direction, rank=rank,
                             speech_id=sid, speaker=speaker_known,
                             party=party, state=state, date=date,
                             ideal_ak=float(ideal_doc[d_idx, k]),
                             theta_dk=float(theta[d_idx, k])))
    print(f"  -> wrote {out_path}")
    return rows


# ============================================================== #
def write_leader_report(leader, ideal, theta, df, ids, author_names,
                         n_speeches):
    """For one party leader: report all 25 per-topic ideal points and
    the top-N representative speeches across the leader's most-polarised
    topics."""
    # Find leader's index in author_map
    target = leader["name"].strip().upper()
    a_idx = None
    for i, nm in enumerate(author_names):
        nm_norm = nm.split("(")[0].strip().upper()
        if nm_norm == target:
            a_idx = i
            break
    if a_idx is None:
        print(f"  WARNING: leader {leader['name']} not in author_map")
        return None

    ip_vec = ideal[a_idx]                                   # (K,)
    # Documents authored by this leader
    author_indices_all = np.load(os.path.join(CLEAN,
                                                "author_indices114.npy"))
    mask = author_indices_all == a_idx
    leader_docs = np.where(mask)[0]
    if leader_docs.size == 0:
        print(f"  WARNING: {leader['name']} has no documents in corpus")
        return None

    # Polarisation = |ideal_k| weighted by per-doc theta_{d,k}; for
    # each doc, pick its top topic and use |ideal_{a, k}| * theta_{d, k}
    # as a representativeness score.
    theta_leader = theta[leader_docs]                       # (n_docs, K)
    score = theta_leader * np.abs(ip_vec)[None, :]          # (n_docs, K)
    best_k_per_doc = score.argmax(axis=1)                   # (n_docs,)
    best_score = score.max(axis=1)                          # (n_docs,)
    rank_order = np.argsort(-best_score)                    # top first

    leader_name_clean = leader["name"].replace(" ", "_")
    out_path = os.path.join(OUT, f"party_leader_{leader_name_clean}_speeches.txt")
    with open(out_path, "w") as fh:
        fh.write("=" * 70 + "\n")
        fh.write(f"Party leader: {leader['name']}  ({leader['party']}, {leader['state']})\n")
        fh.write(f"Role: {leader['role']}\n")
        fh.write(f"Author index in model: {a_idx}\n")
        fh.write(f"Number of speeches in corpus: {leader_docs.size}\n")
        fh.write("=" * 70 + "\n\n")
        fh.write("Per-topic ideal points (ideal_{a,k}):\n")
        for k in range(len(ip_vec)):
            marker = " <-- MOST extreme" if abs(ip_vec[k]) == max(abs(ip_vec)) else ""
            fh.write(f"  Topic {k:2d} : {ip_vec[k]:+.3f}{marker}\n")
        fh.write("\n" + "=" * 70 + "\n")
        fh.write(f"Top {n_speeches} most representative speeches "
                 f"(by theta_{{d,k}} * |ideal_{{a,k}}|):\n")
        fh.write("=" * 70 + "\n\n")
        for rank, doc_pos in enumerate(rank_order[:n_speeches], 1):
            d_idx = int(leader_docs[doc_pos])
            k = int(best_k_per_doc[doc_pos])
            sid = int(ids[d_idx])
            row = df.loc[df["speech_id"] == sid]
            if row.empty:
                fh.write(f"# Speech rank {rank}: speech_id {sid} not in metadata.\n\n")
                continue
            row = row.iloc[0]
            speech = str(row.get("speech", ""))
            fh.write(f"# Rank {rank}: speech_id {sid}\n")
            fh.write(f"  Best topic         : {k}\n")
            fh.write(f"  Ideal_{{a,k}}       : {ip_vec[k]:+.3f}\n")
            fh.write(f"  Theta_{{d,k}}       : {theta[d_idx, k]:.4f}\n")
            fh.write(f"  Date               : {row.get('date', '?')}\n")
            fh.write(f"  Score (theta*|i|)  : {best_score[doc_pos]:.4f}\n")
            fh.write(f"  Speech             : {speech.strip()[:2500]}\n")
            if len(speech) > 2500:
                fh.write(f"  ... [truncated; full length {len(speech)} chars]\n")
            fh.write("\n" + "-" * 70 + "\n\n")
    print(f"  -> wrote {out_path}")
    return ip_vec


# ============================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-direction", type=int, default=5,
                    help="how many left/right speeches to write per topic")
    ap.add_argument("--n-per-leader", type=int, default=8,
                    help="how many representative speeches per party leader")
    ap.add_argument("--theta-batch", type=int, default=512,
                    help="for each topic, restrict to the top-N "
                         "documents by theta_{d,k} before computing "
                         "log-likelihood ratio (saves time + matches "
                         "the reference 'theta_then_loglik_ratio_test'"
                         "strategy from Vavra et al, STBS_CAVI/code/"
                         "influential_speeches.py)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("Loading PolAn posterior means ...")
    theta, beta, eta, ideal, author_indices = load_cavi_fit()
    D, K = theta.shape
    A = ideal.shape[0]
    V = beta.shape[1]
    print(f"  theta: ({D}, {K}),  beta: ({K}, {V}),  "
          f"eta: ({K}, {V}),  ideal: ({A}, {K})")

    print("Loading speeches + descriptions ...")
    df, speech_id_indices, author_names = load_metadata()
    print(f"  speeches: {len(df)},  speech_id_indices: {len(speech_id_indices)},  "
          f"author_map: {len(author_names)}")

    # Per-doc ideal (broadcast author -> doc)
    ideal_doc = ideal[author_indices]                          # (D, K)

    # Need the counts matrix for the LLRT
    from scipy import sparse
    counts = sparse.load_npz(os.path.join(CLEAN, "counts114.npz"))
    print(f"  counts: {counts.shape}  nnz={counts.nnz:,}")

    # =========================================================
    # PART 1 -- per topic, most left / most right influential speeches
    # =========================================================
    print(f"\nComputing per-topic log-likelihood-ratio influence ...")
    summary_rows = []
    for k in range(K):
        # Pre-restrict to high-theta documents (theta_then_loglik step),
        # taking the union of left- and right-leaning candidates so
        # both directions have enough documents to choose from.
        theta_k = theta[:, k]
        topB = min(args.theta_batch, D)
        top_theta_idx = np.argpartition(-theta_k, topB - 1)[:topB]

        # Restrict computation to these top-B docs
        sub_theta = theta[top_theta_idx]
        sub_ideal_doc = ideal_doc[top_theta_idx]
        sub_counts = counts[top_theta_idx]

        # Compute per-doc LLRT contribution for topic k
        rate_null = sub_theta[:, k:k+1] * beta[k:k+1, :]       # (B, V)
        ideo = np.exp(eta[k:k+1, :] * sub_ideal_doc[:, k:k+1]) # (B, V)
        rate_true = rate_null * ideo
        diff = rate_true - rate_null
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(rate_true) - np.log(rate_null)
        lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
        cy = sub_counts.toarray()
        chi_k = (cy * lr).sum(axis=1) - diff.sum(axis=1)         # (B,)

        # Split by sign of ideal_{a_d, k}
        sub_i = sub_ideal_doc[:, k]
        left_pool = np.where(sub_i < 0)[0]
        right_pool = np.where(sub_i > 0)[0]
        # Among each pool, take the top by chi (most extreme LLRT)
        if left_pool.size > 0:
            order = left_pool[np.argsort(-chi_k[left_pool])]
            top_docs_left = top_theta_idx[order]
            summary_rows.extend(write_topic_extreme_speeches(
                "left", k, top_docs_left, df, speech_id_indices,
                author_names, ideal_doc, theta, args.n_per_direction))
        if right_pool.size > 0:
            order = right_pool[np.argsort(-chi_k[right_pool])]
            top_docs_right = top_theta_idx[order]
            summary_rows.extend(write_topic_extreme_speeches(
                "right", k, top_docs_right, df, speech_id_indices,
                author_names, ideal_doc, theta, args.n_per_direction))

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT, "summary.csv"), index=False)
    print(f"  -> wrote summary.csv ({len(summary_rows)} rows)")

    # =========================================================
    # PART 2 -- party leaders
    # =========================================================
    print(f"\nProcessing party leaders ...")
    ip_table = []
    for leader in PARTY_LEADERS:
        ip_vec = write_leader_report(leader, ideal, theta, df,
                                       speech_id_indices, author_names,
                                       args.n_per_leader)
        if ip_vec is not None:
            row = dict(leader=leader["name"], party=leader["party"],
                       state=leader["state"], role=leader["role"])
            row.update({f"topic_{k}": float(ip_vec[k]) for k in range(K)})
            ip_table.append(row)
    pd.DataFrame(ip_table).to_csv(
        os.path.join(OUT, "party_leaders_ideal_points.csv"),
        index=False)
    print(f"  -> wrote party_leaders_ideal_points.csv "
          f"({len(ip_table)} leaders)")

    print(f"\nAll outputs in: {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
