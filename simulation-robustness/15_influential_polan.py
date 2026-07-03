#!/usr/bin/env python3
"""
15_influential_polan.py
=====================
Re-do the influential-speeches and party-leader analyses using the
ORIGINAL PolAn variational fit, so that the topic numbers in the output match
the topic numbers used in the published paper (Topic 5 = Gun Violence,
Topic 16 = Immigration / DHS, etc.).

For analyses that depend only on per-author quantities (ideal points
indexed by author 0..98, which is the same author map for both fits)
this is straightforward. We compute:

  1. Per-leader ideal-point vector ideal_polan[a_leader, :]  for each
     of the six 114th-Senate party-leadership figures.
  2. Per-topic R-mean vs D-mean of the per-topic ideal points,
     ranking the 25 topics by |R-mean - D-mean|.
  3. Per-topic top-5 most-positive and top-5 most-negative authors
     (i.e. by ideal_polan[a, k]).

For per-document analyses (top influential SPEECH TEXTS per topic via
the LLRT statistic) the PolAn fit's saved per-document parameters are in
the PolAn variational solver's *shuffled training order* and the corresponding
permutation key is not stored, so a direct mapping back to the
original speech_id is unavailable. To stay PolAn-only we therefore
report top AUTHORS rather than top speech texts in the per-topic
analysis. If speech text retrieval is desired, the recommended
fallback is to:
  (a) load the PolAn fit's eta_loc, beta_shp/rte to identify each PolAn topic by
      its top words, then
  (b) match each PolAn topic to the CAVI topic with the highest
      cosine-similarity of beta, and
  (c) retrieve the top-LLRT speeches from the CAVI fit (which has
      canonical author/speech-id alignment) for the matched topic.
That hybrid is left for a separate script (16_influential_hybrid.py
if needed).

Inputs
------
  originalPolAn_results/fits/TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25/
      params/ideal_point_location.npy        (A, K) per-author ideal
  STBS_CAVI/data/hein-daily/clean/
      author_map114.txt                       names + parties

Outputs
-------
  results_simulation/influential_polan/
      party_leaders_ideal_points.csv      one row per leader,
                                              one column per topic
      per_topic_party_gap.csv             one row per topic,
                                              columns R_mean, D_mean,
                                              gap, |gap|, top + bottom
                                              author + ideal values
      topic_KK_authors.txt                per-topic top-N most-
                                              positive and top-N most-
                                              negative authors (text)
"""
import os, re, argparse
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.normpath(os.path.join(REPO, ".."))

POLAN_FIT = os.path.join(PROJ, "originalPolAn_results", "fits",
                        "TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25",
                        "params")
CLEAN = os.path.join(PROJ, "STBS_CAVI", "data", "hein-daily", "clean")
OUT = os.path.join(REPO, "results_simulation", "influential_polan")


PARTY_LEADERS = [
    {"name": "MITCH MCCONNELL", "party": "R", "state": "KY",
     "role": "Majority Leader"},
    {"name": "JOHN CORNYN",     "party": "R", "state": "TX",
     "role": "Majority Whip"},
    {"name": "HARRY REID",      "party": "D", "state": "NV",
     "role": "Minority Leader (until Jan 2017)"},
    {"name": "RICHARD DURBIN",  "party": "D", "state": "IL",
     "role": "Minority Whip"},
    {"name": "CHARLES SCHUMER", "party": "D", "state": "NY",
     "role": "Minority Leader (from Jan 2017)"},
    {"name": "ORRIN HATCH",     "party": "R", "state": "UT",
     "role": "President pro tempore"},
]


def load_polan_ideal():
    """Load PolAn posterior-mean ideal point matrix."""
    ideal = np.load(os.path.join(POLAN_FIT, "ideal_point_location.npy"))
    return ideal


def load_author_map():
    """Read author_map114.txt and parse names + parties."""
    with open(os.path.join(CLEAN, "author_map114.txt")) as f:
        names = [l.strip() for l in f if l.strip()]
    parties = []
    for n in names:
        m = re.search(r"\((\w)\)\s*$", n)
        parties.append(m.group(1) if m else "?")
    return names, np.array(parties)


def find_author_index(target_name, names):
    """Match a target name (e.g. 'MITCH MCCONNELL') to its index in names."""
    target = target_name.strip().upper()
    for i, nm in enumerate(names):
        nm_norm = nm.split("(")[0].strip().upper()
        if nm_norm == target:
            return i
    return None


# ============================================================== #
def write_per_topic_authors(ideal, names, parties, K, n_per_dir):
    """For each topic, write a text file with the top-N most-positive
    and most-negative authors by ideal[a, k] (PolAn fit).
    """
    rows_all = []
    for k in range(K):
        col = ideal[:, k]
        # top-N positive (right) and bottom-N (left)
        order_right = np.argsort(-col)[:n_per_dir]
        order_left = np.argsort(col)[:n_per_dir]
        out_path = os.path.join(OUT, f"topic_{k:02d}_authors.txt")
        with open(out_path, "w") as fh:
            fh.write("=" * 72 + "\n")
            fh.write(f"Topic {k:2d} -- top {n_per_dir} authors by PolAn ideal\n")
            fh.write(f"Source: originalPolAn_results/fits/.../params/"
                     f"ideal_point_location.npy\n")
            fh.write("=" * 72 + "\n\n")
            fh.write("# RIGHT (most-positive ideal_{a,k})\n")
            fh.write("-" * 72 + "\n")
            for rank, a in enumerate(order_right, 1):
                fh.write(f"  rank {rank}: {names[a]:35s}  "
                         f"ideal={col[a]:+.3f}\n")
                rows_all.append(dict(topic=k, direction="right",
                                     rank=rank, author=names[a],
                                     party=parties[a], ideal=float(col[a])))
            fh.write("\n# LEFT (most-negative ideal_{a,k})\n")
            fh.write("-" * 72 + "\n")
            for rank, a in enumerate(order_left, 1):
                fh.write(f"  rank {rank}: {names[a]:35s}  "
                         f"ideal={col[a]:+.3f}\n")
                rows_all.append(dict(topic=k, direction="left",
                                     rank=rank, author=names[a],
                                     party=parties[a], ideal=float(col[a])))
        print(f"  wrote {out_path}")
    return rows_all


def write_per_leader_report(leader, ideal, names, K):
    """For one leader: dump the 25-vector of ideal points + the topic
    where they are most extreme."""
    a = find_author_index(leader["name"], names)
    if a is None:
        print(f"  WARNING: leader {leader['name']} not in author_map")
        return None
    ip_vec = ideal[a]
    k_extreme = int(np.abs(ip_vec).argmax())

    out_path = os.path.join(OUT,
                              f"leader_{leader['name'].replace(' ', '_')}.txt")
    with open(out_path, "w") as fh:
        fh.write("=" * 72 + "\n")
        fh.write(f"Party leader: {leader['name']}  "
                 f"({leader['party']}, {leader['state']})\n")
        fh.write(f"Role: {leader['role']}\n")
        fh.write(f"Author index in author_map114.txt: {a}\n")
        fh.write(f"Author name (canonical):           {names[a]}\n")
        fh.write(f"Most-extreme topic: k={k_extreme}  ideal={ip_vec[k_extreme]:+.3f}\n")
        fh.write("=" * 72 + "\n\n")
        fh.write(f"Per-topic ideal points  (PolAn fit, "
                 f"originalPolAn_results/fits/.../ideal_point_location.npy)\n")
        for k in range(K):
            marker = "  <-- MOST EXTREME" if k == k_extreme else ""
            fh.write(f"  topic {k:2d}: {ip_vec[k]:+.3f}{marker}\n")
    print(f"  wrote {out_path}")
    return dict(leader=leader["name"], party=leader["party"],
                state=leader["state"], role=leader["role"],
                a_idx=a, k_extreme=k_extreme,
                ideal_at_extreme=float(ip_vec[k_extreme]),
                avg_ideal=float(ip_vec.mean()),
                **{f"topic_{k}": float(ip_vec[k]) for k in range(K)})


# ============================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-dir", type=int, default=5,
                    help="number of authors per direction (left/right) per topic")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("Loading PolAn fit + author map ...")
    ideal = load_polan_ideal()
    names, parties = load_author_map()
    A, K = ideal.shape
    print(f"  PolAn ideal: {ideal.shape}")
    print(f"  authors  : {len(names)}  (R={int((parties=='R').sum())}, "
          f"D={int((parties=='D').sum())}, I={int((parties=='I').sum())})")

    # ------------- Per-topic R/D gap ----------------
    R = parties == "R"
    D_ = parties == "D"
    rows = []
    for k in range(K):
        r_mean = float(ideal[R, k].mean())
        d_mean = float(ideal[D_, k].mean())
        col = ideal[:, k]
        rows.append(dict(
            k=k,
            R_mean=r_mean,
            D_mean=d_mean,
            gap=r_mean - d_mean,
            abs_gap=abs(r_mean - d_mean),
            R_std=float(ideal[R, k].std()),
            D_std=float(ideal[D_, k].std()),
            top_author=names[col.argmax()],
            top_ideal=float(col.max()),
            bottom_author=names[col.argmin()],
            bottom_ideal=float(col.min()),
        ))
    df_gap = pd.DataFrame(rows).sort_values("abs_gap", ascending=False)
    out_csv = os.path.join(OUT, "per_topic_party_gap.csv")
    df_gap.to_csv(out_csv, index=False)
    print(f"\nTop-10 topics by |R-mean - D-mean| (PolAn):")
    print(df_gap[["k", "R_mean", "D_mean", "abs_gap",
                  "top_author", "top_ideal",
                  "bottom_author", "bottom_ideal"]].head(10).round(3)
            .to_string(index=False))
    print(f"  -> wrote {out_csv}")

    # ------------- Per-topic top authors ----------------
    print(f"\nWriting per-topic top-{args.n_per_dir} authors ...")
    rows_authors = write_per_topic_authors(ideal, names, parties, K,
                                            args.n_per_dir)
    pd.DataFrame(rows_authors).to_csv(
        os.path.join(OUT, "per_topic_top_authors.csv"), index=False)

    # ------------- Per-leader IP vector ----------------
    print(f"\nWriting per-leader ideal vectors ...")
    rows_leaders = []
    for leader in PARTY_LEADERS:
        rec = write_per_leader_report(leader, ideal, names, K)
        if rec is not None:
            rows_leaders.append(rec)
    pd.DataFrame(rows_leaders).to_csv(
        os.path.join(OUT, "party_leaders_ideal_points.csv"),
        index=False)
    print(f"  -> wrote party_leaders_ideal_points.csv ({len(rows_leaders)} leaders)")

    print(f"\nAll PolAn outputs in: {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
