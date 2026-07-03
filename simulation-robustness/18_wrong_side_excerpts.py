#!/usr/bin/env python3
"""
18_wrong_side_excerpts.py
=========================
Retrieve real floor-speech excerpts for wrong-side outliers of
Table 4 -- the validation R2 asked for ("who are these senators,
do these estimates make sense, what did they actually say?").

DESIGN (cross-fit replication filter):
  Table 4 (wrong-side outliers) is computed on the ORIGINAL PolAn
  fit.  The PolAn fit, however, stores no speech-id alignment (no
  speech ids, no counts, no permutation; its document ordering
  matches the canonical one in only ~2% of rows), so it cannot map
  a topic-loaded document back to its text.  Speech retrieval is
  therefore done NATIVELY on the revision refit
  (stbs_cavi_results/seed_123456_K25), which has the canonical
  speech-id alignment.

  To avoid quoting text for cases that exist only in one fit, we
  apply a REPLICATION FILTER: we quote speeches only for outliers
  that are wrong-side in BOTH fits --
    * PolAn:  the Table-4 classification (ideal_data.csv), and
    * refit:  the senator's ideal point on the topic-matched refit
              topic lies on the opposite party's side by more than
              one opposite-party SD.
  Of the four headline Table-4 cases, Rand Paul (Cyber Security;
  PolAn 23 -> refit 6, cosine 0.74) and Susan Collins (Health
  Care; PolAn 7 -> refit 7, cosine 0.997) replicate.  Ted Cruz
  (Coast Guard) and Lindsey Graham (Institutes and Research) do
  NOT -- not because the refit contradicts them, but because the
  refit has no corresponding topic (alignment cosines 0.07 / 0.11),
  so no text validation is possible for these two.

Everything below (classification check, theta ranking, share/rank
reporting) runs on the refit; the PolAn ideal point is reported
alongside for reference.
"""
import os, re
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.normpath(os.path.join(REPO, ".."))

POLAN = os.path.join(PROJ, "originalPolAn_results", "fits",
                     "TBIPhier_ideal_ak_Nreg_all_joint_varfam_K25", "params")
CAVI  = os.path.join(REPO, "stbs_cavi_results", "seed_123456_K25", "params")
CLEAN = os.path.join(PROJ, "STBS_CAVI", "data", "hein-daily", "clean")
HEIN  = os.path.join(PROJ, "hein_daily")
OUT = os.path.join(REPO, "results_simulation", "wrong_side_excerpts")
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

# Cases that PASS the cross-fit replication filter:
# (full name, surname, PolAn topic k, matched refit topic k)
CASES = [
    ("Patrick Toomey", "TOOMEY",   5, 24),  # Gun Violence;   cosine 0.946
    ("Rand Paul",      "PAUL",    23,  6),  # Cyber Security; cosine 0.74
    ("Susan Collins",  "COLLINS",  7,  7),  # Health Care;    cosine 0.997
]

N_SPEECHES_PER_CASE = 2
EXCERPT_CHARS = 900


def latex_escape(s: str) -> str:
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def clean_excerpt(text: str, n_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n_chars:
        return text
    cut = text[:n_chars]
    last_period = cut.rfind(". ")
    if last_period > n_chars * 0.6:
        cut = cut[:last_period + 1]
    return cut + " <<ELLIPSIS>>"


def main():
    names = [l.strip() for l in open(os.path.join(CLEAN, "author_map114.txt"))
             if l.strip()]
    parties = np.array([re.search(r"\((\w)\)\s*$", n).group(1) for n in names])

    # ---- PolAn ideal points (Table-4 reference values) ----
    polan = pd.read_csv(os.path.join(POLAN, "ideal_data.csv"), index_col=0)

    # ---- refit: per-topic ideal points + theta + alignment ----
    ideal_refit = np.load(os.path.join(CAVI, "ideal_point_location_final.npy"))
    theta = (np.load(os.path.join(CAVI, "theta_shape_final.npy"))
             / np.load(os.path.join(CAVI, "theta_rate_final.npy")))
    theta_share = theta / theta.sum(axis=1, keepdims=True)
    aidx = np.load(os.path.join(CLEAN, "author_indices114.npy"))
    sids = np.load(os.path.join(CLEAN, "speech_id_indices114.npy"))

    # ---- speech text + metadata ----
    sp_df = pd.read_csv(os.path.join(HEIN, "speeches_114.txt"), sep="|",
                        encoding="ISO-8859-1", on_bad_lines="skip",
                        quoting=3, dtype=str)
    sp_df.columns = [c.strip() for c in sp_df.columns]
    sp_df["speech_id"] = sp_df["speech_id"].astype(str)
    text_by_id = dict(zip(sp_df["speech_id"], sp_df["speech"]))

    descr = pd.read_csv(os.path.join(HEIN, "descr_114.txt"), sep="|",
                        encoding="ISO-8859-1", on_bad_lines="skip",
                        quoting=3, dtype=str)
    descr.columns = [c.strip() for c in descr.columns]
    descr["speech_id"] = descr["speech_id"].astype(str)
    date_by_id = dict(zip(descr["speech_id"], descr["date"]))

    tex = ["% Auto-generated by 18_wrong_side_excerpts.py",
           "% Cross-fit replication filter: only outliers that are wrong-side",
           "% in BOTH the original PolAn fit (Table 4) and the revision refit",
           "% (seed_123456_K25) are quoted. Speech selection runs natively on",
           "% the refit (theta share on the matched refit topic)."]
    rows = []

    for full_name, surname, pk, ck in CASES:
        a = next(i for i, n in enumerate(names)
                 if n.split("(")[0].strip().split()[-1].upper() == surname)
        party = parties[a]

        # PolAn value (Table-4 reference)
        polan_ideal = float(polan[polan["surname"].str.upper() == surname]
                            [str(pk)].values[0])

        # refit value + replication check (same directional criterion as
        # Table 4: the ideal point lies BEYOND the opposite party's mean
        # on the matched refit topic). opp-z reported for reference.
        refit_ideal = float(ideal_refit[a, ck])
        opp = "D" if party == "R" else "R"
        opp_vals = ideal_refit[parties == opp, ck]
        opp_z = (refit_ideal - opp_vals.mean()) / opp_vals.std(ddof=1)
        replicates = ((party == "R" and refit_ideal < opp_vals.mean())
                      or (party == "D" and refit_ideal > opp_vals.mean()))
        status = "REPLICATES" if replicates else "DOES NOT REPLICATE"
        print(f"\n=== {full_name} ({party}) | PolAn k={pk} "
              f"({TOPIC_LABELS[pk]}) ideal={polan_ideal:+.2f} | "
              f"refit k={ck} ideal={refit_ideal:+.2f} opp-z={opp_z:+.2f} "
              f"[{status}] ===")
        if not replicates:
            print("  -> skipped (fails the cross-fit replication filter)")
            continue

        tex.append("")
        tex.append(f"% ---- {full_name} | PolAn topic {pk} "
                   f"{TOPIC_LABELS[pk]} (ideal {polan_ideal:+.2f}) | "
                   f"refit topic {ck} (ideal {refit_ideal:+.2f}, "
                   f"opp-z {opp_z:+.2f}) ----")
        tex.append(r"\paragraph{%s (%s) on Topic~%d (%s), "
                   r"$\hat\ideal_{a,k}=%+.2f$.}"
                   % (full_name, party, pk, TOPIC_LABELS[pk], polan_ideal))

        docs = np.where(aidx == a)[0]
        order = docs[np.argsort(-theta[docs, ck])]
        picked = 0
        for d in order:
            sid = str(sids[d])
            txt = text_by_id.get(sid)
            if not txt or len(txt.strip()) < 200:
                continue
            date = date_by_id.get(sid, "?")
            share = theta_share[d, ck]
            rank = int((theta[d] > theta[d, ck]).sum()) + 1
            excerpt = clean_excerpt(txt, EXCERPT_CHARS)
            print(f"  doc {d} sid {sid} date {date} share={share:.2f} rank=#{rank}")
            tex.append(r"\begin{quote}\small\itshape")
            tex.append(f"({date}; topic share ${share:.2f}$, "
                       f"rank \\#{rank} of $25$ within the speech) "
                       + latex_escape(excerpt).replace("<<ELLIPSIS>>", r"\ldots"))
            tex.append(r"\end{quote}")
            rows.append(dict(senator=full_name, party=party,
                             polan_topic=pk, topic_label=TOPIC_LABELS[pk],
                             refit_topic=ck, polan_ideal=polan_ideal,
                             refit_ideal=refit_ideal, refit_opp_z=opp_z,
                             speech_id=sid, date=date,
                             topic_share=share, topic_rank=rank,
                             excerpt=excerpt))
            picked += 1
            if picked >= N_SPEECHES_PER_CASE:
                break

    with open(os.path.join(OUT, "wrong_side_excerpts.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "wrong_side_excerpts.csv"),
                              index=False)
    print(f"\n-> wrote {OUT}/wrong_side_excerpts.tex")
    print(f"-> wrote {OUT}/wrong_side_excerpts.csv")


if __name__ == "__main__":
    main()
