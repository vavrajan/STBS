#!/usr/bin/env python3
"""
23_twitter.py  --  STBS on 114th-Congress Senator TWEETS (second application, R1 #9)
====================================================================================
Dataset: Kaggle "US Congressional Tweets" (oscaryezfeijo), unzipped into  tweets/
  - tweets/tweets.json : NDJSON, one tweet per line
        fields used: screen_name, created_at (UNIX seconds), text
  - tweets/users.json  : NDJSON, one Twitter profile per line
        fields used: screen_name, name, description, location

Pipeline
--------
  python3 23_twitter.py --inspect      # sanity-check schema + senator coverage
  python3 23_twitter.py --build-map    # write a CANDIDATE handle->senator map to verify
  # >>> open data_twitter/senate114_handles.csv, fix/extend it, save <<<
  python3 23_twitter.py --prep         # stream tweets.json -> bigram DTM + X
  ./venv_gpu/bin/python3 23_twitter.py --fit        # STBS fit (Metal GPU)
  python3 23_twitter.py --correlate    # corr with main speech IP + DW-NOMINATE

Maps senators by matching the Twitter profile "name" to our 99-senator metadata
(surname), restricted to profiles whose description mentions the Senate.  Because
surname alone over-matches (House/Governors share surnames), the map is written to
CSV for manual verification before --prep.
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = os.path.dirname(os.path.abspath(__file__))
TW_DIR   = os.path.join(REPO, "tweets")
TWEETS   = os.path.join(TW_DIR, "tweets.json")
USERS    = os.path.join(TW_DIR, "users.json")
WORK_DIR = os.path.join(REPO, "data_twitter", "congress114_senate", "clean")
RES_DIR  = os.path.join(REPO, "stbs_cavi_results_twitter114")
OUT_DIR  = os.path.join(REPO, "results_simulation", "twitter114")
HANDLE_MAP = os.path.join(REPO, "data_twitter", "senate114_handles.csv")

META_99   = os.path.join(REPO, "..", "STBS_CAVI", "data", "hein-daily",
                         "clean", "author_detailed_info114.csv")
DW_MAIN   = os.path.join(REPO, "stbs_cavi_results", "comparison_dw_nominate.csv")
STOPWORDS = os.path.join(REPO, "..", "STBS_CAVI", "data", "hein-daily",
                         "orig", "stopwords.txt")

# 114th Congress window, as UNIX seconds (UTC)
T0 = int(pd.Timestamp("2015-01-03", tz="UTC").timestamp())
T1 = int(pd.Timestamp("2017-01-03", tz="UTC").timestamp())

NGRAM = (2, 2)
# tweets are short -> use an ABSOLUTE min_df (a bigram must occur in >= MIN_DF tweets).
# Smaller corpus for a tractable fit: cap tweets/senator and the vocabulary size.
MIN_DF, MAX_DF_FRAC, MIN_AUTHORS = 20, 0.30, 5
MAX_FEATURES = 2500                 # keep the MAX_FEATURES most frequent bigrams
MAX_TWEETS_PER_SENATOR = 5000       # random subsample per senator
SAMPLE_SEED = 314159
COVARIATES = ["party", "gender", "region", "generation", "exper_chamber"]


# ----------------------------- helpers ----------------------------- #
def load_senators():
    s = pd.read_csv(META_99)
    s["surname"] = s["surname_x"].astype(str).str.upper().str.strip()
    return s

def iter_ndjson(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def users_table():
    rows = [(r.get("screen_name", ""), r.get("name", ""),
             r.get("description", "") or "", r.get("location", "") or "")
            for r in iter_ndjson(USERS)]
    u = pd.DataFrame(rows, columns=["handle", "name", "desc", "loc"])
    u["surname"] = u["name"].astype(str).str.split().str[-1].str.upper().str.strip()
    u["is_sen"] = u["desc"].str.contains("senat", case=False, na=False)
    return u


# ------------------------------ inspect ---------------------------- #
def cmd_inspect():
    print("=== tweets.json sample (first tweet) ===")
    for r in iter_ndjson(TWEETS):
        print({k: r.get(k) for k in ["screen_name", "created_at", "lang", "text"]})
        print("  created_at ->", pd.to_datetime(r["created_at"], unit="s"))
        break
    sen = load_senators(); sur = set(sen["surname"])
    u = users_table()
    cand = u[u["surname"].isin(sur) & u["is_sen"]]
    print(f"\nusers.json accounts: {len(u)};  senator-candidate accounts "
          f"(surname match + 'senat' in bio): {len(cand)} "
          f"covering {cand['surname'].nunique()} of {len(sur)} senators")
    miss = sorted(sur - set(cand["surname"]))
    print(f"senators with no candidate account ({len(miss)}): {miss}")
    print("\n-> run --build-map, then verify data_twitter/senate114_handles.csv")


# --------------------------- build map ----------------------------- #
def cmd_build_map():
    os.makedirs(os.path.dirname(HANDLE_MAP), exist_ok=True)
    sen = load_senators(); sur = set(sen["surname"])
    u = users_table()
    cand = u[u["surname"].isin(sur)].copy()
    cand["keep"] = cand["is_sen"].astype(int)   # default: keep accounts whose bio says Senate
    cand = cand.sort_values(["surname", "is_sen"], ascending=[True, False])
    cand[["handle", "name", "surname", "desc", "loc", "is_sen", "keep"]].to_csv(
        HANDLE_MAP, index=False)
    print(f"-> wrote {len(cand)} candidate accounts to {HANDLE_MAP}")
    print("   VERIFY: set keep=1 for the correct senator account(s), keep=0 for "
          "House/Governor namesakes; add rows for missing senators by hand.")


def load_handle_map():
    if not os.path.exists(HANDLE_MAP):
        sys.exit(f"{HANDLE_MAP} not found. Run --build-map and verify it first.")
    hm = pd.read_csv(HANDLE_MAP)
    hm = hm[hm.get("keep", 1) == 1]
    hm["handle"] = hm["handle"].astype(str).str.strip()
    hm["surname"] = hm["surname"].astype(str).str.upper().str.strip()
    return dict(zip(hm["handle"], hm["surname"]))   # screen_name -> senator surname


# ------------------------------- prep ------------------------------ #
def cmd_prep():
    from sklearn.feature_extraction.text import CountVectorizer
    os.makedirs(WORK_DIR, exist_ok=True)
    h2s = load_handle_map()
    print(f"handle->senator map: {len(h2s)} accounts, "
          f"{len(set(h2s.values()))} senators")

    # stream the 1.6 GB tweets.json; keep senator tweets in the 114th window
    surnames, texts = [], []
    n = kept = 0
    for r in iter_ndjson(TWEETS):
        n += 1
        if n % 1_000_000 == 0:
            print(f"  scanned {n:,} tweets, kept {kept:,}")
        sn = r.get("screen_name")
        if sn not in h2s:
            continue
        t = r.get("created_at")
        if t is None or not (T0 <= int(t) < T1):
            continue
        txt = r.get("text") or ""
        if not txt:
            continue
        surnames.append(h2s[sn]); texts.append(txt); kept += 1
    print(f"scanned {n:,} tweets total; kept {kept:,} senator tweets in 114th window")
    if kept == 0:
        sys.exit("No tweets kept -- check the handle map and the date window.")

    # random subsample up to MAX_TWEETS_PER_SENATOR tweets per senator
    surnames = np.array(surnames)
    rng = np.random.default_rng(SAMPLE_SEED)
    sel = []
    for a in np.unique(surnames):
        idx = np.where(surnames == a)[0]
        if len(idx) > MAX_TWEETS_PER_SENATOR:
            idx = rng.choice(idx, MAX_TWEETS_PER_SENATOR, replace=False)
        sel.append(idx)
    sel = np.sort(np.concatenate(sel))
    surnames = surnames[sel]
    texts = [texts[i] for i in sel]
    print(f"after capping at {MAX_TWEETS_PER_SENATOR}/senator: {len(texts):,} tweets")
    authors = sorted(set(surnames))
    a_idx = {a: i for i, a in enumerate(authors)}
    author_indices = np.array([a_idx[s] for s in surnames], dtype=np.int32)
    print(f"senators with tweets: {len(authors)}; per-senator tweet counts: "
          f"min {np.bincount(author_indices).min()}, "
          f"median {int(np.median(np.bincount(author_indices)))}, "
          f"max {np.bincount(author_indices).max()}")

    sw = open(STOPWORDS).read().split()
    cv = CountVectorizer(ngram_range=NGRAM, token_pattern="[a-zA-Z]+",
                         stop_words=sw, lowercase=True,
                         min_df=MIN_DF, max_df=MAX_DF_FRAC,
                         max_features=MAX_FEATURES)
    counts = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    used_by = np.zeros(counts.shape[1], int)
    for a in range(len(authors)):
        used_by += (np.asarray((counts[author_indices == a] > 0).sum(0)).ravel() > 0)
    keep = used_by >= MIN_AUTHORS
    counts, vocab = counts[:, keep], vocab[keep]
    nz = np.asarray((counts > 0).sum(1)).ravel() > 0
    counts, author_indices = counts[nz], author_indices[nz]
    # re-index authors that survived
    surv = sorted(set(author_indices.tolist()))
    remap = {o: i for i, o in enumerate(surv)}
    authors = [authors[o] for o in surv]
    author_indices = np.array([remap[a] for a in author_indices], dtype=np.int32)
    print(f"DTM: {counts.shape[0]:,} tweets x {counts.shape[1]:,} bigrams, "
          f"nnz={counts.nnz:,}; senators kept={len(authors)}")

    # covariate matrix X in author order
    sen = load_senators().set_index("surname")
    sen_a = sen.loc[authors].reset_index()
    cols, labels = [np.ones(len(authors))], ["intercept"]
    for cov in COVARIATES:
        d = pd.get_dummies(sen_a[cov].astype("category"), prefix=cov,
                           drop_first=True).astype(float)
        for c in d.columns:
            cols.append(d[c].to_numpy()); labels.append(c)
    X = np.column_stack(cols).astype(np.float32)

    # STBS expects float counts (int64 triggers a dtype mismatch in the CAVI step)
    sp.save_npz(os.path.join(WORK_DIR, "counts.npz"),
                counts.astype(np.float32).tocsr())
    np.save(os.path.join(WORK_DIR, "author_indices.npy"), author_indices)
    open(os.path.join(WORK_DIR, "vocabulary.txt"), "w").write("\n".join(vocab) + "\n")
    open(os.path.join(WORK_DIR, "author_map.txt"), "w").write("\n".join(authors) + "\n")
    np.save(os.path.join(WORK_DIR, "X.npy"), X)
    json.dump({"labels": labels, "shape": list(X.shape)},
              open(os.path.join(WORK_DIR, "X_labels.json"), "w"), indent=2)
    print(f"X: {X.shape}  covariates={labels}")
    print(f"-> wrote prepared data to {WORK_DIR}")


# -------------------------------- fit ------------------------------ #
def cmd_fit(num_epochs=120, seed=314159, num_topics=25):
    est = os.path.join(REPO, "estimate_STBS_twitter.py")
    out = os.path.join(RES_DIR, f"seed_{seed}_K{num_topics}")
    os.makedirs(RES_DIR, exist_ok=True)
    # party-aligned ideal-point initialisation (+1 R, -1 D, 0 I), to break the
    # global-sign symmetry as in the main fit (sign re-checked post-hoc anyway)
    authors = [l.strip() for l in open(os.path.join(WORK_DIR, "author_map.txt")) if l.strip()]
    sen = load_senators().set_index("surname")
    party = sen.loc[authors, "party"].astype(str).str.upper()
    init = np.where(party == "R", 1.0, np.where(party == "D", -1.0, 0.0)).astype(np.float32)
    init_path = os.path.join(WORK_DIR, "init_ideal.npy")
    np.save(init_path, init)
    cmd = [sys.executable, "-u", est, "--num-epochs", str(num_epochs),
           "--seed", str(seed), "--num-topics", str(num_topics),
           "--data-dir", WORK_DIR, "--x-override", os.path.join(WORK_DIR, "X.npy"),
           "--init-ideal-npy", init_path, "--output-dir", out]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, env=dict(os.environ, TF_USE_LEGACY_KERAS="1"), check=True)


# ----------------------------- correlate --------------------------- #
def _latest_ideal(res):
    """Return (ideal-location array, label). Prefer the final fit; otherwise the
    latest epoch checkpoint (lets us analyse a run that hit the Metal memory
    leak before completing)."""
    fin = os.path.join(res, "params", "ideal_point_location_final.npy")
    if os.path.exists(fin):
        return np.load(fin), "final"
    cps = glob.glob(os.path.join(res, "params", "ideal_point_location_epoch*.npy"))
    if not cps:
        sys.exit(f"No ideal-point file in {res}/params/.")
    ep = max(int(c.split("epoch")[-1].split(".")[0]) for c in cps)
    return np.load(os.path.join(res, "params",
                                f"ideal_point_location_epoch{ep}.npy")), f"epoch{ep}"


def cmd_correlate(seed=314159, num_topics=25):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    os.makedirs(OUT_DIR, exist_ok=True)
    res = os.path.join(RES_DIR, f"seed_{seed}_K{num_topics}")

    ideal, tag = _latest_ideal(res)
    A, K = ideal.shape
    authors = [l.strip() for l in open(os.path.join(WORK_DIR, "author_map.txt")) if l.strip()]
    # theta-weighted aggregate (final theta if present, else PF-init intensities)
    try:
        ts = np.load(os.path.join(res, "params", "theta_shape_final.npy"))
        tr = np.load(os.path.join(res, "params", "theta_rate_final.npy"))
    except FileNotFoundError:
        ts = np.load(os.path.join(res, "pf_fits", f"document_shape_K{K}.npy"))
        tr = np.load(os.path.join(res, "pf_fits", f"document_rate_K{K}.npy"))
    ai = np.load(os.path.join(WORK_DIR, "author_indices.npy"))
    theta = ts / tr
    w = np.zeros((A, K))
    for a in range(A):
        w[a] = theta[ai == a].sum(0)
    w /= w.sum(1, keepdims=True)
    ip = (w * ideal).sum(1)

    # attach party, main speech ideal point and DW-NOMINATE
    sen = load_senators()[["surname", "party"]]
    df = pd.DataFrame({"surname": [a.upper() for a in authors], "ip_tweet": ip}) \
        .merge(sen, on="surname", how="left")
    # orient axis so Republicans are positive (global sign is not identified)
    if df.loc[df.party == "R", "ip_tweet"].mean() < df.loc[df.party == "D", "ip_tweet"].mean():
        df["ip_tweet"] *= -1
    main = pd.read_csv(DW_MAIN)
    main["surname"] = main["author"].astype(str).str.split().str[-1].str.upper().str.strip()
    m = df.merge(main[["surname", "ip_mean", "dw_nominate"]], on="surname", how="inner")
    mb = m[m["party"].isin(["D", "R"])]

    auc = roc_auc_score((mb["party"] == "R").astype(int), mb["ip_tweet"])
    rs = pearsonr(m["ip_tweet"], m["ip_mean"])
    rd = pearsonr(m["ip_tweet"], m["dw_nominate"])
    sd = spearmanr(m["ip_tweet"], m["dw_nominate"])
    z = lambda v: (v - v.mean()) / v.std()
    print(f"=== TWITTER-114 STBS ideal points ({tag}; {len(df)} senators, "
          f"matched {len(m)}) ===")
    print(f"AUC (R vs D, party separation) : {auc:.3f}")
    print(f"corr(tweet IP, main speech IP) : r={rs[0]:+.3f} (p={rs[1]:.2g})")
    print(f"corr(tweet IP, DW-NOMINATE)    : r={rd[0]:+.3f} (p={rd[1]:.2g}); "
          f"Spearman={sd[0]:+.3f}")
    print(f"dispersion |z|: tweets={np.abs(z(m['ip_tweet'])).mean():.3f}  "
          f"speeches={np.abs(z(m['ip_mean'])).mean():.3f}  "
          f"(>1 => more extreme on Twitter)")
    print("\nmost liberal:")
    print(df.sort_values("ip_tweet").head(8)[["surname", "party", "ip_tweet"]].to_string(index=False))
    print("most conservative:")
    print(df.sort_values("ip_tweet").tail(8)[["surname", "party", "ip_tweet"]].to_string(index=False))

    # scatter plot: Twitter STBS ideal point vs DW-NOMINATE, with OLS line
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    col = {"D": "#2c6fbb", "R": "#c0392b", "I": "#7f7f7f"}
    for p, g in m.groupby("party"):
        ax.scatter(g["dw_nominate"], g["ip_tweet"], s=38, alpha=0.85,
                   edgecolor="none", color=col.get(p, "#7f7f7f"),
                   label={"D": "Democrat", "R": "Republican", "I": "Independent"}.get(p, p))
    b1, b0 = np.polyfit(m["dw_nominate"], m["ip_tweet"], 1)
    xs = np.linspace(m["dw_nominate"].min(), m["dw_nominate"].max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="black", lw=1.2)
    ax.set_xlabel("DW-NOMINATE (first dimension)")
    ax.set_ylabel(r"Twitter-based STBS ideal point $\bar{\mathrm{i}}_a$")
    ax.set_title(f"$r = {rd[0]:.2f}$  ($n={len(m)}$ senators)")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "twitter114_dw_scatter.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "twitter114_dw_scatter.png"), dpi=150)
    plt.close()

    df.sort_values("ip_tweet").to_csv(os.path.join(OUT_DIR, "twitter114_ideal_points.csv"), index=False)
    m.to_csv(os.path.join(OUT_DIR, "twitter114_correlations.csv"), index=False)
    json.dump({"checkpoint": tag, "n_senators": int(len(df)), "n_matched": int(len(m)),
               "auc_RvsD": round(float(auc), 3),
               "r_tweet_speech": round(float(rs[0]), 3),
               "r_tweet_dwnominate": round(float(rd[0]), 3),
               "spearman_tweet_dw": round(float(sd[0]), 3)},
              open(os.path.join(OUT_DIR, "twitter114_summary.json"), "w"), indent=2)
    print(f"\n-> {OUT_DIR}/ (ideal_points.csv, correlations.csv, summary.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ["inspect", "build-map", "prep", "fit", "correlate", "all"]:
        ap.add_argument("--" + f, action="store_true")
    a = vars(ap.parse_args())
    if a["inspect"]:   cmd_inspect()
    if a["build_map"]: cmd_build_map()
    if a["prep"] or a["all"]:      cmd_prep()
    if a["fit"] or a["all"]:       cmd_fit()
    if a["correlate"] or a["all"]: cmd_correlate()
    if not any(a.values()):        ap.print_help()
