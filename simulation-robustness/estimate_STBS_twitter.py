"""
estimate_STBS_without_party.py
==============================
Reviewer R1 (Major Comment 8) asks how much of the heavy lifting is
done by the *party* indicators in the STBS regression layer. STBS
is sold as text-based scaling that should discover the ideological
axis from text alone; once party is in the regression matrix X the
model is no longer purely unsupervised.

This script fits STBS on the 114th Senate corpus with the SAME
hyperparameters and the SAME corpus as the headline fit, but:
  (a) drops the two party-indicator columns from X (Republican and
      Independent dummies; Democrats are the baseline);
  (b) drops every party-by-X interaction (so no party leakage
      through interactions either);
  (c) initialises the ideal points to zero rather than to the
      party-derived +/-1 used in the headline run -- otherwise the
      "no party" claim leaks back in through the warm start.

After fitting it computes the theta-weighted aggregate ideal point
\bar{\hat\imath}_a (the same aggregator used in supplement S.1.5)
and reports the Pearson correlation with the DW-NOMINATE 1st
dimension. The headline (with-party) fit achieves r = 0.856; the
no-party correlation tells the reader how much of that comes from
the party labels in X versus from the text itself.

Usage (run from STBS_CAVI/ for venv reasons, same as 01_estimate_STBS.py):
    ./venv_gpu/bin/python3 ../Revision_code_CAVI/estimate_STBS_without_party.py \
        --num-epochs 200 --seed 314159

Outputs (default): stbs_cavi_results_no_party/seed_<SEED>_K<K>/
    params/{ideal_point_location_final.npy, ideal_loc.csv,
             iota_location_final.npy, iota_loc.csv, ...}
    dw_nominate_no_party_correlation.csv
    dw_nominate_no_party_scatter.{pdf,png}
    summary.json
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# IMPORTANT: every existing successful STBS-CAVI run sets this BEFORE
# importing TensorFlow.  Without it the Keras-3 backend builds model
# variables lazily and model.trainable_variables is empty inside the first
# @tf.function-wrapped train_step, which crashes with
# 'ValueError: not enough values to unpack (expected 2, got 0)' inside
# optim.apply_gradients(zip(grads, model.trainable_variables)).
# Mirrors 02_run_multi_seeds.py (line 72) and run_centered_replicate_fits.sh
# (line 79).
os.environ["TF_USE_LEGACY_KERAS"] = "1"

REPO = os.path.dirname(os.path.abspath(__file__))
STBS_CAVI_DIR = os.path.normpath(os.path.join(REPO, "..", "STBS_CAVI"))
sys.path.insert(0, os.path.join(STBS_CAVI_DIR, "code"))

import tensorflow as tf
import tensorflow_probability as tfp

tf.config.set_soft_device_placement(True)
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs available:", gpus)
if gpus:
    print("Running on Apple Metal GPU (with soft device placement)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("WARNING: No GPU found, running on CPU")

from var_and_prior_family import VariationalFamily, PriorFamily  # noqa: F401
from input_pipeline import build_input_pipeline
from stbs import STBS
from utils import print_topics, print_ideal_points  # noqa: F401
from poisson_factorization import (
    get_normalizer, update_variational_document_shape,
    update_variational_document_rate, update_variational_document_prior_rate,
    update_variational_topic_shape, update_variational_topic_rate,
    update_variational_topic_prior_rate,
)
import scipy.sparse as sparse


@tf.function
def train_step(model, inputs, outputs, optim, seed, step=None):
    model.perform_cavi_updates(inputs, outputs, step)
    with tf.GradientTape() as tape:
        recon_loss_batch, log_prior_loss, entropy_loss, seed = model(
            inputs, outputs, seed, nsamples=model.num_samples)
        recon_loss = recon_loss_batch * model.minibatch_scaling
        total_loss = recon_loss + log_prior_loss + entropy_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, recon_loss, log_prior_loss, entropy_loss, seed


# ====================================================================
# CLI
# ====================================================================
parser = argparse.ArgumentParser(
    description="Estimate STBS *without* the party indicators (R1 ablation).")
parser.add_argument("--seed", type=int, default=314159,
                    help="Random seed (default: 314159, matches headline fit).")
parser.add_argument("--num-epochs", type=int, default=200,
                    help="Number of epochs (default: 200).")
parser.add_argument("--num-topics", type=int, default=10,
                    help="K (default: 10; smaller/homogeneous tweet corpus, "
                         "also relieves the Metal unified-memory pressure).")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output dir (default: stbs_cavi_results_no_party/seed_<SEED>_K<K>).")
parser.add_argument("--init-ideal", type=str, default="zero",
                    choices=["zero", "noise", "random_pm1", "party_pm1"],
                    help="How to initialise the per-author ideal points: "
                         "'zero' (default) -- all 0; "
                         "'noise' -- N(0, 0.1^2); "
                         "'random_pm1' -- uniform random +/-1 (no party); "
                         "'party_pm1' -- TBIP convention D=-1, R=+1, I=0 "
                         "(this DOES leak party info via the init).")
parser.add_argument("--dw-nominate-csv", type=str,
                    default=os.path.join(REPO, "stbs_cavi_results",
                                          "comparison_dw_nominate.csv"),
                    help="CSV with author + dw_nominate columns for the "
                         "post-fit correlation analysis.")
parser.add_argument("--skip-fit", action="store_true",
                    help="Skip the actual training and just re-run the "
                         "DW-NOMINATE correlation analysis on an existing fit "
                         "in output-dir/params/. Useful for iterating on "
                         "the post-processing.")
parser.add_argument("--data-dir", type=str, default=None,
                    help="Override input data dir (clean/ with counts.npz, "
                         "author_map.txt, author_indices.npy, vocabulary.txt).")
parser.add_argument("--x-override", type=str, default=None,
                    help="Path to .npy covariate matrix X (A x L); skips the "
                         "built-in gender-only X construction.")
parser.add_argument("--init-ideal-npy", type=str, default=None,
                    help="Path to .npy with per-author initial ideal-point "
                         "locations (A,) or (A,K); overrides --init-ideal.")
cli_args = parser.parse_args()


# ====================================================================
# CONFIGURATION  (mirrors 01_estimate_STBS.py exactly except for X
# construction and ideal-point init -- so the comparison is clean)
# ====================================================================
data_name = "candidate-tweets"
addendum = ""          # twitter files have no 114-style suffix

num_topics = cli_args.num_topics
seed = cli_args.seed
num_epochs = cli_args.num_epochs

batch_size = 512
num_samples = 1
learning_rate = 0.01
RobMon_exponent = -0.7
exact_entropy = True
geom_approx = False
aux_prob_sparse = False
iota_coef_jointly = True
save_every = max(num_epochs // 10, 10)
print_steps = 500

pf_max_steps = 300
pf_seed = seed

prior_choice = {
    "theta": "Garte",
    "exp_verbosity": "None",
    "beta": "Gvrte",
    "eta": "NkprecF",
    "ideal_dim": "ak",
    "ideal_mean": "Nreg",
    "ideal_prec": "Naprec",
    "iota_dim": "kl",
    "iota_prec": "NlprecF",
    "iota_mean": "Nlmean",
}

eta_kappa = 10.0
iota_kappa = 10.0
eta_prec_shp = 0.3
eta_prec_rate_shp = 0.3
eta_prec_rate_rte = eta_prec_shp / eta_prec_rate_shp * eta_kappa / 2.0
iota_prec_shp = 0.3
iota_prec_rate_shp = 0.3
iota_prec_rate_rte = iota_prec_shp / iota_prec_rate_shp * iota_kappa / 2.0

prior_hyperparameter = {
    "theta": {"shape": 0.3, "rate": 0.3},
    "theta_rate": {"shape": 0.3, "rate": 0.3 / 0.3},
    "beta": {"shape": 0.3, "rate": 0.3},
    "beta_rate": {"shape": 0.3, "rate": 0.3 / 0.3},
    "exp_verbosity": {"location": 0.0, "scale": 1.0, "shape": 0.3, "rate": 0.3},
    "eta": {"location": 0.0, "scale": 1.0},
    "eta_prec": {"shape": eta_prec_rate_shp, "rate": eta_prec_shp * 2.0 / eta_kappa},
    "eta_prec_rate": {"shape": eta_prec_shp, "rate": eta_prec_rate_rte},
    "ideal": {"location": 0.0, "scale": 1.0},
    "ideal_prec": {"shape": 0.3, "rate": 0.3},
    "iota": {"location": 0.0, "scale": 1.0},
    "iota_prec": {"shape": iota_prec_rate_shp, "rate": iota_prec_shp * 2.0 / iota_kappa},
    "iota_prec_rate": {"shape": iota_prec_shp, "rate": iota_prec_rate_rte},
    "iota_mean": {"location": 0.0, "scale": 1.0},
}

# ====================================================================
# OUTPUT
# ====================================================================
if cli_args.output_dir is not None:
    output_dir = cli_args.output_dir
else:
    output_dir = os.path.join(
        REPO, "stbs_cavi_results_twitter",
        f"seed_{seed}_K{num_topics}",
    )
os.makedirs(output_dir, exist_ok=True)
param_dir = os.path.join(output_dir, "params")
os.makedirs(param_dir, exist_ok=True)
fig_dir = os.path.join(output_dir, "figs")
os.makedirs(fig_dir, exist_ok=True)

# ====================================================================
# LOAD DATA
# ====================================================================
print("=" * 60)
print(f"STBS on TWITTER candidate-tweets-2020 ({data_name})")
print(f"  K={num_topics}, seed={seed}, epochs={num_epochs}")
print(f"  init-ideal = {cli_args.init_ideal}")
print(f"  output     = {output_dir}")
print("=" * 60)

tf.random.set_seed(seed)
random_state = np.random.RandomState(seed)

data_dir = cli_args.data_dir if cli_args.data_dir is not None else \
    os.path.join(REPO, "data_twitter", "candidate-tweets-2020-bigrams", "clean")

(dataset, permutation, all_author_indices, vocabulary, author_map,
 author_info) = build_input_pipeline(
    data_name, data_dir, batch_size, random_state, None, "nothing", addendum)

num_documents = len(permutation)
num_words = len(vocabulary)
num_authors = len(author_map)        # twitter loader returns author_info=None

print(f"TWITTER candidate-tweets-2020: "
      f"Documents={num_documents}, Words={num_words}, Authors={num_authors}")

# ====================================================================
# Build the covariate matrix X for the candidate-tweets corpus from
# candidate_tweets_covariates.csv (manually coded; see that file's
# `covariate_source` column -- gender/generation for the six 114th-
# Senate candidates are taken from the Senate covariates, the rest
# from public record). All 19 candidates are Democrats, so there is
# no party covariate and no party-aligned initialisation.
#
# Covariates: intercept + gender only. All 19 candidates are Democrats,
# so there is no party variation (a party column would be constant and
# collinear with the intercept); office/generation are available in
# candidate_tweets_covariates.csv but, with only 19 authors, are left
# out to keep the regression layer parsimonious. gender is the single
# author-level covariate.
# ====================================================================
if cli_args.x_override is not None:
    X_np = np.load(cli_args.x_override).astype(np.float32)
    assert X_np.shape[0] == num_authors, \
        f"X-override has {X_np.shape[0]} rows, expected {num_authors} authors"
    labels = [f"x{j}" for j in range(X_np.shape[1])]
    print(f"[X-construction] loaded X override {cli_args.x_override}: {X_np.shape}")
else:
    print("\n[X-construction] building covariate matrix (intercept + gender)...")
    cov = pd.read_csv(os.path.join(REPO, "candidate_tweets_covariates.csv"))
    # order covariate rows to match the model's author_map order
    cov = cov.set_index("handle").loc[list(author_map)].reset_index()
    assert len(cov) == num_authors, "covariate/author_map length mismatch"
    cols, labels = [], []
    cols.append(np.ones(num_authors));               labels.append("intercept")
    cols.append((cov["gender"] == "F").to_numpy(float)); labels.append("gender_F")
    X_np = np.stack(cols, axis=1).astype(np.float32)   # (A, L)
    print(f"  X shape: {X_np.shape}  (L={X_np.shape[1]} covariates)")
    print(f"  Covariate labels: {labels}")

X = tf.constant(X_np, dtype=tf.float32)
pd.DataFrame(X_np, columns=labels).to_csv(
    os.path.join(param_dir, "X_twitter.csv"), index=False)
with open(os.path.join(param_dir, "X_labels.json"), "w") as fh:
    json.dump({"labels": labels, "shape": list(X_np.shape)}, fh, indent=2)

# ====================================================================
# *** SECOND DELTA: NEUTRAL ideal-point initialisation ***
# 01_estimate_STBS.py initialises ideal_location to +/-1 based on
# party.  That re-introduces party through the warm start.  We use a
# neutral init instead (zero, or small Gaussian noise to break
# symmetry).  Sign of the recovered axis is identifiable only up to
# a global flip, which we resolve post-hoc against DW-NOMINATE.
# ====================================================================
if cli_args.init_ideal_npy is not None:
    auxloc = np.load(cli_args.init_ideal_npy).astype(np.float32).ravel()
    assert auxloc.shape[0] == num_authors, \
        f"init-ideal-npy has {auxloc.shape[0]} rows, expected {num_authors}"
    print(f"\n[init] ideal_location loaded from {cli_args.init_ideal_npy} "
          f"(shape {auxloc.shape}); #(+)= {int((auxloc>0).sum())}, "
          f"#(-)= {int((auxloc<0).sum())}.")
elif cli_args.init_ideal == "zero":
    auxloc = np.zeros(num_authors, dtype=np.float32)
    print("\n[init] ideal_location initialised to ZERO for every author.")
elif cli_args.init_ideal == "noise":
    rng_init = np.random.default_rng(seed)
    auxloc = rng_init.normal(scale=0.1, size=num_authors).astype(np.float32)
    print(f"\n[init] ideal_location initialised to small Gaussian noise "
          f"(scale 0.1, seed {seed}).")
elif cli_args.init_ideal == "random_pm1":
    rng_init = np.random.default_rng(seed)
    auxloc = rng_init.choice([-1.0, 1.0], size=num_authors).astype(np.float32)
    n_plus, n_minus = int((auxloc > 0).sum()), int((auxloc < 0).sum())
    print(f"\n[init] ideal_location: RANDOM +/-1 (seed {seed}), "
          f"#(+1)={n_plus}, #(-1)={n_minus}. NO party info used.")
elif cli_args.init_ideal == "party_pm1":
    # TBIP convention: D = -1, R = +1, I = 0.
    # Runtime author_map is CamelCase-concatenated and carries no "(R)/(D)"
    # suffix, so we load the original author_map114.txt to recover party.
    am_path = os.path.join(STBS_CAVI_DIR, "data", "hein-daily", "clean",
                           "author_map114.txt")
    with open(am_path) as _f:
        _full_names = [l.strip() for l in _f if l.strip()]
    if len(_full_names) != num_authors:
        raise ValueError(
            f"author_map114.txt has {len(_full_names)} rows, "
            f"expected {num_authors}.")
    party_init = []
    for nm in _full_names:
        if "(R)" in nm:
            party_init.append(+1.0)
        elif "(D)" in nm:
            party_init.append(-1.0)
        else:
            party_init.append(0.0)
    auxloc = np.array(party_init, dtype=np.float32)
    n_R = int((auxloc > 0).sum())
    n_D = int((auxloc < 0).sum())
    n_I = int((auxloc == 0).sum())
    print(f"\n[init] ideal_location: BY PARTY (TBIP convention: D=-1, R=+1, I=0). "
          f"#R={n_R}, #D={n_D}, #I={n_I}. "
          f"WARNING: this DOES leak party info into the fit via the init.")
else:
    raise ValueError(f"unknown --init-ideal: {cli_args.init_ideal}")

ideal_topic_dim = num_topics  # ak parameterisation
initial_ideal_location = tf.repeat(
    tf.constant(auxloc)[:, None], ideal_topic_dim, axis=1)
print(f"  initial_ideal_location shape: {initial_ideal_location.shape}")

# ====================================================================
# Monkey-patch (Metal GPU): disable IsFinite-based checks
# ====================================================================
STBS.print_non_finite_parameters = lambda self, *a, **kw: None
STBS.check_and_print_non_finite = lambda self, *a, **kw: None

# Skip-fit fast-path: jump straight to the DW-NOMINATE correlation
if cli_args.skip_fit:
    print("\n--skip-fit set: skipping training, going to post-processing.")
else:
    # ================================================================
    # PF pre-initialisation (exactly as 01_estimate_STBS.py)
    # ================================================================
    counts_sparse = sparse.load_npz(os.path.join(data_dir,
                                                  "counts" + addendum + ".npz"))
    counts_dense = np.array(counts_sparse.todense()).astype(np.float64)

    print("\n" + "=" * 60)
    print(f"Running Poisson Factorization for {pf_max_steps} steps...")
    print("=" * 60)

    pf_random_state = np.random.RandomState(pf_seed)
    prior_shape_pf = 0.3
    hyperprior_shape_pf = 0.3
    hyperprior_rate_pf = 0.3

    variational_topic_shape = np.exp(0.1 * pf_random_state.randn(num_topics, num_words))
    variational_topic_rate = np.exp(0.1 * pf_random_state.randn(num_topics, num_words))
    variational_document_shape = np.exp(0.1 * pf_random_state.randn(num_documents, num_topics))
    variational_document_rate = np.exp(0.1 * pf_random_state.randn(num_documents, num_topics))
    variational_document_prior_rate = np.exp(0.1 * pf_random_state.randn(num_documents))
    variational_topic_prior_rate = np.exp(0.1 * pf_random_state.randn(num_words))
    variational_document_prior_shape = hyperprior_shape_pf + num_topics * prior_shape_pf
    variational_topic_prior_shape = hyperprior_shape_pf + num_topics * prior_shape_pf

    pf_start = time.time()
    for pf_step in range(pf_max_steps):
        normalizer = get_normalizer(
            variational_topic_shape, variational_topic_rate,
            variational_document_shape, variational_document_rate)
        new_doc_shape = update_variational_document_shape(
            counts_dense, variational_topic_shape, variational_topic_rate,
            variational_document_shape, variational_document_rate,
            normalizer, prior_shape_pf)
        new_doc_rate = update_variational_document_rate(
            variational_topic_shape, variational_topic_rate,
            variational_document_prior_shape, variational_document_prior_rate)
        new_doc_prior_rate = update_variational_document_prior_rate(
            new_doc_shape, new_doc_rate, hyperprior_rate_pf)
        new_topic_shape = update_variational_topic_shape(
            counts_dense, variational_topic_shape, variational_topic_rate,
            variational_document_shape, variational_document_rate,
            normalizer, prior_shape_pf)
        new_topic_rate = update_variational_topic_rate(
            new_doc_shape, new_doc_rate,
            variational_topic_prior_shape, variational_topic_prior_rate)
        new_topic_prior_rate = update_variational_topic_prior_rate(
            new_topic_shape, new_topic_rate, hyperprior_rate_pf)

        variational_document_shape = new_doc_shape
        variational_document_rate = new_doc_rate
        variational_document_prior_rate = new_doc_prior_rate
        variational_topic_shape = new_topic_shape
        variational_topic_rate = new_topic_rate
        variational_topic_prior_rate = new_topic_prior_rate

        if pf_step % 50 == 0 or pf_step == pf_max_steps - 1:
            print(f"  PF step {pf_step:>3d}/{pf_max_steps}  "
                  f"({time.time()-pf_start:.0f}s)")

    fitted_document_shape = variational_document_shape.astype(np.float32)
    fitted_document_rate = variational_document_rate.astype(np.float32)
    fitted_topic_shape = variational_topic_shape.astype(np.float32)
    fitted_topic_rate = variational_topic_rate.astype(np.float32)
    del counts_dense

    pf_dir = os.path.join(output_dir, "pf_fits")
    os.makedirs(pf_dir, exist_ok=True)
    np.save(os.path.join(pf_dir, f"document_shape_K{num_topics}{addendum}"),
            variational_document_shape)
    np.save(os.path.join(pf_dir, f"document_rate_K{num_topics}{addendum}"),
            variational_document_rate)
    np.save(os.path.join(pf_dir, f"topic_shape_K{num_topics}{addendum}"),
            variational_topic_shape)
    np.save(os.path.join(pf_dir, f"topic_rate_K{num_topics}{addendum}"),
            variational_topic_rate)
    print(f"PF complete in {time.time()-pf_start:.0f}s")

    # ================================================================
    # MODEL + TRAINING
    # ================================================================
    optim = tf.optimizers.Adam(learning_rate=learning_rate)
    model = STBS(num_documents, num_topics, num_words, num_authors, num_samples,
                 X, all_author_indices,
                 initial_ideal_location=initial_ideal_location,
                 fitted_document_shape=fitted_document_shape,
                 fitted_document_rate=fitted_document_rate,
                 fitted_objective_topic_shape=fitted_topic_shape,
                 fitted_objective_topic_rate=fitted_topic_rate,
                 prior_hyperparameter=prior_hyperparameter,
                 prior_choice=prior_choice,
                 batch_size=batch_size,
                 RobMon_exponent=RobMon_exponent,
                 exact_entropy=exact_entropy,
                 geom_approx=geom_approx,
                 aux_prob_sparse=aux_prob_sparse,
                 iota_coef_jointly=iota_coef_jointly)

    print(f"\nTraining for {num_epochs} epochs...")
    _, seed_tf = tfp.random.split_seed(seed)
    model_state = {"ELBO": [], "entropy": [], "log_prior": [],
                   "reconstruction": [], "epoch": [], "batch": [], "step": []}

    start_time = time.time()
    batches_per_epoch = len(dataset)
    for epoch in range(num_epochs):
        for batch_index, batch in enumerate(iter(dataset)):
            step = batches_per_epoch * epoch + batch_index
            inputs, outputs = batch
            (total_loss, recon_loss, log_prior_loss, entropy_loss,
             seed_tf) = train_step(model, inputs, outputs, optim,
                                   seed_tf, tf.constant(step))
            model_state["ELBO"].append(-total_loss.numpy())
            model_state["entropy"].append(-entropy_loss.numpy())
            model_state["log_prior"].append(-log_prior_loss.numpy())
            model_state["reconstruction"].append(-recon_loss.numpy())
            model_state["epoch"].append(epoch)
            model_state["batch"].append(batch_index)
            model_state["step"].append(step)
            if step % print_steps == 0:
                print(f"Step {step:>6d} | Epoch {epoch:>3d} | "
                      f"ELBO={-total_loss.numpy():>12.1f} | "
                      f"({time.time()-start_time:.0f}s)")
        sec_per_epoch = (time.time() - start_time) / (epoch + 1)
        print(f"Epoch {epoch:>3d} done | "
              f"ELBO={-total_loss.numpy():>12.1f} | "
              f"({sec_per_epoch:.1f} sec/epoch)")
        if (epoch + 1) % save_every == 0:
            np.save(os.path.join(param_dir, f"ideal_point_location_epoch{epoch}"),
                    model.ideal_varfam.location.numpy())
            np.save(os.path.join(param_dir, f"ideal_point_scale_epoch{epoch}"),
                    model.ideal_varfam.scale.numpy())
            # also checkpoint iota (covariate coefficients) so we keep them
            # even if the Metal training-loop leak kills the run before the end
            if hasattr(model.iota_varfam, "location"):
                np.save(os.path.join(param_dir, f"iota_location_epoch{epoch}"),
                        model.iota_varfam.location.numpy())
            print(f"  -> checkpoint at epoch {epoch}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.0f}s "
          f"({total_time/60:.1f} min)")

    # ================================================================
    # SAVE
    # ================================================================
    ideal_loc = model.ideal_varfam.location.numpy()
    ideal_scl = model.ideal_varfam.scale.numpy()
    np.save(os.path.join(param_dir, "ideal_point_location_final"), ideal_loc)
    np.save(os.path.join(param_dir, "ideal_point_scale_final"), ideal_scl)

    ip_cols = [f"topic_{k}" for k in range(num_topics)]
    ip_df = pd.DataFrame(ideal_loc, columns=ip_cols)
    ip_df.insert(0, "author", author_map)
    ip_df.to_csv(os.path.join(output_dir, "ideal_points.csv"), index=False)

    if hasattr(model.iota_varfam, "location"):
        iota_loc = model.iota_varfam.location.numpy()
        np.save(os.path.join(param_dir, "iota_location_final"), iota_loc)
        pd.DataFrame(iota_loc).to_csv(
            os.path.join(param_dir, "iota_loc.csv"), index=False, header=False)
        if hasattr(model.iota_varfam, "scale_tril"):
            np.save(os.path.join(param_dir, "iota_scale_tril_final"),
                    model.iota_varfam.scale_tril.numpy())

    pd.DataFrame(ideal_loc).to_csv(
        os.path.join(param_dir, "ideal_loc.csv"), index=False, header=False)
    pd.DataFrame(ideal_scl).to_csv(
        os.path.join(param_dir, "ideal_scl.csv"), index=False, header=False)

    eta_loc = model.eta_varfam.location.numpy()
    eta_scl = model.eta_varfam.scale.numpy()
    np.save(os.path.join(param_dir, "eta_location_final"), eta_loc)
    np.save(os.path.join(param_dir, "eta_scale_final"), eta_scl)
    pd.DataFrame(eta_loc).to_csv(
        os.path.join(param_dir, "eta_loc.csv"), index=False, header=False)

    beta_shp = model.beta_varfam.shape.numpy()
    beta_rte = model.beta_varfam.rate.numpy()
    np.save(os.path.join(param_dir, "beta_shape_final"), beta_shp)
    np.save(os.path.join(param_dir, "beta_rate_final"), beta_rte)

    theta_shp = model.theta_varfam.shape.numpy()
    theta_rte = model.theta_varfam.rate.numpy()
    np.save(os.path.join(param_dir, "theta_shape_final"), theta_shp)
    np.save(os.path.join(param_dir, "theta_rate_final"), theta_rte)

    pd.DataFrame(model_state).to_csv(
        os.path.join(output_dir, "training_loss.csv"), index=False)

    print(f"Saved fits to {param_dir}")

# ====================================================================
# POST-FIT: theta-weighted aggregate ideal point + candidate ranking
# (no DW-NOMINATE here: these are presidential candidates, not the
#  114th Senate, so there is no external roll-call benchmark).
# ====================================================================
print("\n" + "=" * 60)
print("Twitter candidate ideal-point ranking (theta-weighted aggregate)")
print("=" * 60)

# Re-load whatever was just saved (also works in --skip-fit mode)
ideal_loc = np.load(os.path.join(param_dir, "ideal_point_location_final.npy"))
theta_shp = np.load(os.path.join(param_dir, "theta_shape_final.npy"))
theta_rte = np.load(os.path.join(param_dir, "theta_rate_final.npy"))
print(f"  ideal_loc shape: {ideal_loc.shape}   "
      f"(should be ({num_authors}, {num_topics}))")
print(f"  theta_shp shape: {theta_shp.shape}   "
      f"(should be (D, {num_topics}))")

# E[theta_dk] = theta_shp / theta_rte
theta_mean = theta_shp / theta_rte   # (D, K)

# Aggregate per author: average theta over documents authored by that author
# all_author_indices is a long (sum_d N_d) array of author indices per token;
# easier here to recover document->author from author_info / build_input_pipeline.
# We use the same theta-weighted aggregate as supplement S.1.5:
#   w_{a,k} \propto sum over docs d authored by a of E[theta_{d,k}]
#   \bar{x}_a = sum_k w_{a,k} * \hat\imath_{a,k}
#
# Recover doc->author from clean/author_indices file
auth_idx_file = os.path.join(data_dir, f"author_indices{addendum}.npy")
if os.path.exists(auth_idx_file):
    doc_author = np.load(auth_idx_file)
else:
    # Fallback: try author_indices.csv  (per-document)
    candidates = [
        os.path.join(data_dir, f"author_indices{addendum}.csv"),
        os.path.join(data_dir, "author_indices.csv"),
    ]
    doc_author = None
    for c in candidates:
        if os.path.exists(c):
            doc_author = pd.read_csv(c, header=None).iloc[:, 0].to_numpy()
            break
    if doc_author is None:
        raise FileNotFoundError(
            f"Could not find author_indices file in {data_dir}; expected one "
            f"of: author_indices{addendum}.npy, author_indices.csv")
print(f"  doc->author array: {doc_author.shape}")

theta_per_author = np.zeros((num_authors, num_topics), dtype=np.float32)
counts = np.zeros(num_authors, dtype=np.int64)
for d in range(theta_mean.shape[0]):
    a = int(doc_author[d])
    theta_per_author[a] += theta_mean[d]
    counts[a] += 1
nz = counts > 0
theta_per_author[nz] /= counts[nz][:, None]

# Now form the weighted aggregate IP per author
w = theta_per_author / theta_per_author.sum(axis=1, keepdims=True).clip(min=1e-12)
ip_agg = (w * ideal_loc).sum(axis=1)   # (A,)
print(f"  theta-weighted aggregate ideal point: range "
      f"[{ip_agg.min():+.3f}, {ip_agg.max():+.3f}]")

# Candidate-level aggregate ideal-point ranking. The sign of an STBS
# axis is identifiable only up to a global flip; we orient it so that
# Bernie Sanders (a widely-recognised progressive anchor) sits on the
# negative end, giving the conventional left(-)/right(+) reading within
# the Democratic field.
ranking = pd.DataFrame({
    "handle": list(author_map),
    "name": cov["name"].to_numpy(),
    "office": cov["office"].to_numpy(),
    "gender": cov["gender"].to_numpy(),
    "aggregate_ideal_point": ip_agg,
}).sort_values("aggregate_ideal_point").reset_index(drop=True)

sanders_ip = float(ranking.loc[ranking.handle == "Berniesanders",
                               "aggregate_ideal_point"].values[0])
if sanders_ip > ranking["aggregate_ideal_point"].median():
    ranking["aggregate_ideal_point"] *= -1
    ip_agg = -ip_agg
    ideal_loc = -ideal_loc
    ranking = ranking.sort_values("aggregate_ideal_point").reset_index(drop=True)
    print("  flipped global sign so Sanders anchors the negative (progressive) end.")

print("\n  Candidate ranking (aggregate ideal point, ascending):")
print(ranking.to_string(index=False,
                        formatters={"aggregate_ideal_point": "{:+.3f}".format}))

ranking.to_csv(os.path.join(output_dir, "twitter_candidate_ranking.csv"),
               index=False)

# per-author per-topic ideal points (oriented consistently with ip_agg)
topic_cols = [f"topic_{k}" for k in range(num_topics)]
pd.DataFrame(ideal_loc, columns=topic_cols).assign(
    handle=list(author_map), name=cov["name"].to_numpy()
).to_csv(os.path.join(output_dir, "twitter_ideal_points.csv"), index=False)

# horizontal lollipop plot of the aggregate ranking
fig, ax = plt.subplots(figsize=(7, 7))
yy = np.arange(len(ranking))
colours = ["#2874a6" if g == "F" else "#7f8c8d"
           for g in ranking["gender"]]
ax.hlines(yy, 0, ranking["aggregate_ideal_point"], color="grey", lw=0.8)
ax.scatter(ranking["aggregate_ideal_point"], yy, c=colours, s=45,
           edgecolor="black", lw=0.4, zorder=3)
ax.set_yticks(yy)
ax.set_yticklabels(ranking["name"], fontsize=9)
ax.axvline(0, color="black", lw=0.5, ls=":")
ax.set_xlabel(r"STBS aggregate ideal point  $\bar{\hat\imath}_a$ "
              r"($\theta$-weighted)", fontsize=11)
ax.set_title("2020 Democratic candidates on Twitter\n"
             "(STBS topic-varying ideal points, aggregated; "
             "sign oriented so Sanders is negative)", fontsize=10)
ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "twitter_candidate_ranking.pdf"),
            bbox_inches="tight")
fig.savefig(os.path.join(output_dir, "twitter_candidate_ranking.png"),
            dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  -> wrote twitter_candidate_ranking.{{csv,pdf,png}} and "
      f"twitter_ideal_points.csv")

# Summary JSON
summary = dict(
    dataset="candidate-tweets-2020",
    seed=int(seed), num_topics=int(num_topics), num_epochs=int(num_epochs),
    num_authors=int(num_authors), num_documents=int(num_documents),
    L_covariates_in_X=int(X_np.shape[1]), covariate_labels=labels,
    aggregate_ip_range=[float(ip_agg.min()), float(ip_agg.max())],
    most_progressive=ranking.iloc[0]["name"],
    most_moderate=ranking.iloc[-1]["name"],
)
with open(os.path.join(output_dir, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"  -> wrote summary.json")

print("\nDone.")
print(f"  All outputs under: {output_dir}")
print(f"  most progressive: {ranking.iloc[0]['name']}  "
      f"({ranking.iloc[0]['aggregate_ideal_point']:+.3f})")
print(f"  most moderate:    {ranking.iloc[-1]['name']}  "
      f"({ranking.iloc[-1]['aggregate_ideal_point']:+.3f})")
