"""
estimate_STBS.py
================
Estimate the STBS model on the 114th Senate data using the original
TensorFlow/CAVI implementation with Apple Metal GPU acceleration.

Usage:
    cd /Users/paul.hofmarcher/Desktop/PolAn_Revision/STBS_CAVI
    ./venv_gpu/bin/python3 ../Revision_code_CAVI/estimate_STBS.py

The script imports the STBS code from STBS_CAVI/code/ and uses the
data from STBS_CAVI/data/hein-daily/clean/.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress TF C++ info/warning logs (Metal GPU random op notes, etc.)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================================================================== #
# Setup paths — must run from STBS_CAVI/ directory
# ================================================================== #
# Add the STBS code directory to the path
STBS_CAVI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'STBS_CAVI')
STBS_CAVI_DIR = os.path.normpath(STBS_CAVI_DIR)
sys.path.insert(0, os.path.join(STBS_CAVI_DIR, 'code'))

import tensorflow as tf
import tensorflow_probability as tfp

# Enable soft device placement — Metal GPU doesn't support all ops (e.g. IsFinite),
# this lets TF fall back to CPU for unsupported operations.
tf.config.set_soft_device_placement(True)

# Check GPU availability
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)
if gpus:
    print("Running on Apple Metal GPU (with soft device placement)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("WARNING: No GPU found, running on CPU")

from var_and_prior_family import VariationalFamily, PriorFamily
from input_pipeline import build_input_pipeline
from stbs import STBS
from create_X import create_X
from utils import print_topics, print_ideal_points


# Fixed train_step for Keras 3.x compatibility:
# model.num_samples (int) must be passed as keyword argument, not positional.
@tf.function
def train_step(model, inputs, outputs, optim, seed, step=None):
    model.perform_cavi_updates(inputs, outputs, step)
    with tf.GradientTape() as tape:
        reconstruction_loss_batch, log_prior_loss, entropy_loss, seed = model(
            inputs, outputs, seed, nsamples=model.num_samples)
        reconstruction_loss = reconstruction_loss_batch * model.minibatch_scaling
        total_loss = reconstruction_loss + log_prior_loss + entropy_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    # Skip print_non_finite_parameters — uses IsFinite which is not supported on Metal GPU
    return total_loss, reconstruction_loss, log_prior_loss, entropy_loss, seed
from poisson_factorization import (
    get_normalizer, update_variational_document_shape,
    update_variational_document_rate, update_variational_document_prior_rate,
    update_variational_topic_shape, update_variational_topic_rate,
    update_variational_topic_prior_rate,
)
import scipy.sparse as sparse

# ================================================================== #
# CLI ARGUMENTS (optional overrides)
# ================================================================== #
parser = argparse.ArgumentParser(description="Estimate STBS model")
parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides default)")
parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs (overrides default)")
parser.add_argument("--num-topics", type=int, default=None, help="Number of topics K (overrides default)")
parser.add_argument("--output-dir", type=str, default=None, help="Output directory (overrides default)")
parser.add_argument("--data-dir", type=str, default=None,
                    help="Override the data directory (default: "
                         "STBS_CAVI/data/hein-daily/clean). Used to run "
                         "STBS on simulated datasets.")
parser.add_argument("--x-override", type=str, default=None,
                    help="Path to a .npy file containing a custom (N, L) "
                         "covariate matrix X. If set, bypasses "
                         "create_X_hein_daily and uses this X directly "
                         "(used for simulation studies).")
parser.add_argument("--warm-start-dir", type=str, default=None,
                    help="Path to a directory containing ground-truth or "
                         "previously-fit parameters. If set, skips the "
                         "Poisson Factorization init and uses these values "
                         "as the initial variational parameters. Expected "
                         "files in the directory (either .npy or .csv):\n"
                         "  beta_shp[_final].npy / .csv  and  "
                         "beta_rte[_final].npy / .csv\n"
                         "  theta_shp[_final].npy / .csv  and  "
                         "theta_rte[_final].npy / .csv")
parser.add_argument("--hp-overrides", type=str, default=None,
                    help="JSON string of prior_hyperparameter overrides, e.g. "
                         "'{\"theta\":{\"shape\":0.1},\"theta_rate\":{\"shape\":0.1}}'")
parser.add_argument("--ideal-dim", type=str, default="ak", choices=["ak", "a"],
                    help="Ideal-point parameterisation: 'ak' (per-author per-topic, "
                         "full STBS, default) or 'a' (per-author only, TBIP-style "
                         "constant-IP). 'a' implies --iota-dim=l.")
parser.add_argument("--iota-dim", type=str, default=None, choices=["kl", "l"],
                    help="Regression-coefficient parameterisation: 'kl' "
                         "(per-topic per-covariate, full STBS) or 'l' (per-covariate "
                         "shared across topics). Required when --ideal-dim=a (must "
                         "be 'l'). Default: 'kl' if --ideal-dim=ak, 'l' if "
                         "--ideal-dim=a.")
cli_args, _ = parser.parse_known_args()

# Resolve iota-dim default given ideal-dim
if cli_args.iota_dim is None:
    cli_args.iota_dim = "l" if cli_args.ideal_dim == "a" else "kl"
if cli_args.ideal_dim == "a" and cli_args.iota_dim != "l":
    raise ValueError("--ideal-dim=a requires --iota-dim=l (check_prior.py "
                     "rejects ideal_dim=a + iota_dim=kl).")

# ================================================================== #
# CONFIGURATION
# ================================================================== #

# Data settings
data_name = "hein-daily"
addendum = "114"
covariates = "all"  # Full covariate structure as in the paper

# Model settings
num_topics = cli_args.num_topics if cli_args.num_topics is not None else 25
batch_size = 512
num_samples = 1
learning_rate = 0.01
RobMon_exponent = -0.7
exact_entropy = True
geom_approx = False
aux_prob_sparse = False
iota_coef_jointly = True
seed = cli_args.seed if cli_args.seed is not None else 314159

# Training settings
num_epochs = cli_args.num_epochs if cli_args.num_epochs is not None else 50
save_every = max(num_epochs // 10, 10)  # save ~10 checkpoints
print_steps = 500

# PF initialization settings
pf_max_steps = 300
pf_seed = seed  # PF seed also follows main seed

# Prior choices — full STBS model as in the paper
prior_choice = {
    "theta": "Garte",        # Author-specific rates for theta
    "exp_verbosity": "None",  # No verbosity (handled by Garte)
    "beta": "Gvrte",          # Word-specific rates for beta
    "eta": "NkprecF",         # Topic-specific precisions with flexible rates
    "ideal_dim": cli_args.ideal_dim,  # 'ak' (default) or 'a' (constant-IP)
    "ideal_mean": "Nreg",     # Regression on ideal point means
    "ideal_prec": "Naprec",   # Author-specific precisions
    "iota_dim": cli_args.iota_dim,    # 'kl' (default) or 'l' (constant-iota)
    "iota_prec": "NlprecF",   # Coefficient-specific precisions with flexible rates
    "iota_mean": "Nlmean",    # Each coefficient has its own prior mean
}

# Prior hyperparameters — as in the paper (Section 3.1.1)
eta_kappa = 10.0
iota_kappa = 10.0

# For NkprecF: switch shapes and adjust rates by kappa
eta_prec_shp = 0.3   # becomes eta_prec_rate shape
eta_prec_rate_shp = 0.3  # becomes eta_prec shape
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

# Apply CLI hyperparameter overrides (for sensitivity analysis)
if cli_args.hp_overrides is not None:
    import json
    hp_ov = json.loads(cli_args.hp_overrides)
    for key, sub_dict in hp_ov.items():
        if key in prior_hyperparameter:
            prior_hyperparameter[key].update(sub_dict)
        else:
            prior_hyperparameter[key] = sub_dict
    print(f"Applied hyperparameter overrides: {hp_ov}")

# ================================================================== #
# OUTPUT DIRECTORIES
# ================================================================== #
if cli_args.output_dir is not None:
    output_dir = cli_args.output_dir
else:
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stbs_cavi_results')
os.makedirs(output_dir, exist_ok=True)
fig_dir = os.path.join(output_dir, 'figs')
os.makedirs(fig_dir, exist_ok=True)
param_dir = os.path.join(output_dir, 'params')
os.makedirs(param_dir, exist_ok=True)

# ================================================================== #
# LOAD DATA
# ================================================================== #
print("=" * 60)
print(f"STBS estimation on {data_name}_{addendum}")
print("=" * 60)

tf.random.set_seed(seed)
random_state = np.random.RandomState(seed)

if cli_args.data_dir is not None:
    data_dir = cli_args.data_dir
    print(f"  Using custom data dir: {data_dir}")
else:
    data_dir = os.path.join(STBS_CAVI_DIR, 'data', data_name, 'clean')

(dataset, permutation, all_author_indices, vocabulary, author_map, author_info) = build_input_pipeline(
    data_name, data_dir, batch_size, random_state, None, "nothing", addendum
)
num_documents = len(permutation)
num_words = len(vocabulary)
num_authors = author_info.shape[0]

print(f"Documents: {num_documents}, Words: {num_words}, Authors: {num_authors}")

# Create regression matrix
# ideal_topic_dim is the second dimension of the ideal-point tensor:
#   ideal_dim = "ak" -> num_topics  (per-author per-topic)
#   ideal_dim = "a"  -> 1           (per-author only, TBIP-style)
ideal_topic_dim = num_topics if cli_args.ideal_dim == "ak" else 1
if cli_args.x_override is not None:
    import tensorflow as _tf_for_X
    X_np = np.load(cli_args.x_override).astype(np.float32)
    print(f"  Using X override: {cli_args.x_override}  shape={X_np.shape}")
    X = _tf_for_X.constant(X_np, dtype=_tf_for_X.float32)
    # Initial ideal location: mirror create_X's convention
    #   -1 Democrats, +1 Republicans, 0 Independents (if party info present)
    party_col = None
    if "party" in author_info.columns:
        party_col = author_info["party"]
    if party_col is not None:
        auxloc = (
            -1.0 * (party_col == "D").to_numpy(dtype=np.float32)
            + 1.0 * (party_col == "R").to_numpy(dtype=np.float32)
            + 0.0 * (party_col == "I").to_numpy(dtype=np.float32)
        )
    else:
        auxloc = np.zeros(X_np.shape[0], dtype=np.float32)
    initial_ideal_location = _tf_for_X.repeat(
        _tf_for_X.constant(auxloc)[:, None], ideal_topic_dim, axis=1
    )
else:
    X, initial_ideal_location = create_X(data_name, author_info, covariates, ideal_topic_dim)
print(f"Regression coefficients: {X.shape[1]}")

# Monkey-patch: disable NaN checking — IsFinite op not supported on Metal GPU
STBS.print_non_finite_parameters = lambda self, *args, **kwargs: None
STBS.check_and_print_non_finite = lambda self, *args, **kwargs: None

# ================================================================== #
# STEP 0: INITIALISATION
#   - default: run Poisson Factorization for pf_max_steps
#   - warm-start: load beta/theta shape & rate from a directory
# ================================================================== #
# Load counts (needed for PF; not needed for warm start, but cheap to load)
counts_sparse = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + ".npz"))
counts_dense = np.array(counts_sparse.todense()).astype(np.float64)

if cli_args.warm_start_dir is not None:
    print("\n" + "=" * 60)
    print(f"  Warm-start init from: {cli_args.warm_start_dir}")
    print("=" * 60)

    def _load_array(dirpath, base):
        """Try (in order) <base>_final.npy, <base>.npy, <base>.csv."""
        for candidate in [f"{base}_final.npy", f"{base}.npy"]:
            p = os.path.join(dirpath, candidate)
            if os.path.exists(p):
                return np.load(p)
        p = os.path.join(dirpath, f"{base}.csv")
        if os.path.exists(p):
            import pandas as pd
            return pd.read_csv(p, index_col=0).to_numpy()
        raise FileNotFoundError(
            f"Could not find {base}_final.npy / {base}.npy / {base}.csv "
            f"in {dirpath}"
        )

    def _load_first(dirpath, bases):
        """Try each base name in turn; return the first that loads."""
        last_exc = None
        for base in bases:
            try:
                return _load_array(dirpath, base)
            except FileNotFoundError as e:
                last_exc = e
        raise last_exc

    # Topic-word: beta shape & rate
    beta_shp = _load_first(cli_args.warm_start_dir, ["beta_shape", "beta_shp"])
    beta_rte = _load_first(cli_args.warm_start_dir, ["beta_rate", "beta_rte"])

    # Document-topic: theta shape & rate
    theta_shp = _load_first(cli_args.warm_start_dir, ["theta_shape", "theta_shp"])
    theta_rte = _load_first(cli_args.warm_start_dir, ["theta_rate", "theta_rte"])

    variational_topic_shape = beta_shp.astype(np.float64)
    variational_topic_rate = beta_rte.astype(np.float64)
    variational_document_shape = theta_shp.astype(np.float64)
    variational_document_rate = theta_rte.astype(np.float64)

    # Prior rates for PF hyperparameters -- reasonable defaults
    pf_random_state = np.random.RandomState(pf_seed)
    variational_document_prior_rate = np.exp(
        0.1 * pf_random_state.randn(num_documents)
    )
    variational_topic_prior_rate = np.exp(
        0.1 * pf_random_state.randn(num_words)
    )
    variational_document_prior_shape = 0.3 + num_topics * 0.3
    variational_topic_prior_shape = 0.3 + num_topics * 0.3

    print(f"  beta_shape  : {variational_topic_shape.shape}  "
          f"mean={variational_topic_shape.mean():.4f}")
    print(f"  beta_rate   : {variational_topic_rate.shape}  "
          f"mean={variational_topic_rate.mean():.4f}")
    print(f"  theta_shape : {variational_document_shape.shape}  "
          f"mean={variational_document_shape.mean():.4f}")
    print(f"  theta_rate  : {variational_document_rate.shape}  "
          f"mean={variational_document_rate.mean():.4f}")
    print(f"  (PF initialisation was SKIPPED.)")

else:
    print("\n" + "=" * 60)
    print(f"Running Poisson Factorization for {pf_max_steps} steps...")
    print("=" * 60)

    pf_random_state = np.random.RandomState(pf_seed)
    prior_shape_pf = 0.3
    hyperprior_shape_pf = 0.3
    hyperprior_rate_pf = 0.3

    # Initialize PF variational parameters
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
        normalizer = get_normalizer(variational_topic_shape, variational_topic_rate,
                                    variational_document_shape, variational_document_rate)
        new_doc_shape = update_variational_document_shape(
            counts_dense, variational_topic_shape, variational_topic_rate,
            variational_document_shape, variational_document_rate, normalizer, prior_shape_pf)
        new_doc_rate = update_variational_document_rate(
            variational_topic_shape, variational_topic_rate,
            variational_document_prior_shape, variational_document_prior_rate)
        new_doc_prior_rate = update_variational_document_prior_rate(
            new_doc_shape, new_doc_rate, hyperprior_rate_pf)
        new_topic_shape = update_variational_topic_shape(
            counts_dense, variational_topic_shape, variational_topic_rate,
            variational_document_shape, variational_document_rate, normalizer, prior_shape_pf)
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
            pf_elapsed = time.time() - pf_start
            print(f"  PF step {pf_step:>3d}/{pf_max_steps} ({pf_elapsed:.0f}s)")

# Save PF results
pf_dir = os.path.join(output_dir, 'pf_fits')
os.makedirs(pf_dir, exist_ok=True)
np.save(os.path.join(pf_dir, f"document_shape_K{num_topics}{addendum}"), variational_document_shape)
np.save(os.path.join(pf_dir, f"document_rate_K{num_topics}{addendum}"), variational_document_rate)
np.save(os.path.join(pf_dir, f"topic_shape_K{num_topics}{addendum}"), variational_topic_shape)
np.save(os.path.join(pf_dir, f"topic_rate_K{num_topics}{addendum}"), variational_topic_rate)

# Cast to float32 for TF
fitted_document_shape = variational_document_shape.astype(np.float32)
fitted_document_rate = variational_document_rate.astype(np.float32)
fitted_topic_shape = variational_topic_shape.astype(np.float32)
fitted_topic_rate = variational_topic_rate.astype(np.float32)

del counts_dense  # free memory
if cli_args.warm_start_dir is None:
    pf_time = time.time() - pf_start
    print(f"PF complete in {pf_time:.0f}s ({pf_time/60:.1f}min)")
else:
    print("Warm-start init complete (PF was skipped).")

# ================================================================== #
# MODEL INITIALIZATION (with PF pre-initialization)
# ================================================================== #
optim = tf.optimizers.Adam(learning_rate=learning_rate)

model = STBS(num_documents,
             num_topics,
             num_words,
             num_authors,
             num_samples,
             X,
             all_author_indices,
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

# Optional: warm-start the eta variational location from disk.
# If --warm-start-dir contains `eta_location_final.npy` (or eta_loc.npy /
# eta_loc.csv), assign it to model.eta_varfam.location. Keeps the eta scale
# at its prior init; only the mean is initialized.
if cli_args.warm_start_dir is not None:
    eta_init = None
    for cand in ["eta_location_final.npy", "eta_location.npy",
                 "eta_loc.npy"]:
        p = os.path.join(cli_args.warm_start_dir, cand)
        if os.path.exists(p):
            eta_init = np.load(p)
            print(f"  Loading eta warm-start from: {p}")
            break
    if eta_init is None:
        p = os.path.join(cli_args.warm_start_dir, "eta_loc.csv")
        if os.path.exists(p):
            import pandas as pd
            eta_init = pd.read_csv(p, index_col=0).to_numpy()
            print(f"  Loading eta warm-start from: {p}")
    if eta_init is not None:
        eta_init = eta_init.astype(np.float32)
        if eta_init.shape != tuple(model.eta_varfam.location.shape):
            raise ValueError(
                f"eta warm-start shape {eta_init.shape} != model "
                f"shape {tuple(model.eta_varfam.location.shape)}"
            )
        model.eta_varfam.location.assign(eta_init)
        print(f"  eta_varfam.location warm-started: shape={eta_init.shape}, "
              f"std={float(np.std(eta_init)):.4f}")
    else:
        print("  (no eta warm-start file found in warm-start-dir; "
              "eta will be learned from prior init.)")

# Optional: warm-start the iota variational location from disk.
# If --warm-start-dir contains `iota_location_final.npy` (or iota_loc.csv),
# assign it to model.iota_varfam.location. Keeps the iota scale_tril at
# its prior init; only the per-coefficient mean per topic is initialised.
if cli_args.warm_start_dir is not None:
    iota_init = None
    for cand in ["iota_location_final.npy", "iota_location.npy",
                 "iota_loc.npy"]:
        p = os.path.join(cli_args.warm_start_dir, cand)
        if os.path.exists(p):
            iota_init = np.load(p)
            print(f"  Loading iota warm-start from: {p}")
            break
    if iota_init is None:
        p = os.path.join(cli_args.warm_start_dir, "iota_loc.csv")
        if os.path.exists(p):
            import pandas as pd
            iota_init = pd.read_csv(p, index_col=0).to_numpy()
            print(f"  Loading iota warm-start from: {p}")
    if iota_init is not None:
        iota_init = iota_init.astype(np.float32)
        target_shape = tuple(model.iota_varfam.location.shape)
        if iota_init.shape != target_shape:
            # Misspecified-fit case: GT was simulated with iota_dim='kl' so
            # the warm-start file is (K, J), but the current model uses
            # iota_dim='l' with target shape (1, J). Skip the warm-start
            # rather than fail — the constant-iota model has no analogous
            # truth to load.
            if cli_args.iota_dim == "l" and iota_init.ndim == 2 \
                    and target_shape == (1, iota_init.shape[1]):
                print(f"  Skipping iota warm-start: GT shape "
                      f"{iota_init.shape} incompatible with iota_dim='l' "
                      f"target {target_shape}; using prior init.")
            else:
                raise ValueError(
                    f"iota warm-start shape {iota_init.shape} != model "
                    f"shape {target_shape}"
                )
        else:
            model.iota_varfam.location.assign(iota_init)
            print(f"  iota_varfam.location warm-started: shape={iota_init.shape}, "
                  f"std={float(np.std(iota_init)):.4f}")
    else:
        print("  (no iota warm-start file found in warm-start-dir; "
              "iota will be learned from prior init.)")

print(f"\nModel initialized with PF pre-initialization. Training for {num_epochs} epochs...")

# ================================================================== #
# TRAINING LOOP
# ================================================================== #
_, seed_tf = tfp.random.split_seed(seed)
model_state = {'ELBO': [], 'entropy': [], 'log_prior': [], 'reconstruction': [],
               'epoch': [], 'batch': [], 'step': []}

start_time = time.time()
batches_per_epoch = len(dataset)

for epoch in range(num_epochs):
    for batch_index, batch in enumerate(iter(dataset)):
        step = batches_per_epoch * epoch + batch_index
        inputs, outputs = batch
        (total_loss, reconstruction_loss, log_prior_loss, entropy_loss, seed_tf) = train_step(
            model, inputs, outputs, optim, seed_tf, tf.constant(step))

        model_state['ELBO'].append(-total_loss.numpy())
        model_state['entropy'].append(-entropy_loss.numpy())
        model_state['log_prior'].append(-log_prior_loss.numpy())
        model_state['reconstruction'].append(-reconstruction_loss.numpy())
        model_state['epoch'].append(epoch)
        model_state['batch'].append(batch_index)
        model_state['step'].append(step)

        if step % print_steps == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:>6d} | Epoch {epoch:>3d} | "
                  f"ELBO: {-total_loss.numpy():>12.1f} | "
                  f"Recon: {-reconstruction_loss.numpy():>12.1f} | "
                  f"({elapsed:.0f}s elapsed)")

    # End of epoch summary
    sec_per_epoch = (time.time() - start_time) / (epoch + 1)
    print(f"Epoch {epoch:>3d} done | ELBO: {-total_loss.numpy():>12.1f} | "
          f"({sec_per_epoch:.1f} sec/epoch)")

    # Save intermediate results
    if (epoch + 1) % save_every == 0:
        np.save(os.path.join(param_dir, f"ideal_point_location_epoch{epoch}"),
                model.ideal_varfam.location.numpy())
        np.save(os.path.join(param_dir, f"ideal_point_scale_epoch{epoch}"),
                model.ideal_varfam.scale.numpy())
        print(f"  -> Saved checkpoint at epoch {epoch}")

total_time = time.time() - start_time
print(f"\nTraining complete! Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

# ================================================================== #
# SAVE FINAL RESULTS
# ================================================================== #
print("\nSaving results...")

# Ideal points: [num_authors, num_topics]
ideal_loc = model.ideal_varfam.location.numpy()
ideal_scl = model.ideal_varfam.scale.numpy()
np.save(os.path.join(param_dir, "ideal_point_location_final"), ideal_loc)
np.save(os.path.join(param_dir, "ideal_point_scale_final"), ideal_scl)

# Save as CSV with author names. Column count adapts to ideal_dim:
#   "ak" -> 25 columns topic_0..topic_24
#   "a"  -> 1 column  ideal
ip_cols = ([f"topic_{k}" for k in range(num_topics)]
           if cli_args.ideal_dim == "ak" else ["ideal"])
ip_df = pd.DataFrame(ideal_loc, columns=ip_cols)
ip_df.insert(0, "author", author_map)
ip_df.to_csv(os.path.join(output_dir, "ideal_points.csv"), index=False)

# Iota (regression coefficients): [num_topics, num_coef] or [1, num_coef]
if hasattr(model.iota_varfam, 'location'):
    iota_loc = model.iota_varfam.location.numpy()
    np.save(os.path.join(param_dir, "iota_location_final"), iota_loc)
    if hasattr(model.iota_varfam, 'scale'):
        iota_scl = model.iota_varfam.scale.numpy()
        np.save(os.path.join(param_dir, "iota_scale_final"), iota_scl)
    if hasattr(model.iota_varfam, 'scale_tril'):
        iota_scale_tril = model.iota_varfam.scale_tril.numpy()
        np.save(os.path.join(param_dir, "iota_scale_tril_final"), iota_scale_tril)
        print(f"  iota_scale_tril saved: shape {iota_scale_tril.shape}")

# ---- Export CSVs for R scripts ----
csv_dir = param_dir
pd.DataFrame(iota_loc).to_csv(os.path.join(csv_dir, "iota_loc.csv"), index=False, header=False)
pd.DataFrame(ideal_loc).to_csv(os.path.join(csv_dir, "ideal_loc.csv"), index=False, header=False)
pd.DataFrame(ideal_scl).to_csv(os.path.join(csv_dir, "ideal_scl.csv"), index=False, header=False)
if hasattr(model.iota_varfam, 'scale_tril'):
    pd.DataFrame(iota_scale_tril).to_csv(os.path.join(csv_dir, "iota_scale_tril.csv"),
                                          index=False, header=False)
print("CSVs for R scripts exported.")

# Eta (polarity loadings): [num_topics, num_words]
eta_loc = model.eta_varfam.location.numpy()
eta_scl = model.eta_varfam.scale.numpy()
np.save(os.path.join(param_dir, "eta_location_final"), eta_loc)
np.save(os.path.join(param_dir, "eta_scale_final"), eta_scl)
pd.DataFrame(eta_loc).to_csv(os.path.join(param_dir, "eta_loc.csv"), index=False, header=False)

# Beta (topic-word distributions): [num_topics, num_words]
beta_shp = model.beta_varfam.shape.numpy()
beta_rte = model.beta_varfam.rate.numpy()
np.save(os.path.join(param_dir, "beta_shape_final"), beta_shp)
np.save(os.path.join(param_dir, "beta_rate_final"), beta_rte)

# Theta (document-topic intensities): [num_documents, num_topics]
theta_shp = model.theta_varfam.shape.numpy()
theta_rte = model.theta_varfam.rate.numpy()
np.save(os.path.join(param_dir, "theta_shape_final"), theta_shp)
np.save(os.path.join(param_dir, "theta_rate_final"), theta_rte)

# Training loss
model_state_df = pd.DataFrame(model_state)
model_state_df.to_csv(os.path.join(output_dir, "training_loss.csv"), index=False)

# ================================================================== #
# PLOTS
# ================================================================== #

# ELBO over steps
plt.figure(figsize=(10, 4))
plt.plot(model_state_df['step'], model_state_df['ELBO'], alpha=0.5, linewidth=0.5)
plt.xlabel('Step')
plt.ylabel('ELBO')
plt.title('ELBO during training')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'elbo_training.png'), dpi=150)
plt.close()

# Average ELBO per epoch
avg_elbo = model_state_df.groupby('epoch')['ELBO'].mean()
plt.figure(figsize=(10, 4))
plt.plot(avg_elbo.index, avg_elbo.values)
plt.xlabel('Epoch')
plt.ylabel('Average ELBO')
plt.title('Average ELBO per epoch')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'avg_elbo_per_epoch.png'), dpi=150)
plt.close()

print(f"\nResults saved to: {output_dir}")
print(f"Ideal points shape: {ideal_loc.shape}")
print(f"Iota shape: {iota_loc.shape if hasattr(model.iota_varfam, 'location') else 'N/A'}")
print("Done!")
