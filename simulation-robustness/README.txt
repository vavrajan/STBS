Revision_code_CAVI — Overview, Scripts and Folder Structure
=============================================================

This directory contains estimation scripts using the original TensorFlow/CAVI
implementation of the STBS model (from ../STBS_CAVI/).

==============================================================================
TABLE OF CONTENTS
==============================================================================
  1. Scripts (numbered pipeline)
  2. Folder structure
  3. Virtual Environment / dependencies
  4. Running the pipeline
  5. Metal GPU compatibility fixes
  6. Performance notes


==============================================================================
1. SCRIPTS
==============================================================================

The scripts are numbered in the order they are typically used.

01_estimate_STBS.py
    Core estimation driver. Takes CLI flags:
        --seed            random seed (default 314159)
        --num-epochs      number of CAVI epochs (default 300)
        --num-topics      K (default 25)
        --output-dir      where to save results (default auto-named)
        --hp-overrides    JSON overriding prior_hyperparameter entries
        --data-dir        override the data directory (e.g., to run on
                          a simulated DTM in data_simulation/sim_XX/clean)
        --x-override      path to a .npy covariate matrix (N, L) that
                          bypasses create_X_hein_daily (used for simulations)
        --warm-start-dir  path to a directory with beta_shape/rate and
                          theta_shape/rate files. Skips the Poisson-
                          Factorisation init and starts CAVI at those values.
                          Accepts both .npy (beta_shape_final, beta_rate_final)
                          and .csv (beta_shp, beta_rte) naming conventions.
    Writes params/*.npy, figs/*.png and tabs/*.csv to the output dir.

02_run_multi_seeds.py
    Runs 01_estimate_STBS.py SEQUENTIALLY for three seeds
    (314159, 42, 123456) with K=25, 300 epochs. After each run it also
    launches the R plots (05) and word clouds (03). Used for seed-
    robustness analysis.

02c_multi_K.py
    Runs 01_estimate_STBS.py sequentially for K in {15, 20, 30} with
    seed 123456. (K=25 is the baseline that already exists from 02_.)
    Same post-processing as 02_run_multi_seeds.py.

02d_simulation_hyperparameters.py
    Hyperparameter sensitivity analysis. Varies 5 groups
    (omega, beta, rho, theta, ideal) at two deviating values
    {0.1, 1.0} from the baseline 0.3, one group at a time.
    Produces hp_<group>_<value>_K25/ output dirs + an analysis CSV and
    plot in stbs_cavi_results/figs/.
    Sub-commands:   all | omega | beta | rho | theta | ideal | analyze

03_generate_wordclouds.py
    Generates plain wordclouds for each of the K topics in a given result
    directory using E[beta] = shape/rate as word probabilities.
    Saves PDF + PNG to <dir>/figs/wordcloud_k_*.{pdf,png}. Labels are
    data-driven (top-5 words).

03b_generate_wordclouds_logscale.py
    Generates the six-panel log-scale wordclouds in the style of
    plot_wordclouds_slides (used by the paper figures). Per topic k:
        top row    :  E[log beta] - eta   |   E[log beta]   |   E[log beta] + eta
        bottom row :  -eta (beta > thr)    |   Topic k       |   +eta (beta > thr)
    Uses E[log beta] = digamma(shape) - log(rate). Saves
    <dir>/figs/wordcloud_logscale_k_*.png.
    Usage:  python3 03b_generate_wordclouds_logscale.py --results-dir <dir>

04_compare_simulations_DW.py
    Compares theta-weighted aggregated ideal points across the three
    seeds (314159, 42, 123456) against DW-NOMINATE first-dimension
    scores for the 114th Senate. Writes comparison_dw_nominate.csv
    and correlation_summary.csv to stbs_cavi_results/.

04b_compare_topics_across_seeds.py
    Solves the label-switching problem across the 3 seeds via the
    Hungarian algorithm on the cosine-similarity matrix of topic-word
    distributions (beta). Reports per-topic comparisons (beta cosine,
    eta correlation, iota correlation, eta*ideal correlation) and
    writes CSVs + figures to stbs_cavi_results/topic_comparison/.

04c_compare_topics_across_K.py
    Same idea but across the FOUR K values (15, 20, 25, 30), seed fixed
    at 123456. Uses a (possibly rectangular) Hungarian matching for the
    similarity matrix. Also reports DW-NOMINATE correlation per K.
    Output: stbs_cavi_results/K_comparison/.

05_run_R_plots.R
    R post-processing script that generates the paper's diagnostic plots
    (ideal points by party, eta effects, topic heatmaps, etc.) for one
    result directory. Called automatically by 02_run_multi_seeds.py and
    02c_multi_K.py. Standalone use:
        Rscript 05_run_R_plots.R <result_dir>

06_simulate_DTM.py
    Large simulation study. Loads ORIGINAL paper fit (beta, theta, eta)
    and rewrites iota so that HALF of the K topics have iota = 0 and
    the other HALF have iota = 3 x iota_original. Draws 20 Monte-Carlo
    DTMs from the Poisson generative model (same ground truth, different
    Poisson seeds per replication). Saves each DTM in a drop-in layout
    so STBS can re-estimate on it (counts114.npz + copied author/vocab
    files). Output: data_simulation/ground_truth/ + sim_01/ ... sim_20/.

06b_simulate_small.py
    Small simulation study with ONLY 6 covariates (no interactions):
        c0 Party_R (active on all 25 topics),
        c1 Gender_F (active on 12 topics),
        c2 Region_South (active on 6 topics),
        c3, c4, c5 inactive (iota = 0).
    iota magnitudes drawn uniform in [0.5, 1.0] with random signs.
    Produces a single DTM + ground truth at
    data_simulation/sim_00_6covariates/ plus a (99, 6) X_override.npy
    that STBS can use via --x-override.

07_compare_simulation.py
    After STBS fits a simulated DTM, solves label-switching via
    Hungarian matching on beta cosine, sign-aligns topics, and computes
    per-topic recovery metrics (beta_cos, cor(eta), cor(iota),
    cor(ideal), eta*ideal corr). Writes
        results_simulation/<sim>/recovery_table.csv
        docs/sim_results_table_<sim>.tex
    Usage:  python3 07_compare_simulation.py --sim sim_01


==============================================================================
2. FOLDER STRUCTURE
==============================================================================

Revision_code_CAVI/
├── README.txt                   (this file)
│
├── 01_estimate_STBS.py          ... (see Section 1)
├── 02_run_multi_seeds.py
├── 02c_multi_K.py
├── 02d_simulation_hyperparameters.py
├── 03_generate_wordclouds.py
├── 03b_generate_wordclouds_logscale.py
├── 04_compare_simulations_DW.py
├── 04b_compare_topics_across_seeds.py
├── 04c_compare_topics_across_K.py
├── 05_run_R_plots.R
├── 06_simulate_DTM.py
├── 06b_simulate_small.py
├── 07_compare_simulation.py
│
├── docs/                        LaTeX write-ups of the studies
│   ├── seed_comparison.tex           ─── Seed sensitivity (3 seeds, K=25)
│   │                                     + comparison with original paper fit
│   ├── K_comparison.tex              ─── K sensitivity (K=15,20,25,30)
│   ├── hyperparameter_comparison.tex ─── Prior hyperparameter sensitivity
│   ├── simulation.tex                ─── Simulation study (20 DTMs, 3x iota)
│   └── sim_results_table_*.tex       ─── auto-generated recovery tables
│
├── stbs_cavi_results/           Outputs of the estimation pipeline
│   └── (see stbs_cavi_results/README.txt for the full layout)
│
├── data_simulation/             Simulated data produced by 06_simulate_DTM.py
│   ├── ground_truth/                 shared across sim_01..sim_20
│   │   ├── iota_sim.npy              (K, L) modified iota
│   │   ├── iota_orig.npy             (K, L) original for reference
│   │   ├── iota_zero_topics.npy      indices with iota = 0
│   │   ├── iota_strong_topics.npy    indices with 3x iota
│   │   ├── ideal_points_sim.npy      (N, K) ground truth ideal points
│   │   ├── X.npy                     covariate matrix
│   │   ├── simulation_meta.json
│   │   └── figs/ideal_points_hist.png
│   │
│   ├── sim_01/  sim_02/  ...  sim_20/   one per MC replication (20 DTMs)
│   │   ├── clean/
│   │   │   ├── counts114.npz         simulated DTM (drop-in for STBS)
│   │   │   ├── author_indices114.npy (copied)
│   │   │   ├── author_map114.txt     (copied)
│   │   │   ├── author_info114.csv    (copied)
│   │   │   ├── author_detailed_info114.csv
│   │   │   ├── author_detailed_info_with_religion114.csv
│   │   │   ├── vocabulary114.txt
│   │   │   └── speech_id_indices114.npy
│   │   ├── sim_meta.json             Poisson seed & DTM stats
│   │   └── figs/counts_compare.png
│   │
│   ├── sim_00_6covariates/           small simulation (06b script)
│   │   ├── clean/                    STBS input (6-column X_override.npy
│   │   │                             and counts114.npz)
│   │   └── ground_truth/             iota_sim (25,6), effect_pattern,
│   │                                 ideal_points_sim, X, labels, meta
│   │
│   ├── simulation_index.csv          summary of all 20 MC sims
│   └── simulate_DTM.log
│
└── results_simulation/          STBS re-fits on the simulated data
    ├── sim_01/                         51-cov fit on first MC replication
    ├── sim_00_6covariates/             6-cov fit on small simulation
    │   └── recovery_table.csv          (from 07_compare_simulation.py)
    └── sim_00_6covariates_warm/        6-cov fit with --warm-start-dir
                                        (init from originalPolAn_results)


==============================================================================
3. VIRTUAL ENVIRONMENT
==============================================================================

Location: ../STBS_CAVI/venv_gpu/
Python:   3.11.6

Created with:
    python3.11 -m venv /path/to/STBS_CAVI/venv_gpu
    ./venv_gpu/bin/python3 -m pip install --upgrade pip

Core packages:
    tensorflow==2.16.2
    tensorflow-metal==1.1.0       # Apple Silicon GPU support
    tensorflow-probability==0.24.0
    tf_keras==2.16.0              # Required by TFP 0.24

Utilities:
    numpy==1.26.4
    scipy==1.17.1
    pandas==3.0.2
    matplotlib==3.10.8
    seaborn==0.13.2
    scikit-learn==1.8.0
    wordcloud==1.9.6

Full install command:
    pip install tensorflow==2.16.2 tensorflow-metal==1.1.0 \
                tensorflow-probability==0.24.0 tf_keras==2.16.0 \
                numpy scipy pandas matplotlib seaborn scikit-learn wordcloud

IMPORTANT: Pin tf_keras to 2.16.0! Newer versions pull in TF 2.21 which
breaks tensorflow-metal compatibility.

GPU support
-----------
- tensorflow-metal==1.1.0 enables Apple Metal GPU acceleration on Apple Silicon
- Verified: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
- NOTE: tensorflow-metal is NOT compatible with TF >= 2.17.
  Do NOT upgrade TensorFlow without checking metal compatibility.


==============================================================================
4. RUNNING THE PIPELINE
==============================================================================

All Python scripts must be launched from the STBS_CAVI/ directory so that
sys.path.insert('code') finds the STBS source files.

    cd /path/to/STBS_CAVI
    TF_USE_LEGACY_KERAS=1 ./venv_gpu/bin/python3 -u \
        ../Revision_code_CAVI/01_estimate_STBS.py

    # Or with logging (nohup so it survives terminal closing):
    TF_USE_LEGACY_KERAS=1 nohup ./venv_gpu/bin/python3 -u \
        ../Revision_code_CAVI/02_run_multi_seeds.py \
        >> ../Revision_code_CAVI/stbs_cavi_results/multi_seed.log 2>&1 &

    # Monitor progress:
    tail -f /path/to/Revision_code_CAVI/stbs_cavi_results/multi_seed.log

    # Keep Mac from sleeping during long runs (tie to master PID):
    caffeinate -i -w <PID> &


-------- Simulation workflow (06 -> 01 -> 07) --------

    # 1. Build the 20 MC DTMs (or the small 6-covariate simulation):
    ./venv_gpu/bin/python3 ../Revision_code_CAVI/06_simulate_DTM.py
    # or
    ./venv_gpu/bin/python3 ../Revision_code_CAVI/06b_simulate_small.py

    # 2. Fit STBS on one of the simulated DTMs (example: sim_00_6covariates):
    TF_USE_LEGACY_KERAS=1 ./venv_gpu/bin/python3 -u \
        ../Revision_code_CAVI/01_estimate_STBS.py \
        --seed 314159 --num-epochs 300 --num-topics 25 \
        --data-dir   .../data_simulation/sim_00_6covariates/clean \
        --x-override .../data_simulation/sim_00_6covariates/clean/X_override.npy \
        --warm-start-dir .../originalPolAn_results/fits/.../params   \
        --output-dir .../results_simulation/sim_00_6covariates_warm
        # --warm-start-dir is optional; with it CAVI skips the PF init
        # and starts at the provided beta/theta (eliminates label-switching
        # and scale-split confound).

    # 3. Compute the recovery table (iota / ideal / beta_cos per topic):
    ./venv_gpu/bin/python3 ../Revision_code_CAVI/07_compare_simulation.py \
        --sim sim_01
    # writes results_simulation/<sim>/recovery_table.csv and
    # docs/sim_results_table_<sim>.tex


==============================================================================
5. METAL GPU COMPATIBILITY FIXES
==============================================================================

The original STBS code was written for TF 2.x / Keras 2. Running on
TF 2.16 + Metal GPU requires these workarounds (already applied in
01_estimate_STBS.py):

1. TF_USE_LEGACY_KERAS=1 — TFP 0.24 requires Keras 2 API (tf_keras).
   Without this, Keras 3 rejects int positional args in model.__call__.

2. Monkey-patched print_non_finite_parameters / check_and_print_non_finite
   to no-ops — these use tf.math.is_finite (IsFinite op) which the Metal
   GPU plugin does not support.

3. Custom train_step() defined locally — passes nsamples as keyword arg
   (Keras 3 compatibility) and skips the NaN check call.


==============================================================================
6. PERFORMANCE NOTES
==============================================================================

On Apple M4 Pro (48 GB):
- PF initialization (300 steps): ~2.5 min (NumPy/CPU, all cores)
- STBS training: ~6 sec/epoch on Metal GPU at K=25
- 300 epochs total for K=25: ~45-75 min
- K=15/20: slightly faster, K=30: slightly slower

Memory
------
Each run uses ~3-5 GB RAM actively. Running ONE run at a time is
required — parallel runs on the same GPU were ~9x slower per epoch.
On machines with <32 GB RAM, close browsers and other memory-heavy
apps before long runs; segmentation faults can occur when the system
swap runs out.

Important
---------
- This venv (venv_gpu) is separate from the old venv_stbs (TF 2.21, no GPU).
- Do NOT install JAX in this venv — JAX 0.9.x requires numpy>=2.0 and
  ml_dtypes>=0.5.0, which conflict with TF 2.16's numpy==1.26 and ml_dtypes==0.3.
- For JAX-based code (poisson_topicmodels), use the separate environment in
  ../Revision_code/.
