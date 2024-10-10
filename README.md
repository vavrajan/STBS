# Structural Text-Based Scaling Model
Source code for the paper: 
[Structural Text-Based Scaling Model by Jan Vávra, Bernd Prostmeier, Bettina Grün, and Paul Hofmarcher (2024)](web to the paper).

## Directories and main files overview:

* `analysis` - contains the scripts for performing the estimation and post-processing
  * `analysis_cluster` - estimates STBS model on computational cluster 
  * `hein_daily_laptop` - py script that follows an estimation of STBS on hein-daily data
* `code` - source code `.py` files for STBS estimation 
  * `poisson_factorization` - run first for initialization
  * `stbs` - the main file containing the definition of the model
  * ...
* `create_slurms` - `.py` files to create `.slurm` files for submitting jobs on computational cluster
* `data` - contains data in separate folders
  * `hein-daily` - contains data from Hein-Daily (here only session 114)
    * `clean` - string '114' is an *addendum* that is added to the end of the file name to specify a different version of your dataset such as different session or differently pre-processed data
      * `author_(detailed_)info_...114.csv` - author-specific covariates
      * `author_indices114.npy` - a vector of the same length as the number of documents, contains indices of authors of documents
      * `author_map114.npy` - vector of author names + parties "Jan Vávra (I)"
      * `counts114.npz` - document-term matrix in sparse format
      * `vocabulary114.txt` - each row corresponds to one of the terms in vocabulary
    * `pf-fits` - initial values created by `poisson_factorization` initial values, `--` abbreviates `str(num_topics) + addendum`
      * `document_shape_K--.npy`, `document_rate_K--.npy` - initial shapes and rates for thetas
      * `topic_shape_K--.npy`, `topic_rate_K--.npy` - inital shapes and rates for betas
    * `fits`, `figs`, `txts`, `tabs` - directories for STBS estimated parameters and checkpoints, figures, text files (influential speeches) and tables. Contains directories for specific model settings - defined by the `name` in `create_slurms` files.
* `err` - directory for error files, properly structured
* `out` - directory for output files, properly structured
* `R` - `.R` files for creating plots using estimated values
  * `.R` - creates the regression summary plots
* `slurm` - directory for `.slurm` files that submit jobs on cluster, properly structured

## Adding a new dataset

First, create a new subdirectory in `data` named by `your_data_name` and add all the necessary folders. 
You can follow the same format of the data in `clean`, then you only need to replace `data == 'hein-daily'` 
with `data_name in ['hein-daily', 'your_data_name']` in `input_pipeline`. 
This will expect the same input files as for `hein-daily`.

There are other `data_name` sensitive functions in `code`: 
`create_X`, `plotting_functions`, `influential_speeches`. 
In case you want to have the data stored differently, create your own model matrix, 
create your own plots or
define your own way how to find the most influential speeches
then you need to write your own function and add it to the corresponding wrapper.
You can find some examples of these tweaks already implemented for other than `hein-daily` dataset.

## Model definition

The implementation is very flexible and allow for many different models to be fitted. 
With this source code you can fit the original TBIP model without any regression behind 
fixed ideological positions
as well as
our STBS model with topic-specific ideological positions and as elaborate prior distribution 
as you wish. The paper presents the most complex setting, so with this implementation you can only simplify it, 
but not suppose something even more comlicated. That is left for you to implement. :) 

### The choice of the prior distribution

There are two important inputs to STBS that define the structural choice of the estimated model.
* `prior_choice` - a dictionary that defines which model parameters are present and their dimensionality,
* `prior_hyperparameter` -  a dictionary containing the fixed values of hyperparameters of prior distributions.

Both can be defined from `FLAGS` argument with functions in `check_prior.py`. 
The choices and their meanings are all enumerated in details in `analysis_cluster.py`. 
Note that some of the choices are mutually exclusive. 
`check_prior.py` warns you about such inappropriate choices. 

Let's explain it on several examples. First, these settings give you the original TBIP:
```{python, eval=False}
prior_choice = {
        "theta": "Gfix",          # Gamma with fixed 'theta_shp' and 'theta_rte' 
        "exp_verbosity": "LNfix", # Log-normal with fixed 'exp_verbosity_loc' and 'exp_verbosity_scl' 
        "beta": "Gfix",           # Gamma with fixed 'beta_shp' and 'beta_rte' 
        "eta": "Nfix",            # Normal with fixed 'eta_loc' and 'eta_scl' 
        "ideal_dim": "a",         # ideal points have only author dimension (ideal points fixed for all topics)
        "ideal_mean": "Nfix",     # Normal with fixed 'ideal_loc' --> no iota at all
        "ideal_prec": "Nfix,      # Normal with fixed 'ideal_scl' 
        "iota_dim": "l",          # ... irrelevant, iota (regression coefficients) do not exist
        "iota_mean": "None",      # ... irrelevant
        "iota_prec": "Nfix",      # ... irrelevant
    }
```

Now STBS model with regression but still fixed ideological positions across all topics:
```{python, eval=False}
prior_choice = {
        "theta": "Garte",         # Gamma with fixed 'theta_shp' and flexible author-specific rates
        "exp_verbosity": "None",  # will not exist in model, covered by theta_rate parameter instead (Gamma with fix values)
        "beta": "Gvrte",          # Gamma with fixed 'beta_shp' and flexible word-specific rates
        "eta": "NkprecF",         # Normal with topic-specific Fisher-Snedecor distributed precision (= triple gamma prior with eta_prec, eta_prec_rate)
        "ideal_dim": "a",         # ideal points have only author dimension (ideal points fixed for all topics)
        "ideal_mean": "Nreg",     # Normal with location determined by regression
        "ideal_prec": "Nprec,     # Normal with unknown precision common to all authors 
        "iota_dim": "l",          # regression coefficients (=iota) dimension only specific to covariates (cannot be to topics since ideal is not topic-specific) 
        "iota_mean": "None",      # iota will be apriori Normal with fixed mean to 'iota_loc'
        "iota_prec": "NlprecG",   # iota will be apriori Normal with coefficient-specific precision with apriori fixed Gamma
    }
```

Now STBS model with regression in its full complexity presented in the paper:
```{python, eval=False}
prior_choice = {
        "theta": "Garte",         # Gamma with fixed 'theta_shp' and flexible author-specific rates
        "exp_verbosity": "None",  # will not exist in model, covered by theta_rate parameter instead (Gamma with fix values)
        "beta": "Gvrte",          # Gamma with fixed 'beta_shp' and flexible word-specific rates
        "eta": "NkprecF",         # Normal with topic-specific Fisher-Snedecor distributed precision (= triple gamma prior with eta_prec, eta_prec_rate)
        "ideal_dim": "ak",        # ideal points have author and topic dimension (topic-specific ideal points)
        "ideal_mean": "Nreg",     # Normal with location determined by regression
        "ideal_prec": "Naprec,    # Normal with unknown precision specific to each author
        "iota_dim": "lk",         # regression coefficients (=iota) specific to each covariate and topic
        "iota_mean": "Nlmean",    # iota ~ Normal with flexible iota_mean parameter with Normal prior with fixed values
        "iota_prec": "NlprecF",   # iota ~ Normal with coefficient-specific precision that follows triple Gamma prior - iota_prec, iota_prec_rate
    }
```

### Other model tuning parameters

Moreover, there are other parameters that define the way STBS is estimated.

* `batch_size`: The batch size.
* `RobMon_exponent`: Exponent in [-1, -0.5) satisfying Robbins-Monroe condition to 
create convex-combinations of old and a new value.
* `exact_entropy`: Should we compute the exact entropy (True) 
or approximate it with Monte Carlo (False)?
* `geom_approx`: Should we use the geometric mean approximation (True) 
or exact computation (False) for the expected ideological term Edkv?
* `aux_prob_sparse`: Should we work with counts and auxiliary proportions 
as with sparse matrices (True/False)? (From experience, strangely, sparse format does not lead to faster computation.)
* `iota_coef_jointly`: Should we suppose joint variational family over iota coefficients (True) 
instead of mean-field variational family which imposes independence between coefficients (False)?

## Pre-processing speech data

Follow the steps of Keyon Vafa 
[Text-Based Ideal Points (2020)](https://github.com/keyonvafa/tbip).

## Post-processing the results

### Model comparisons

### Regression summary plots using R
