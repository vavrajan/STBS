# Structural Text-Based Scaling Model
Source code for the paper: 
[Structural Text-Based Scaling Model for Analyzing Political Discourse by Jan Vávra, Bernd Prostmeier, Bettina Grün, and Paul Hofmarcher (2024)](http://arxiv.org/abs/2410.11897).

## Directories and main files overview:

* [analysis](analysis) - 
contains the scripts for performing the estimation and post-processing
  * [preprocess_speeches](analysis/preprocess_speeches.py) - 
  load `hein-daily` original texts (possibly all sessions 97-114), 
  process them with 
  [CountVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
  and save into [data/hein-daily/clean](data/hein-daily/clean) 
  * [author_info_congress_data_and_religion](analysis/author_info_congress_data_and_religion.py) - 
  merge preprocessed [author_info](data/hein-daily/clean/author_info114.csv) 
  with [data_aging_congress](data/hein-daily/orig/data_aging_congress.csv)
  and for session 114 with [data_religion_114](data/hein-daily/orig/data_religion_114.csv)
  * [estimate_STBS_cluster](analysis/estimate_STBS_cluster.py) - 
  estimates STBS model on computational cluster 
  * [compare_TBIP_with_STBS](analysis/compare_TBIP_with_STBS.py), 
  [compare_variability_of_ideal_term.](analysis/compare_variability_of_ideal_term.py) 
  compare two STBS outputs in terms of ideological positions and variability in ideological space
  * [hein_daily_laptop](analysis/hein_daily_laptop.py) - 
  an old py script that follows an estimation of STBS on `hein-daily` data
* [code](code) - source code `.py` files for STBS estimation 
  * [poisson_factorization](code/poisson_factorization.py) - run first for initialization
  * [stbs](code/stbs.py) - the main file containing the definition of the STBS model
  * [check_prior](code/check_prior.py), 
  [input_pipeline](code/input_pipeline.py) and 
  [create_X](code/create_X.py) prepare inputs for STBS model
  * [train_step](code/train_step.py), 
  [information_criteria](code/information_criteria.py), 
  [utils](code/utils.py) and 
  [var_and_prior_families](code/var_and_prior_family.py) used for estimation of STBS
  * [plotting_functions](code/plotting_functions.py) contains functions to create descriptive plots using the latest STBS model parameter values 
    * `create_all_general_descriptive_figures` - for any dataset (histograms, barplots, wordclouds, ...)
    * `create_all_figures_specific_to_data` - specific to each dataset (Republicans vs Democrats, ...)
* [create_slurms](create_slurms) - `.py` files to create `.slurm` files for submitting jobs on computational cluster,
these files are specific to the computing environment used and are included for documentation (and inspiration) purposes 
* [data](data) - contains data in separate folders
  * [hein-daily](data/hein-daily) - contains data from [Hein-Daily](https://data.stanford.edu/congress_text) (here only session 114)
    * [orig](data/hein-daily/orig) - original `hein-daily` data for session 114
      * [stopwords](data/hein-daily/orig/stopwords.txt) - 
      list of stopwords used to process the speeches
      * `114_SpeakerMap.txt`, `byparty_2gram_114.txt`, `byspeaker_2gram_114.txt`, `descr_114.txt`, `speeches_114.txt` - 
      data from [Hein-Daily](https://data.stanford.edu/congress_text) (not here on GitHub)
      * [data_aging_congress.csv](data/hein-daily/orig/data_aging_congress.csv) - 
      congress demographics data 
      [Congress today is older than it’s ever been by Skelley G. (2023)](https://fivethirtyeight.com/features/aging-congress-boomers/)
      * [data_religion_114.csv](data/hein-daily/orig/data_religion_114.csv) - 
      religion data for session 114 only 
      [from Pew Research Center](https://www.pewresearch.org/religion/2015/01/05/members-of-congress-religious-affiliations/)
    * [clean](data/hein-daily/clean) - string '114' is an *addendum* that is added to the end of the file name 
    to specify a different version of your dataset such as different session or differently pre-processed data
      * `author_(detailed_)info_...114.csv` - author-specific covariates
      * [author_indices114](data/hein-daily/clean/author_indices114.npy) - 
      a vector of the same length as the number of documents, contains indices of authors of documents
      * [author_map114](data/hein-daily/clean/author_map114.txt) - 
      vector of author names + parties "Jan Vávra (I)"
      * [counts114](data/hein-daily/clean/counts114.npz) - 
      document-term matrix in sparse format
      * [vocabulary114](data/hein-daily/clean/vocabulary114.txt) - 
      each row corresponds to one of the terms in vocabulary
    * [pf-fits](data/hein-daily/pf-fits) - initial values created by 
    [poisson_factorization](code/poisson_factorization.py) initial values, `--` abbreviates `str(num_topics) + addendum`
      * `document_shape_K--.npy`, `document_rate_K--.npy` - initial shapes and rates for thetas
      * `topic_shape_K--.npy`, `topic_rate_K--.npy` - initial shapes and rates for betas
    * [fits](data/hein-daily/fits), 
    [figs](data/hein-daily/figs), 
    [txts](data/hein-daily/txts), 
    [tabs](data/hein-daily/tabs) - directories for STBS estimated parameters and checkpoints, 
    figures, text files (influential speeches) and tables. 
    Contains directories for specific model settings - 
    defined by the `name` in [create_slurms](create_slurms) files.
* [err](err) - directory for error files, properly structured
* [out](out) - directory for output files, properly structured
* [R](R) - `.R` files for creating plots and tables using R environment
  * [plot_reg_coefs](R/plot_reg_coefs.R) - 
  creates the regression summary plots for STBS model with party-specific effects of other covariates
  * [plot_reg_coefs_interactions](R/plot_reg_coefs_interactions.R) - 
  creates the regression summary plots for STBS model with additive regression formula
  * Similarly, [table_reg_coefs](R/table_reg_coefs.R) and [table_reg_coefs_interactions](R/table_reg_coefs_interactions.R) - 
  create `.tex` files containing the table of regression coefficients
  * [barplot_eta_ideal_variability](R/barplot_eta_ideal_variability.R) - 
  compare the variability of the ideological space between two models via barplot
  * [ideal_party_R2](R/ideal_party_R2.R) -
  compute the R^2 measure of explained variability by party for ideal point estimates
* [slurm](slurm) - directory for `.slurm` files that submit jobs on cluster (very specific to the gpu cluster used for computations, 
user needs to adjust these), properly structured

## Adding a new dataset

First, create a new subdirectory in [data](data) named by `your_data_name` and add all the necessary folders. 
You can supply the same format of the data as in [data/hein-daily/clean](data/hein-daily/clean) 
created analogously to [preprocess_speeches](analysis/preprocess_speeches.py), 
then you only need to replace `data == 'hein-daily'` 
with `data_name in ['hein-daily', 'your_data_name']` in [input_pipeline](code/input_pipeline.py). 
This will expect the same input files as for `hein-daily`.

There are other `data_name` sensitive functions in [code](code): 
[create_X](code/create_X.py), 
[plotting_functions](code/plotting_functions.py), 
[influential_speeches](code/influential_speeches.py). 
In case you want to have the data stored differently, create your own model matrix, 
create your own plots or
define your own way how to find the most influential speeches
then you need to write your own function and add it to the corresponding wrapper.
You can find some examples of these tweaks already implemented for other than `hein-daily` dataset.

## Pre-processing speech data

We follow the steps of Keyon Vafa 
[Text-Based Ideal Points (2020)](https://github.com/keyonvafa/tbip). 
These steps are performed in [preprocess_speeches](analysis/preprocess_speeches.py) in the following order for each session separately:
1. Load and merge speeches with descriptions.
2. Select only speeches by senators given in Senate.
3. Remove senators who make less than 24 speeches.
4. Create mapping between names and IDs and create a data frame of author-specific covariates.
5. Use [CountVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
from [scikit-learn](https://scikit-learn.org/1.5/index.html) 
library to eliminate stopwords, select n-gram range (here bigrams only)
and set the minimal and maximal word-in-speech-appearance frequencies (0.001 and 0.3)
6. Eliminate bigrams spoken by less than 10 Senators.
7. Recall [CountVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
with the shortened vocabulary.
8. Remove empty speeches without any bigram included (row sums are zero).
9. Save sparse `counts.npz`, auxiliary indices and final vocabulary to [data/hein-daily/clean](data/hein-daily/clean).

## Model definition

The implementation is very flexible and allows for many models to be fitted. 
With this source code you can fit the original TBIP model with Gamma variational families 
without any regression behind fixed ideological positions
as well as
our STBS model with topic-specific ideological positions and as elaborate prior distribution 
as you wish. The paper presents the most complex setting, so with this implementation you can only simplify it, 
but not suppose something even more complicated. That is left for you to implement. :) 

### The choice of the prior distribution

There are two important inputs to STBS that define the structural choice of the estimated model.
* `prior_choice` - a dictionary that defines which model parameters are present and their dimensionality,
* `prior_hyperparameter` -  a dictionary containing the fixed values of hyperparameters of prior distributions.

Both can be defined from `FLAGS` argument with functions in [check_prior](code/check_prior.py). 
The choices and their meanings are all enumerated in details in [estimate_STBS_cluster](analysis/estimate_STBS_cluster.py). 
Note that some choices are mutually exclusive. 
[check_prior](code/check_prior.py) warns you about such inappropriate choices. 

Let's explain it on several examples. First, these settings give you the latest implementation 
of the original TBIP by Keyon Vafa (with Gamma variational families):
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
To really fit it the same way, do not forget to set
`exact_entropy = False` and `geom_approx = True`.


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

Unless you change some parameters the default values for the `FLAGS` will be used to create the dictionary
`prior_hyperparameter`. You can find more details in 
[estimate_STBS_cluster](analysis/estimate_STBS_cluster.py) and 
function `get_and_check_prior_hyperparameter` from [check_prior](code/check_prior.py).

### Other model tuning parameters

Moreover, there are other parameters that define the way STBS is estimated.

* `batch_size`: The batch size.
* `RobMon_exponent`: Exponent in [-1, -0.5) satisfying Robbins-Monroe condition to 
create convex-combinations of old and a new value.
* `exact_entropy`: Should the exact entropy be computed (True) 
or approximated with Monte Carlo (False)?
* `geom_approx`: Should the expected ideological term Edkv be approximated by the geometric mean (True) 
or should it be computed exactly (False)?
* `aux_prob_sparse`: Should the counts and auxiliary proportions be worked with 
as with sparse matrices (True/False)? 
* (From experience, strangely, sparse format does not lead to faster computation with Tensorflow.)
* `iota_coef_jointly`: Should the variational family over iota coefficients be joint (with general covariance matrix) (True) 
instead of mean-field variational family which imposes independence between coefficients (False)?

### Monitoring the estimation process

When STBS is estimated with our combination of CAVI updates and SVI, the values of
ELBO, reconstruction, log_prior and entropy are saved into `model_state.csv` file.
It contains these values for each epoch and each step. 
After reaching the maximal number of epochs, traceplots of these quantities are plotted
in the range of between the starting and last epoch. 

If `computeIC_every > 0`, then ELBO is approximated using all batches of documents (not just a single batch).
On top of that 
[VAIC and VBIC](https://onlinelibrary.wiley.com/doi/full/10.1111/anzs.12063) 
are computed alongside this thorough approximation.
Since this approximation takes some non-negligible computation time,
we recommend to compute it only one a while (after every `computeIC_every` epoch). 
Results of this approximation are saved into `epoch_data.csv` including the computation times
for estimation of the epoch as well as evaluation of the information criteria. 
Similarly, traceplots are automatically created afterwards.

## Post-processing the results

Some post-processing is done already in [estimate_STBS_cluster](analysis/estimate_STBS_cluster.py) where after the 
last epoch many useful plots (including barplots, histograms, wordclouds) are created using 
`create_all_general_descriptive_figures` and `create_all_figures_specific_to_data` from 
[plotting_functions](code/plotting_functions.py).
Then, if `num_top_speeches` > 0 then the most influential speeches are found using 
`find_most_influential_speeches` function from 
[influential_speeches](code/influential_speeches.py). 
For `hein-daily` data we decided to first select a batch (of `batch_size`) of documents
with the highest posterior mean of thetas (`shp / rte`) for each topic separately. Then,
`num_top_speeches` documents with the highest log-likelihood-ratio-like test statistic are 
saved into [txts](data/hein-daily/txts) subdirectory as the most influential speeches for the topic. 

Some post-analysis has to be performed by external `.py` or `.R` scripts.

### Model comparisons

What cannot be done immediately after estimation of an STBS is comparison of two STBS outputs.
First, both different settings have to be estimated. Then, we only load the parameters of interest
instead of the whole STBS structure.

We wish to compare the ideological positions estimated by the classical TBIP and estimated STBS
model and also compute the correlation coefficient between them to demonstrate the similarity.
We plot the results together with under many variations using 
[compare_TBIP_with_STBS](analysis/compare_TBIP_with_STBS.py).
The differences are in the way we create the group means for political parties, 
they can be either taken from the regression coefficients iota or just (weighted) averages
of ideological positions of speakers within the respective groups.
The topics are then ordered (and plotted in this order) by the difference between these two party means. 

Next, we also wish to compare the variability of ideological space between STBS model with
fixed and with topic-specific ideological positions. 
Script [compare_variability_of_ideal_term](analysis/compare_variability_of_ideal_term.py)
provides several aspects, in which these two could be topic-wise compared.
In the end, we decided for the variability induced by both 
ideological corrections eta and ideological positions, named with `eta_ideal_variability`. 
It is computed by multiplying location estimates for eta and ideal and reducing this 3D tensor
along author and word axis into variances for each topic. 
A nice barplots for this comparison including labels for the topics is created by 
[barplot_eta_ideal_variability](R/barplot_eta_ideal_variability.R). 
These labels were assigned after exploration of the wordclouds containing the most relevant terms.

### Regression summary plots using R

Base R plotting devices allow us to be more creative with regression summary plots than python environment.
Therefore, we create these plots (and regression summary tables) with `.R` scripts that can be found in `R` folder.
Functions are tailored for `hein-daily` but with some changes it could be used for other datasets as well.
We provide functions for both regression set-ups (additive and party-interaction) 
that are usable regardless of topic-specificity of ideological positions.
First, we have written a function to plot the results vertically to create thin plots.
In the end, its transposed version proved to be better for both paper and slides. 
