# Import global packages
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.sparse as sparse
import warnings

# Import local modules
from var_and_prior_family import VariationalFamily, PriorFamily

prior_hyperparameter = {
    "theta": {"shape": 0.5, "rate": 0.5},
    "theta_rate": {"shape": 0.3, "rate": 0.3 / 0.3},
    "beta": {"shape": 0.3, "rate": 0.3},
    "beta_rate": {"shape": 0.3, "rate": 0.3 / 0.3},
    "exp_verbosity": {"location": 0.0, "scale": 1.0, "shape": 0.3, "rate": 0.3},
    "eta": {"location": 0.0, "scale": 1.0},
    "eta_prec": {"shape": 0.3, "rate": 0.3 * 2.0 / 10.0},
    "eta_prec_rate": {"shape": 0.3, "rate": 0.3 / 0.3 * 10.0 / 2.0},
    "ideal": {"location": 0.0, "scale": 1.0},
    "ideal_prec": {"shape": 0.3, "rate": 0.3},
    "iota": {"location": 0.0, "scale": 1.0},
    "iota_prec": {"shape": 0.3, "rate": 0.3 * 2.0 / 10.0},
    "iota_prec_rate": {"shape": 0.3, "rate": 0.3 / 0.3 * 10.0 / 2.0},
    "iota_mean": {"location": 0.0, "scale": 1.0},
}

prior_choice = {
    "theta": "Gfix",        # Gfix=Gamma fixed, Gdrte=Gamma d-rates, Garte=Gamma a-rates
    "exp_verbosity": "LNfix",# None=deterministic 1, LNfix=LogN(0,1), Gfix=Gamma fixed
    "beta": "Gfix",         # Gfix=Gamma fixed, Gvrte=Gamma v-rates
    "eta": "Nfix",          # Nfix=N(0,1),
                            # NkprecG=N(0,1/eta_prec_k) and eta_prec_k=Gfix,
                            # NkprecF=N(0,1/eta_prec_k) and eta_prec_k=G(.,eta_prec_rate_k)
    "ideal_dim": "ak",      # "ak" - author and topic-specific ideological positions, "a" just author-specific locations
    "ideal_mean": "Nfix",   # Nfix=N(0,.), Nreg=author-level regression N(x^T * iota, .)
    "ideal_prec": "Nfix",   # Nfix=N(.,1), Nprec=N(.,1/ideal_prec), Naprec=N(., 1/ideal_prec_{author})
    "iota_dim": "kl",       # "kl" - topic and coefficient-specific regression coefficients (cannot when ideal_dim="a"),
                            # "l" - just coefficient-specific locations, then shape: [1, num_coef]
    "iota_prec": "None",    # Nfix=N(.,1),
                            # NlprecG=N(.,1/iota_prec_l) and iota_prec_l=Gfix,,
                            # NlprecF=N(.,1/iota_prec_l) and iota_prec_l=G(.,iota_prec_rate_l),
                            # None=iotas do not exist in the model (if ideal_mean=="Nfix")
    "iota_mean": "None",    # None=iotas are centred to a fixed value, Nlmean=each regressor has its own mean across all topics
}

# todo Unify and check warnings. Print a warning by Warning without "raise"? Or warnings.warn?

class STBS(tf.keras.Model):
    """Tensorflow implementation of the Structural Text-Based Scaling Model (STBS)."""

    def __init__(self,
                 num_documents: int,
                 num_topics: int,
                 num_words: int,
                 num_authors: int,
                 num_samples: int,
                 X: tf.Tensor,
                 all_author_indices: int,
                 initial_ideal_location: tf.Tensor = None,
                 fitted_document_shape: np.ndarray = None,
                 fitted_document_rate: np.ndarray = None,
                 fitted_objective_topic_shape: np.ndarray = None,
                 fitted_objective_topic_rate: np.ndarray = None,
                 prior_hyperparameter: dict = prior_hyperparameter,
                 prior_choice: dict = prior_choice,
                 batch_size: int = 1,
                 RobMon_exponent: float = -0.7, # should be something in [-1, -0.5)
                 exact_entropy: bool = False,
                 geom_approx: bool = True,
                 aux_prob_sparse: bool = True,
                 iota_coef_jointly: bool = False):
        """Initialize STBS.

        Args:
            num_documents: The number of documents in the corpus.
            num_topics: The number of topics used for the model.
            num_words: The number of words in the vocabulary.
            num_authors: The number of authors in the corpus.
            num_samples: The number of Monte-Carlo samples to use to approximate the ELBO.
            X: The model matrix for author-level regression of the ideological positions.
                float[num_authors, num_coef]
            all_author_indices: Indices of authors for all documents.
                int[num_documents]
            initial_ideal_location: The initial ideological positions for all authors.
                float[num_authors]
            fitted_document_shape: The fitted document shape parameter from Poisson Factorization.
                Used only if pre-initializing with Poisson Factorization.
            fitted_document_rate: The fitted document rate parameter from Poisson Factorization.
            fitted_objective_topic_shape: The fitted objective topic shape parameter from Poisson Factorization.
            fitted_objective_topic_rate: The fitted objective topic rate parameter from Poisson Factorization.
            prior_hyperparameter: Dictionary of all relevant fixed prior hyperparameter values.
            prior_choice: Dictionary of indicators declaring the chosen hierarchical prior.
            batch_size: The batch size.
            RobMon_exponent: Exponent in [-1, -0.5) satisfying Robbins-Monroe condition to create convex-combinations of
                old and a new value.
            exact_entropy: Should we compute the exact entropy (True) or approximate it with Monte Carlo (False)?
            geom_approx: Should we use the geometric mean approximation (True) or exact computation (False)
                for the expected ideological term Edkv?
            aux_prob_sparse: Should we work with counts and auxiliary proportions as with sparse matrices (True/False)?
            iota_coef_jointly: Should we suppose joint variational family over iota coefficients (True) instead of
                mean-field variational family which imposes independence between coefficients (False)?
        """
        super(STBS, self).__init__()
        self.num_documents = num_documents
        self.num_topics = num_topics
        self.num_words = num_words
        self.num_authors = num_authors
        self.num_samples = num_samples
        self.num_coef = X.shape[1]  # number of columns of X is the number of regression coefficients
        self.all_author_indices = all_author_indices
        self.prior_hyperparameter = prior_hyperparameter
        self.prior_choice = prior_choice
        self.step_size = 1.0
        self.RobMon_exponent = RobMon_exponent
        self.exact_entropy = exact_entropy
        self.geom_approx = geom_approx
        self.aux_prob_sparse = aux_prob_sparse
        self.iota_coef_jointly = iota_coef_jointly
        self.batch_size = batch_size
        # batch_size = tf.shape(counts)[0]
        self.minibatch_scaling = tf.dtypes.cast(self.num_documents / batch_size, tf.float32)
        if X.dtype != "float32":
            raise ValueError("Model matrix X is of different type than tf.float32.")
        self.X = X  # shape: [num_authors, num_coef]
        self.XtX = X[:, :, tf.newaxis] * X[:, tf.newaxis, :]  # shape: [num_authors, num_coef, num_coef]
        # XtX is not yet summed over authors since it is used in this expanded form.

        # theta_rate
        if self.prior_choice["theta"] == "Gfix":
            init_loc = tf.fill([num_documents], prior_hyperparameter["theta"]["rate"])
            self.theta_rate_varfam = VariationalFamily('deterministic', [num_documents],
                                                       cavi=None, fitted_location=init_loc, name="theta_rate")
            self.theta_rate_prior = PriorFamily('deterministic', num_samples=self.num_samples, location=init_loc)
        elif self.prior_choice["theta"] == "Gdrte":
            # document-specific rates for theta
            # The shape parameter is not changed at all by CAVI updates --> can be initialized with it.
            cavi_theta_rate = tf.fill([num_documents],
                                      self.prior_hyperparameter["theta_rate"]["shape"] + self.num_topics *
                                      self.prior_hyperparameter["theta"]["shape"])
            self.theta_rate_varfam = VariationalFamily('gamma', [num_documents],
                                                       cavi=True, fitted_shape=cavi_theta_rate, name="theta_rate")
            self.theta_rate_prior = PriorFamily('gamma', num_samples=self.num_samples,
                shape=tf.fill([num_documents], prior_hyperparameter["theta_rate"]["shape"]),
                rate=tf.fill([num_documents], prior_hyperparameter["theta_rate"]["rate"]))
        elif self.prior_choice["theta"] == "Garte":
            # author-specific rates for theta
            self.theta_rate_varfam = VariationalFamily('gamma', [num_authors],
                                                       cavi=True, name="theta_rate", )
            self.theta_rate_prior = PriorFamily('gamma', num_samples=self.num_samples,
                shape=tf.fill([num_authors], prior_hyperparameter["theta_rate"]["shape"]),
                rate=tf.fill([num_authors], prior_hyperparameter["theta_rate"]["rate"]))
        else:
            raise ValueError("Unrecognized prior choice for theta.")

        # theta
        self.theta_varfam = VariationalFamily('gamma', [num_documents, num_topics],
                                              cavi=True,
                                              fitted_shape=fitted_document_shape,
                                              fitted_rate=fitted_document_rate,
                                              name="theta")
        theta_shapes = tf.fill([num_documents, num_topics], prior_hyperparameter["theta"]["shape"])
        if self.prior_choice["theta"] == "Gfix":
            # Gamma distribution with fixed values (from the dictionary of hyperparameters)
            theta_rates = tf.fill([num_documents, num_topics],
                prior_hyperparameter["theta"]["rate"])
        elif self.prior_choice["theta"] in ["Gdrte", "Garte"]:
            # initialize theta_rates with prior mean value
            theta_rates = tf.fill([num_documents, num_topics],
                prior_hyperparameter["theta_rate"]["shape"] / prior_hyperparameter["theta_rate"]["rate"])
        else:
            raise ValueError("Unrecognized prior choice for theta.")
        self.theta_prior = PriorFamily('gamma', shape=theta_shapes, rate=theta_rates)

        # exp_verbosity
        if self.prior_choice["theta"] == "Gfix":
            if self.prior_choice["exp_verbosity"] == "LNfix":
                # log-Normal(,) --> CAVI updates not possible
                self.exp_verbosity_varfam = VariationalFamily('lognormal', [num_authors],
                                                              cavi=False, name="exp_verbosity", )
                self.exp_verbosity_prior = PriorFamily('lognormal', num_samples=self.num_samples,
                    location=tf.fill([num_authors], prior_hyperparameter["exp_verbosity"]["location"]),
                    scale=tf.fill([num_authors], prior_hyperparameter["exp_verbosity"]["scale"]))
            elif self.prior_choice["exp_verbosity"] == "Gfix":
                # gamma(,) --> CAVI updates available
                self.exp_verbosity_varfam = VariationalFamily('gamma', [num_authors],
                                                              cavi=True, name="exp_verbosity", )
                self.exp_verbosity_prior = PriorFamily('gamma', num_samples=self.num_samples,
                    shape=tf.fill([num_authors], prior_hyperparameter["exp_verbosity"]["shape"]),
                    rate=tf.fill([num_authors], prior_hyperparameter["exp_verbosity"]["rate"]))
            else:
                raise ValueError("Unrecognized prior choice for exp verbosities.")
        elif self.prior_choice["theta"] in ["Gdrte", "Garte"]:
            if self.prior_choice["exp_verbosity"] != "None":
                raise ValueError("Theta has specific rates, hence, no verbosities are needed. "
                                 "(prior_choice = None is expected)")
            # verbosities do not appear in the model
            self.exp_verbosity_varfam = VariationalFamily('deterministic', [num_authors], cavi=None,
                fitted_location=tf.fill([num_authors], 1.0), name="exp_verbosity")
            self.exp_verbosity_prior = PriorFamily('deterministic', num_samples=self.num_samples,
                location=tf.fill([num_authors], 1.0))
        else:
            raise ValueError("Unrecognized combination of prior choice for exp verbosities and theta.")

        # beta_rate
        if self.prior_choice["beta"] == "Gfix":
            init_loc = tf.fill([num_words], prior_hyperparameter["beta"]["rate"])
            self.beta_rate_varfam = VariationalFamily('deterministic', [num_words],
                                                      cavi=None, fitted_location=init_loc, name="beta_rate")
            self.beta_rate_prior = PriorFamily('deterministic', num_samples=self.num_samples, location=init_loc)
        elif self.prior_choice["beta"] == "Gvrte":
            # word-specific rates for beta
            # CAVI updates for shape in this case are always the same
            # --> initialize with it and do not perform any update after
            cavi_beta_rate = tf.fill([num_words],
                                     self.prior_hyperparameter["beta_rate"]["shape"] + self.num_topics *
                                     self.prior_hyperparameter["beta"]["shape"])
            self.beta_rate_varfam = VariationalFamily('gamma', [num_words], cavi=True,
                fitted_shape=cavi_beta_rate, name="beta_rate")
            self.beta_rate_prior = PriorFamily('gamma', num_samples=self.num_samples,
                shape=tf.fill([num_words], prior_hyperparameter["beta_rate"]["shape"]),
                rate=tf.fill([num_words], prior_hyperparameter["beta_rate"]["rate"]))
        else:
            raise ValueError("Unrecognized prior choice for beta.")

        # beta
        self.beta_varfam = VariationalFamily('gamma', [num_topics, num_words], cavi=True,
            fitted_shape=fitted_objective_topic_shape,
            fitted_rate=fitted_objective_topic_rate,
            name="beta")
        beta_shapes = tf.fill([num_topics, num_words], prior_hyperparameter["beta"]["shape"])
        if self.prior_choice["beta"] == "Gfix":
            # Gamma distribution with fixed values (from the dictionary of hyperparameters)
            beta_rates = tf.fill([num_topics, num_words], prior_hyperparameter["beta"]["rate"])
        elif self.prior_choice["beta"] == "Gvrte":
            # initialize beta_rates with prior mean value
            beta_rates = tf.fill([num_topics, num_words],
                                 prior_hyperparameter["beta_rate"]["shape"] / prior_hyperparameter["beta_rate"]["rate"])
        else:
            raise ValueError("Unrecognized prior choice for beta.")
        self.beta_prior = PriorFamily('gamma', num_samples=self.num_samples, shape=beta_shapes, rate=beta_rates)

        # eta
        # variational family is always normal without any initialization
        # Restrict scale determines whether scale should be in (0,1) (True) or (0,infty) (False)
        self.eta_varfam = VariationalFamily('normal', [num_topics, num_words], cavi=False,
                                            restrict_scale=(not self.geom_approx), name="eta", )
        # eta priors
        eta_locs = tf.fill([num_topics, num_words], prior_hyperparameter["eta"]["location"])
        eta_scales = tf.fill([num_topics, num_words], prior_hyperparameter["eta"]["scale"])
        # we initialize under all choices the same way
        self.eta_prior = PriorFamily('normal', num_samples=self.num_samples, location=eta_locs, scale=eta_scales)

        # eta_prec
        if self.prior_choice["eta"] == "Nfix":
            init_loc = tf.fill([num_topics], prior_hyperparameter["eta"]["scale"])
            self.eta_prec_varfam = VariationalFamily('deterministic', [num_topics], cavi=None,
                fitted_location=init_loc, name="eta_prec")
            self.eta_prec_prior = PriorFamily('deterministic', num_samples=self.num_samples, location=init_loc)
        elif self.prior_choice["eta"] in ["NkprecG", "NkprecF"]:
            # CAVI updates for shape in this case are always the same
            # --> initialize with it and do not perform any update after.
            cavi_eta_prec = tf.fill([num_topics], self.prior_hyperparameter["eta_prec"]["shape"] + 0.5 * num_words)
            self.eta_prec_varfam = VariationalFamily('gamma', [num_topics], cavi=True,
                fitted_shape=cavi_eta_prec,
                name="eta_prec")
            self.eta_prec_prior = PriorFamily('gamma', num_samples=self.num_samples,
                shape=tf.fill([num_topics], prior_hyperparameter["eta_prec"]["shape"]),
                rate=tf.fill([num_topics], prior_hyperparameter["eta_prec"]["rate"]))
        else:
            raise ValueError("Unrecognized prior choice for eta.")

        # eta_prec_rate - only possible if prior_choice["eta"] == "NkprecF"
        if self.prior_choice["eta"] in ["Nfix", "NkprecG"]:
            init_loc = tf.fill([num_topics], prior_hyperparameter["eta_prec"]["rate"])
            self.eta_prec_rate_varfam = VariationalFamily('deterministic', [num_topics], cavi=None,
                fitted_location=init_loc, name="eta_prec_rate")
            self.eta_prec_rate_prior = PriorFamily('deterministic', num_samples=self.num_samples, location=init_loc)
        elif self.prior_choice["eta"] == "NkprecF":
            # CAVI updates for shape in this case are always the same
            # --> initialize with it and do not perform any update after.
            cavi_eta_prec_rate = tf.fill([num_topics],
                                         self.prior_hyperparameter["eta_prec_rate"]["shape"] +
                                         self.prior_hyperparameter["eta_prec"]["shape"])
            self.eta_prec_rate_varfam = VariationalFamily('gamma', [num_topics], cavi=True,
                fitted_shape=cavi_eta_prec_rate, name="eta_prec_rate")
            self.eta_prec_rate_prior = PriorFamily('gamma', num_samples=self.num_samples,
                shape=tf.fill([num_topics], prior_hyperparameter["eta_prec_rate"]["shape"]),
                rate=tf.fill([num_topics], prior_hyperparameter["eta_prec_rate"]["rate"]))
        else:
            raise ValueError("Unrecognized prior choice for eta.")

        # Ideal points
        # Variational family is always normal.
        # Initial ideal position locations may have been given.
        # --> fitted_location - if None then initialize randomly.
        if initial_ideal_location is None:
            auxlocation = tf.random.normal((self.num_authors,))[:, tf.newaxis] # [num_authors, 1]
            if self.prior_choice["ideal_dim"] == "ak":
                # repeat for topic-specific dimension
                initial_ideal_location = tf.repeat(auxlocation, self.num_topics, axis=1)

        if self.prior_choice["ideal_dim"] == "ak":
            self.ideal_dim = [num_authors, num_topics]
            if self.prior_choice["iota_dim"] == "kl":
                self.iota_dim = [num_topics, self.num_coef]
            elif self.prior_choice["iota_dim"] == "l":
                self.iota_dim = [1, self.num_coef]
            else:
                raise ValueError("Unrecognized dimension choice for the iota coefficients positions.")
        elif self.prior_choice["ideal_dim"] == "a":
            self.ideal_dim = [num_authors, 1]
            if self.prior_choice["iota_dim"] == "l":
                self.iota_dim = [1, self.num_coef]
            else:
                raise ValueError("Unrecognized or inappropriate (positions common to all authors, "
                                 "but topic-specific iotas) dimension choice for the iota coefficients positions.")
        else:
            raise ValueError("Unrecognized dimension choice for the ideal positions.")
        # Restrict scale determines whether scale should be in (0,1) (True) or (0,infty) (False)
        self.ideal_varfam = VariationalFamily('normal', self.ideal_dim, cavi=False,
                                              restrict_scale=(not self.geom_approx),
                                              fitted_location=initial_ideal_location,
                                              name="ideal", )
        # initialize the prior with Nfix
        self.ideal_prior = PriorFamily('normal', num_samples=self.num_samples,
            location=tf.fill(self.ideal_dim, prior_hyperparameter["ideal"]["location"]),
            scale=tf.fill(self.ideal_dim, prior_hyperparameter["ideal"]["scale"]))

        # Regression coefficients iota
        if self.prior_choice["ideal_mean"] == "Nfix":
            init_loc = tf.fill(self.iota_dim, prior_hyperparameter["iota"]["location"])
            self.iota_varfam = VariationalFamily('deterministic', self.iota_dim, cavi=None,
                fitted_location=init_loc, name="iota")
            self.iota_prior = PriorFamily('deterministic', num_samples=self.num_samples, location=init_loc)
        elif self.prior_choice["ideal_mean"] == "Nreg":
            if self.iota_coef_jointly:
                self.iota_varfam = VariationalFamily('MVnormal', self.iota_dim, cavi=True, name="iota", )
            else:
                self.iota_varfam = VariationalFamily('normal', self.iota_dim, cavi=True, name="iota", )
            self.iota_prior = PriorFamily('normal', num_samples=self.num_samples,
                location=tf.fill(self.iota_dim, prior_hyperparameter["iota"]["location"]),
                scale=tf.fill(self.iota_dim, prior_hyperparameter["iota"]["scale"]))
        else:
            raise ValueError("Unrecognized prior choice for mean value of the ideal positions.")

        # Prior means for the regression coefficients iota
        if self.prior_choice["iota_mean"] == "None":
            init_loc = tf.fill([self.num_coef], prior_hyperparameter["iota_mean"]["location"])
            self.iota_mean_varfam = VariationalFamily('deterministic', [self.num_coef], cavi=None,
                fitted_location=init_loc, name="iota_mean")
            self.iota_mean_prior = PriorFamily('deterministic', num_samples=self.num_samples, location=init_loc)
        elif self.prior_choice["iota_mean"] == "Nlmean":
            if self.prior_choice["iota_dim"] == "l":
                # iotas are not topic specific --> iota means do not make much sense
                warnings.warn("Iotas are fixed for all topics. "
                              "It does not make sense to have elaborate prior choice for iota_mean."
                              "Use None instead (do not appear in the model).")
            self.iota_mean_varfam = VariationalFamily('normal', [self.num_coef],
                                                      cavi=True, name="iota_mean", )
            self.iota_mean_prior = PriorFamily('normal', num_samples=self.num_samples,
                location=tf.fill([self.num_coef], prior_hyperparameter["iota_mean"]["location"]),
                scale=tf.fill([self.num_coef], prior_hyperparameter["iota_mean"]["scale"]))
        else:
            raise ValueError("Unrecognized prior choice for mean value of the coefficients iota.")

        # iota_prec - very analogical to eta_prec
        if self.prior_choice["iota_prec"] in ["None", "Nfix"]:
            init_loc = tf.fill([self.num_coef], prior_hyperparameter["iota"]["scale"])
            self.iota_prec_varfam = VariationalFamily('deterministic', [self.num_coef], cavi=None,
                fitted_location=init_loc, name="iota_prec")
            self.iota_prec_prior = PriorFamily('deterministic', num_samples=self.num_samples, location=init_loc)
        elif self.prior_choice["iota_prec"] in ["NlprecG", "NlprecF"]:
            if self.prior_choice["iota_dim"] == "l":
                # Iotas are not topic-specific --> hierarchical double or triple gamma prior is pointless
                warnings.warn("Iotas are fixed for all topics. "
                              "It does not make sense to have elaborate prior choice for precisions or variances."
                              "Use Nfix (fixed variance to all iota coefficients).")
            # CAVI updates for shape in this case are always the same
            # --> initialize with it and do not perform any update after.
            cavi_iota_prec = tf.fill([self.num_coef],
                                     self.prior_hyperparameter["iota_prec"]["shape"] + 0.5 * self.iota_dim[0])
            self.iota_prec_varfam = VariationalFamily('gamma', [self.num_coef], cavi=True,
                fitted_shape=cavi_iota_prec, name="iota_prec")
            self.iota_prec_prior = PriorFamily('gamma', num_samples=self.num_samples,
                shape=tf.fill([self.num_coef], prior_hyperparameter["iota_prec"]["shape"]),
                rate=tf.fill([self.num_coef], prior_hyperparameter["iota_prec"]["rate"]))
        else:
            raise ValueError("Unrecognized prior choice for iota (regression coefficients) precision.")

        # iota_prec_rate - only possible if prior_choice["iota_prec"] == "NlprecF"
        if self.prior_choice["iota_prec"] in ["None", "Nfix", "NlprecG"]:
            init_loc = tf.fill([self.num_coef], prior_hyperparameter["iota_prec"]["rate"])
            self.iota_prec_rate_varfam = VariationalFamily('deterministic', [self.num_coef], cavi=None,
                fitted_location=init_loc, name="iota_prec_rate")
            self.iota_prec_rate_prior = PriorFamily('deterministic', num_samples=self.num_samples, location=init_loc)
        elif self.prior_choice["iota_prec"] == "NlprecF":
            # CAVI updates for shape in this case are always the same
            # --> initialize with it and do not perform any update after.
            cavi_iota_prec_rate = tf.fill([self.num_coef],
                                          self.prior_hyperparameter["iota_prec_rate"]["shape"] +
                                          self.prior_hyperparameter["iota_prec"]["shape"])
            self.iota_prec_rate_varfam = VariationalFamily('gamma', [self.num_coef], cavi=True,
                fitted_shape=cavi_iota_prec_rate, name="iota_prec_rate")
            self.iota_prec_rate_prior = PriorFamily('gamma', num_samples=self.num_samples,
                shape=tf.fill([self.num_coef], prior_hyperparameter["iota_prec_rate"]["shape"]),
                rate=tf.fill([self.num_coef], prior_hyperparameter["iota_prec_rate"]["rate"]))
        else:
            raise ValueError("Unrecognized prior choice for iota_prec_rate.")

        # Ideal_prec
        if self.prior_choice["ideal_prec"] == "Nfix":
            init_loc = tf.fill([num_authors], prior_hyperparameter["ideal"]["scale"])
            self.ideal_prec_varfam = VariationalFamily('deterministic', [num_authors], cavi=None,
                fitted_location=init_loc, name="ideal_prec")
            self.ideal_prec_prior = PriorFamily('deterministic', num_samples=self.num_samples, location=init_loc)
        elif self.prior_choice["ideal_prec"] == "Nprec":
            # Precision for ideal points common to all authors
            # CAVI updates for shape in this case are always the same
            # --> initialize with it and do not perform any update after.
            cavi_ideal_prec = tf.fill([1], self.prior_hyperparameter["ideal_prec"]["shape"] + 0.5 * self.ideal_dim[
                1] * self.num_authors)
            self.ideal_prec_varfam = VariationalFamily('gamma', [1], cavi=True,
                                                       fitted_shape=cavi_ideal_prec, name="ideal_prec")
            self.ideal_prec_prior = PriorFamily('gamma', num_samples=self.num_samples,
                                                shape=tf.fill([1], prior_hyperparameter["ideal_prec"]["shape"]),
                                                rate=tf.fill([1], prior_hyperparameter["ideal_prec"]["rate"]))
        elif self.prior_choice["ideal_prec"] == "Naprec":
            if self.prior_choice["ideal_dim"] == "a":
                warnings.warn("Ideological positions are the same across all topics. "
                              "Then it does not make much sense to have apriori specific precision for each author."
                              "We recommend to use either None or Nprec setting for ideal_prec.")
            # author-specific precisions
            # CAVI updates for shape in this case are always the same
            # --> initialize with it and do not perform any update after.
            cavi_ideal_prec = tf.fill([num_authors],
                                      self.prior_hyperparameter["ideal_prec"]["shape"] + 0.5 * self.ideal_dim[1])
            self.ideal_prec_varfam = VariationalFamily('gamma', [num_authors], cavi=True,
                fitted_shape=cavi_ideal_prec, name="ideal_prec")
            self.ideal_prec_prior = PriorFamily('gamma', num_samples=self.num_samples,
                shape=tf.fill([num_authors], prior_hyperparameter["ideal_prec"]["shape"]),
                rate=tf.fill([num_authors], prior_hyperparameter["ideal_prec"]["rate"]))
        else:
            raise ValueError("Unrecognized prior choice for ideal.")


    def get_log_prior(self, samples):
        """Compute log prior of samples, which are stored in a dictionary of samples.

        Args:
            samples: Dictionary of samples, e.g. samples["theta"] are samples of theta.

        Returns:
            log_prior: Monte-Carlo estimate of the log prior. A tensor with shape [num_samples].
        """
        log_prior = tf.fill([samples["theta"].shape[0]], 0.0)

        ### Theta contribution
        log_prior += self.theta_rate_prior.get_log_prior(samples["theta_rate"])
        log_prior += self.theta_prior.get_log_prior(samples["theta"])

        ### Verbosities contribution
        log_prior += self.exp_verbosity_prior.get_log_prior(samples["exp_verbosity"])

        ### Beta contribution
        log_prior += self.beta_rate_prior.get_log_prior(samples["beta_rate"])
        log_prior += self.beta_prior.get_log_prior(samples["beta"])

        ### Eta contribution
        log_prior += self.eta_prec_rate_prior.get_log_prior(samples["eta_prec_rate"])
        log_prior += self.eta_prec_prior.get_log_prior(samples["eta_prec"])
        log_prior += self.eta_prior.get_log_prior(samples["eta"])

        ### Iota contribution
        log_prior += self.iota_prec_rate_prior.get_log_prior(samples["iota_prec_rate"])
        log_prior += self.iota_prec_prior.get_log_prior(samples["iota_prec"])
        log_prior += self.iota_mean_prior.get_log_prior(samples["iota_mean"])
        log_prior += self.iota_prior.get_log_prior(samples["iota"])

        ### Ideal contribution
        log_prior += self.ideal_prec_prior.get_log_prior(samples["ideal_prec"])
        log_prior += self.ideal_prior.get_log_prior(samples["ideal"])

        return log_prior


    def get_entropy(self, samples, exact=False):
        """Compute entropies of samples, which are stored in a dictionary of samples.
        Samples have to be from variational families to work as an approximation of entropy.

        Args:
            samples: Dictionary of samples, e.g. samples["theta"] are samples of theta.
            exact: [boolean] True --> exact entropy is computed using .entropy()
                            False --> entropy is approximated using the given samples (from varfam necessary!)

        Returns:
            entropy: Monte-Carlo estimate of the entropy. A tensor with shape [num_samples].
        """
        entropy = tf.fill([samples["theta"].shape[0]], 0.0)

        ### Theta contribution
        entropy += self.theta_rate_varfam.get_entropy(samples["theta_rate"], exact)
        entropy += self.theta_varfam.get_entropy(samples["theta"], exact)

        ### Verbosities contribution
        entropy += self.exp_verbosity_varfam.get_entropy(samples["exp_verbosity"], exact)

        ### Beta contribution
        entropy += self.beta_rate_varfam.get_entropy(samples["beta_rate"], exact)
        entropy += self.beta_varfam.get_entropy(samples["beta"], exact)

        ### Eta contribution
        entropy += self.eta_prec_rate_varfam.get_entropy(samples["eta_prec_rate"], exact)
        entropy += self.eta_prec_varfam.get_entropy(samples["eta_prec"], exact)
        entropy += self.eta_varfam.get_entropy(samples["eta"], exact)

        ### Iota contribution
        entropy += self.iota_prec_rate_varfam.get_entropy(samples["iota_prec_rate"], exact)
        entropy += self.iota_prec_varfam.get_entropy(samples["iota_prec"], exact)
        entropy += self.iota_mean_varfam.get_entropy(samples["iota_mean"], exact)
        entropy += self.iota_varfam.get_entropy(samples["iota"], exact)

        ### Ideal contribution
        entropy += self.ideal_prec_varfam.get_entropy(samples["ideal_prec"], exact)
        entropy += self.ideal_varfam.get_entropy(samples["ideal"], exact)

        return entropy


    def get_empty_samples(self):
        """Creates an empty dictionary for samples of the model parameters."""
        samples = {"theta": None}
        samples["theta_rate"] = None
        samples["exp_verbosity"] = None
        samples["beta"] = None
        samples["beta_rate"] = None
        samples["eta"] = None
        samples["eta_prec"] = None
        samples["eta_prec_rate"] = None
        samples["ideal"] = None
        samples["ideal_prec"] = None
        samples["iota"] = None
        samples["iota_mean"] = None
        samples["iota_prec"] = None
        samples["iota_prec_rate"] = None

        return samples


    def get_samples_and_update_prior_customized(self, samples, seed=None, varfam=True, nsamples=1):
        """
        Follow the structure of the model to sample all model parameters.
        Sample from the most inner prior distributions first and then go up the hierarchy.
        Update the priors simultaneously.
        Compute contributions to log_prior and entropy along the way.
        Return samples needed to recover the Poisson rates for word counts.

        Args:
            samples: Dictionary of samples, e.g. samples["theta"] are samples of theta.
                Some may be empty to be filled, some may be given. If given then not sample.
            seed: Random seed to set the random number generator.
            varfam: True --> sample from variational family
                   False --> sample from prior family
            nsamples: When sampling from variational family, number of samples can be specified.
                        Usually self.num_samples, but an arbitrary number can be supplied.

        Returns:
            samples: Dictionary of samples, e.g. samples["theta"] are samples of theta.
            seed: Random seed to set the random number generator.
        """
        if varfam:
            num_samples = nsamples
        else:
            num_samples = self.num_samples

        ### Theta and its prior hyperparameters
        # Theta_rate
        if samples["theta_rate"] is None:
            if varfam:
                samples["theta_rate"], seed = self.theta_rate_varfam.sample(num_samples, seed=seed)
            else:
                samples["theta_rate"], seed = self.theta_rate_prior.sample((), seed=seed)

        # Update the prior distribution of theta
        if self.theta_rate_varfam.family != 'deterministic':
            if self.prior_choice["theta"] == "Garte":  # author-specific rates
                drates = tf.gather(samples["theta_rate"], self.all_author_indices, axis=1)  # [num_samples, num_documents]
            else:
                drates = samples["theta_rate"]  # [num_samples, num_documents]
            self.theta_prior.rate.assign(tf.repeat(drates[:, :, tf.newaxis], self.num_topics,
                                                   axis=2))  # [num_samples, num_documents, num_topics]

        # Theta
        if samples["theta"] is None:
            if varfam:
                samples["theta"], seed = self.theta_varfam.sample(num_samples, seed=seed)
            else:
                samples["theta"], seed = self.theta_prior.sample((), seed=seed)

        ### Exp_verbosities
        if samples["exp_verbosity"] is None:
            if varfam:
                samples["exp_verbosity"], seed = self.exp_verbosity_varfam.sample(num_samples, seed=seed)
            else:
                samples["exp_verbosity"], seed = self.exp_verbosity_prior.sample((), seed=seed)

        ### Beta and its prior hyperparameters
        # Beta_rate
        if samples["beta_rate"] is None:
            if varfam:
                samples["beta_rate"], seed = self.beta_rate_varfam.sample(num_samples, seed=seed)
            else:
                samples["beta_rate"], seed = self.beta_rate_prior.sample((), seed=seed)

        # Update beta rates
        if self.beta_rate_varfam.family != 'deterministic':
            # There are different rates for betas --> update the rates depending on the type of the prior
            self.beta_prior.rate.assign(tf.repeat(samples["beta_rate"][:, tf.newaxis, :], self.num_topics,
                                                  axis=1))  # [num_samples, num_topics, num_words]
        # Beta
        if samples["beta"] is None:
            if varfam:
                samples["beta"], seed = self.beta_varfam.sample(num_samples, seed=seed)
            else:
                samples["beta"], seed = self.beta_prior.sample((), seed=seed)

        ### Eta and its prior hyperparameters
        # eta_prec_rate
        if samples["eta_prec_rate"] is None:
            # eta_prec_rate is randomized parameter.
            if varfam:
                samples["eta_prec_rate"], seed = self.eta_prec_rate_varfam.sample(num_samples, seed=seed)
            else:
                samples["eta_prec_rate"], seed = self.eta_prec_rate_prior.sample((), seed=seed)
        # Update eta_prec prior
        if self.eta_prec_rate_varfam.family != 'deterministic':
            self.eta_prec_prior.rate.assign(samples["eta_prec_rate"])

        # eta_prec
        if samples["eta_prec"] is None:
            if varfam:
                samples["eta_prec"], seed = self.eta_prec_varfam.sample(num_samples, seed=seed)
            else:
                samples["eta_prec"], seed = self.eta_prec_prior.sample((), seed=seed)

        # update prior
        if self.eta_prec_varfam.family != 'deterministic':
            kscales = tf.math.pow(samples["eta_prec"], -0.5)  # [num_samples, num_topics]
            self.eta_prior.scale.assign(tf.repeat(kscales[:, :, tf.newaxis], self.num_words,
                                                  axis=2))  # [num_samples, num_topics, num_words]
        # Eta
        if samples["eta"] is None:
            if varfam:
                samples["eta"], seed = self.eta_varfam.sample(num_samples, seed=seed)
            else:
                samples["eta"], seed = self.eta_prior.sample((), seed=seed)

        ### Ideal and its prior hyperparameters
        # iota_prec_rate
        if samples["iota_prec_rate"] is None:
            # iota_prec_rate is randomized parameter.
            if varfam:
                samples["iota_prec_rate"], seed = self.iota_prec_rate_varfam.sample(num_samples, seed=seed)
            else:
                samples["iota_prec_rate"], seed = self.iota_prec_rate_prior.sample((), seed=seed)
        # Update prior rates for iota_prec.
        if self.iota_prec_rate_varfam.family != 'deterministic':
            self.iota_prec_prior.rate.assign(samples["iota_prec_rate"])
        # iota_prec
        if samples["iota_prec"] is None:
            if varfam:
                samples["iota_prec"], seed = self.iota_prec_varfam.sample(num_samples, seed=seed)
            else:
                samples["iota_prec"], seed = self.iota_prec_prior.sample((), seed=seed)
        # Update prior scales for iotas.
        if self.iota_prec_prior.family != 'deterministic':
            lscales = tf.math.pow(samples["iota_prec"], -0.5)  # [num_samples, num_coef]
            self.iota_prior.scale.assign(tf.repeat(lscales[:, tf.newaxis, :], self.iota_dim[0],
                                                   axis=1)) # [num_samples, num_topics or 1, num_coef]
        # Iota_mean
        if samples["iota_mean"] is None:
            # Iotas have prior means for each coefficient.
            if varfam:
                samples["iota_mean"], seed = self.iota_mean_varfam.sample(num_samples, seed=seed)
            else:
                samples["iota_mean"], seed = self.iota_mean_prior.sample((), seed=seed)
        # Update the prior means for iota.
        if self.iota_mean_prior.family != 'deterministic':
            # From shape: [num_samples, num_coef]    -->   shape: [num_samples, num_topics or 1, num_coef]
            self.iota_prior.location.assign(tf.repeat(samples["iota_mean"][:, tf.newaxis, :], self.iota_dim[0], axis=1))

        # Iota
        if samples["iota"] is None:
            # Ideal points are modelled by regression with iota coefficients.
            if varfam:
                samples["iota"], seed = self.iota_varfam.sample(num_samples, seed=seed)
            else:
                samples["iota"], seed = self.iota_prior.sample((), seed=seed)
        # Update the prior locations for ideological positions.
        if self.iota_prior.family != 'deterministic':
            # Locations are ideal_{ak} = (X_a)^T * iota_k, hence, ideal = X * iota.
            # The problem is the first dimension of iota denoting the id of the sample.
            # X is  of shape: [num_authors, num_coef]               --> add first dimension to num_samples
            # iota  of shape: [num_samples, num_topics or 1, num_coef] --> to be transposed to shape
            # --> [num_samples, num_coef, num_topics or 1] for matrix multiplication
            # ideal of shape: [num_samples, num_authors, num_topics]
            # However, tf.linalg.matmul can handle several matrix multiplications at once:
            # Reversing iota dimensions:
            # transposed_iota = tf.transpose(samples["iota"], perm = [0, 2, 1])
            ideal_locs = tf.linalg.matmul(self.X[tf.newaxis, :, :],
                                          tf.transpose(samples["iota"], perm=[0, 2, 1]))
                                          #samples["iota"])
            # The last dimension is either [1] or [num_topics], depending on self.iota_dim.
            # If [1] but locations are topic-specific, then repeat num_topics times.
            if self.prior_choice["iota_dim"] == "l" and self.prior_choice["ideal_dim"] == "ak":
                ideal_locations = tf.repeat(ideal_locs, self.ideal_dim[1], axis=2)
            else:
                ideal_locations = ideal_locs
            self.ideal_prior.location.assign(ideal_locations)

        # Ideal_prec
        if samples["ideal_prec"] is None:
            if varfam:
                samples["ideal_prec"], seed = self.ideal_prec_varfam.sample(num_samples, seed=seed)
            else:
                samples["ideal_prec"], seed = self.ideal_prec_prior.sample((), seed=seed)
        if self.ideal_prec_prior.family != 'deterministic':
            if self.prior_choice["ideal_prec"] == "Nprec":
                scales = tf.math.pow(samples["ideal_prec"], -0.5)  # [num_samples, 1]
                ascales = tf.repeat(scales, self.num_authors, axis=1)   # [num_samples, num_authors]
            elif self.prior_choice["ideal_prec"] == "Naprec":
                ascales = tf.math.pow(samples["ideal_prec"], -0.5)  # [num_samples, num_authors]
            else:
                raise ValueError("ideal_prec distribution is not None, but prior choice is not Nprec or Naprec!")
            self.ideal_prior.scale.assign(tf.repeat(ascales[:, :, tf.newaxis], self.ideal_dim[1],
                                                    axis=2))  # [num_samples, num_authors, num_topics or 1]

        # Ideal
        if samples["ideal"] is None:
            if varfam:
                samples["ideal"], seed = self.ideal_varfam.sample(num_samples, seed=seed)
            else:
                samples["ideal"], seed = self.ideal_prior.sample((), seed=seed)

        return samples, seed


    def get_rates(self, samples, document_indices, author_indices):
        """
        Given samples of theta, beta, eta, ideal and verbosities computes the rates for Poisson counts in STBS.

        Args:
            samples: A dictionary of samples including theta, beta, eta, ideal and verbosities.
            document_indices: Indices of documents in the batch. A tensor with shape [batch_size].
            author_indices: Indices of authors in the batch. A tensor with shape [batch_size].

        Returns:
            rate: float[num_samples, batch_size, num_words]
        """
        ### Compute rate for each document in batch.
        # Start with subsetting required samples to documents included in current batch
        selected_document_samples = tf.gather(samples["theta"], document_indices,
                                              axis=1)  # [num_samples, batch_size, num_topics]
        selected_ideal_points = tf.gather(samples["ideal"], author_indices,
                                          axis=1)  # [num_samples, batch_size, num_topics or 1]

        # Compute ideological term
        selected_ideological_topic_samples = tf.exp(
            selected_ideal_points[:, :, :, tf.newaxis] *
            samples["eta"][:, tf.newaxis, :, :])  # [num_samples, batch_size, num_topics, num_words]
        # num_topics dimension comes from etas regardless of dimension of ideal positions

        rate = tf.reduce_sum(
            selected_document_samples[:, :, :, tf.newaxis] *
            samples["beta"][:, tf.newaxis, :, :] *
            selected_ideological_topic_samples[:, :, :, :],
            axis=2)  # sum over all topics (3rd dimension)

        # multiply by exp_verbosities (should be just ones if verbosities are not part of the model (prior_choice["exp_verbosity"] == "None"))
        if self.exp_verbosity_prior.family != 'deterministic':
            selected_author_verbosities = tf.gather(samples["exp_verbosity"], author_indices,
                                                    axis=1)  # [num_samples, batch_size]
            rate = rate * selected_author_verbosities[:, :, tf.newaxis]  # [num_samples, batch_size, num_words]
        # else multiply it with ones... not necessary to be performed
        return rate

    def check_and_print_non_finite(self, p, info_string, name=''):
        """Checks given parameter for NaN and infinite values and prints them if so."""
        nfp = tf.math.logical_not(tf.math.is_finite(p))
        if tf.math.reduce_any(nfp):
            indices = tf.where(nfp)
            print(info_string)
            print("Found " + str(tf.shape(indices)[0]) + " NaN or infinite values for parameter: " + name + ".")
            print("Indices: ")
            print(indices)
            print("Values:")
            print(tf.gather_nd(p, indices))

    def print_non_finite_parameters(self, info_string):
        """Checks which model parameters have NaN values and prints them and indices."""
        # Theta parameters
        self.check_and_print_non_finite(self.theta_varfam.shape, info_string, self.theta_varfam.shape.name)
        self.check_and_print_non_finite(self.theta_varfam.rate, info_string, self.theta_varfam.rate.name)
        if self.theta_rate_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.theta_rate_varfam.shape, info_string, self.theta_rate_varfam.shape.name)
            self.check_and_print_non_finite(self.theta_rate_varfam.rate, info_string, self.theta_rate_varfam.rate.name)

        # Exp_verbosity
        if self.exp_verbosity_varfam.family == 'lognormal':
            self.check_and_print_non_finite(self.exp_verbosity_varfam.location, info_string, self.exp_verbosity_varfam.location.name)
            self.check_and_print_non_finite(self.exp_verbosity_varfam.scale, info_string, self.exp_verbosity_varfam.scale.name)
        elif self.exp_verbosity_varfam.family == 'gamma':
            self.check_and_print_non_finite(self.exp_verbosity_varfam.shape, info_string, self.exp_verbosity_varfam.shape.name)
            self.check_and_print_non_finite(self.exp_verbosity_varfam.rate, info_string, self.exp_verbosity_varfam.rate.name)

        # Beta parameters
        self.check_and_print_non_finite(self.beta_varfam.shape, info_string, self.beta_varfam.shape.name)
        self.check_and_print_non_finite(self.beta_varfam.rate, info_string, self.beta_varfam.rate.name)
        if self.beta_rate_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.beta_rate_varfam.shape, info_string, self.beta_rate_varfam.shape.name)
            self.check_and_print_non_finite(self.beta_rate_varfam.rate, info_string, self.beta_rate_varfam.rate.name)

        # Eta parameters
        self.check_and_print_non_finite(self.eta_varfam.location, info_string, self.eta_varfam.location.name)
        if self.eta_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.eta_varfam.scale, info_string, self.eta_varfam.scale.name)

        if self.eta_prec_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.eta_prec_varfam.shape, info_string, self.eta_prec_varfam.shape.name)
            self.check_and_print_non_finite(self.eta_prec_varfam.rate, info_string, self.eta_prec_varfam.rate.name)

        if self.eta_prec_rate_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.eta_prec_rate_varfam.shape, info_string, self.eta_prec_rate_varfam.shape.name)
            self.check_and_print_non_finite(self.eta_prec_rate_varfam.rate, info_string, self.eta_prec_rate_varfam.rate.name)

        # Iota parameters
        self.check_and_print_non_finite(self.iota_varfam.location, info_string, self.iota_varfam.location.name)
        if self.iota_varfam.family == "normal":
            self.check_and_print_non_finite(self.iota_varfam.scale, info_string, self.iota_varfam.scale.name)
        elif self.iota_varfam.family == "MVnormal":
            self.check_and_print_non_finite(self.iota_varfam.scale_tril, info_string, self.iota_varfam.scale_tril.name)

        self.check_and_print_non_finite(self.iota_mean_varfam.location, info_string, self.iota_mean_varfam.location.name)
        if self.iota_mean_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.iota_mean_varfam.scale, info_string, self.iota_mean_varfam.scale.name)

        if self.iota_prec_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.iota_prec_varfam.shape, info_string, self.iota_prec_varfam.shape.name)
            self.check_and_print_non_finite(self.iota_prec_varfam.rate, info_string, self.iota_prec_varfam.rate.name)
        if self.iota_prec_rate_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.iota_prec_rate_varfam.shape, info_string, self.iota_prec_rate_varfam.shape.name)
            self.check_and_print_non_finite(self.iota_prec_rate_varfam.rate, info_string, self.iota_prec_rate_varfam.rate.name)

        # Ideal parameters
        self.check_and_print_non_finite(self.ideal_varfam.location, info_string, self.ideal_varfam.location.name)
        if self.ideal_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.ideal_varfam.scale, info_string, self.ideal_varfam.scale.name)
        if self.ideal_prec_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.ideal_prec_varfam.shape, info_string, self.ideal_prec_varfam.shape.name)
            self.check_and_print_non_finite(self.ideal_prec_varfam.rate, info_string, self.ideal_prec_varfam.rate.name)

    def get_gamma_distribution_Eqmean_subset(self, distribution, indices, log=False):
        """Returns mean of the variational family (Eqmean) for gamma-distributed parameter,
        e.g.theta - document intensities.
        First takes only a subset of shapes and rates corresponding to given document indices.

        Args:
            distribution: gamma distribution to work with
            log: [boolean] Should we compute E_q [ X ] (False) or E_q [ log(X) ] (True)?
            indices: Indices for the first dimension to subset. A tensor with shape [batch_size].

        Returns:
            Eqmean: [batch_size, num_topics]
        """
        shp = tf.gather(distribution.shape, indices, axis=0)
        rte = tf.gather(distribution.rate, indices, axis=0)
        if log:
            Eqmean = tf.math.digamma(shp) - tf.math.log(rte)
        else:
            Eqmean = shp / rte
        return Eqmean

    def get_Eqmean(self, distribution, log=False):
        """Returns mean of the variational family (Eqmean).

        Args:
            distribution: VariationalFamily probability distribution to work with.
            log: [boolean] Should we compute E_q [ X ] (False) or E_q [ log(X) ] (True)?

        Returns:
            Eqmean: variational mean or log-scale mean.

        """
        if log:
            # todo Can we implement a new method for each distribution that computes E_q [log(X)]?
            if distribution.family == 'deterministic':
                Eqmean = tf.math.log(distribution.location)
            elif distribution.family == 'lognormal':
                Eqmean = distribution.location
            elif distribution.family == 'gamma':
                Eqmean = tf.math.digamma(distribution.shape) - tf.math.log(distribution.rate)
            elif distribution.family in ['normal', 'MVnormal']:
                raise ValueError("Cannot compute E_q log(X) if X ~ Normal.")
            else:
                raise ValueError("Unrecognized distributional family.")
        else:
            Eqmean = distribution.distribution.mean()
        return Eqmean

    def get_ideological_term(self, author_indices):
        """Compute variational ideological term for CAVI.

        More specifically, we wish to compute:
        E[ exp(eta_kv * ideal_{a_d k}) ]
        which is hardly tractable, yet not impossible to compute.

        if geom_approx, then compute geometric mean approximation
          exp(E[log(exp(eta_kv * x_{a_d}))]) = exp(E[eta_kv] * E[ideal_{a_d k}]),

        where a_d is the author of document d.

        If (not geom_approx), then the exact computation is performed, see details in the manuscript.
        The problem is that only if product of variances of eta_kv and ideal_{a_d k} are higher or equal to 1,
        the expected value is infinity! If any of these violate this condition we have to consider
        the geometric mean approximation instead.
        For this reason we employed a restriction on the scales to be all limited in (0,1) by a suitable bijector.
        This should prevent any violation of the condition for finiteness.

        The output is used directly for the computation of the auxiliary terms for CAVI updates.

        Args:
            author_indices: Indices of authors in the batch.
                int[batch_size]

        Returns:
            expected_ideological_term: Exact expected ideological term or its geometric mean approximation.
                float[batch_size, num_topics, num_words]
        """
        ideal_point_loc = tf.gather(self.ideal_varfam.location, author_indices, axis=0)  # limitation by current batch
        # contribution of verbosities is moved to different method

        # Indicate evaluation type
        if self.geom_approx:
            expected_ideological_term = tf.exp(
                self.eta_varfam.location[tf.newaxis, :, :] * ideal_point_loc[:, :, tf.newaxis]
                # num_topics dimension got from etas despite being [1] or [num_topics] for ideal points
            )
        else:
            # Exact computation of the expected value - Works only if the variance condition is satisfied

            # Save ideal position location and scale squared
            ideal_point_loc = tf.gather(self.ideal_varfam.location, author_indices, axis=0)
            ideal_point_loc2 = tf.math.square(ideal_point_loc)
            ideal_point_var = tf.math.square(tf.gather(self.ideal_varfam.scale, author_indices, axis=0))
            # topics dimension either [1] or [num_topics] --> always corrected by eta which has num_topics always

            # Compute variance product and subtract it from 1
            var_prod = 1.0 - (ideal_point_var[:, :, tf.newaxis] * tf.math.square(self.eta_varfam.scale)[tf.newaxis, :, :])

            # Check whether the condition for finiteness of expected ideological term is fulfilled.
            all_vars_in_0_1 = tf.math.reduce_all(var_prod > 0)
            if all_vars_in_0_1:
                expected_ideological_term = tf.math.pow(var_prod, -0.5) * tf.exp(
                    0.5 *
                    (tf.math.square(self.eta_varfam.location)[tf.newaxis, :, :] * ideal_point_var[:, :, tf.newaxis] +
                     2 * self.eta_varfam.location[tf.newaxis, :, :] * ideal_point_loc[:, :, tf.newaxis] +
                     tf.math.square(self.eta_varfam.scale)[tf.newaxis, :, :] * ideal_point_loc2[:, :, tf.newaxis]) / var_prod
                )
            else:
                print(all_vars_in_0_1)
                print(var_prod)
                print(tf.math.reduce_sum(var_prod))
                print(tf.math.reduce_sum(tf.cast(var_prod > 0, tf.float32)))
                print(tf.math.reduce_sum(tf.cast(var_prod < 0, tf.float32)))
                print(tf.math.reduce_sum(tf.cast(var_prod == 0, tf.float32)))
                print("Could not compute the exact ideological term, since: " + str(all_vars_in_0_1) + " is not True.")
                # The expected ideological term is supposed to be +infinity for at least 1 term
                # --> return geometric mean approximation as safety measure
                expected_ideological_term = tf.exp(
                    self.eta_varfam.location[tf.newaxis, :, :] * ideal_point_loc[:, :, tf.newaxis]
                )
                # raise UserWarning("The exact computation of ideological term could not be computed. "
                #                   "Geometric mean approximation has been computed instead.")
                # warnings.warn(
                #     "The exact computation of ideological term could not be computed. "
                #     "Geometric mean approximation has been computed instead.")
        return expected_ideological_term

    def get_cavi_auxiliary_proportions(self, document_indices, author_indices):
        """Perform CAVI update for auxiliary proportion variables.

        Args:
            document_indices: Indices of documents in the batch.
                int[batch_size]
            author_indices: Indices of authors in the batch.
                int[batch_size]

        Returns:
            auxiliary_proportions: The updated auxiliary proportions. The tensor is normalized across topics,
                so it can be interpreted as the proportion of each topic belong to each word.
                float[batch_size, num_topics, num_words]
        """
        ## Variational means on log-scale are needed.
        # First extract only the current batch-related quantities
        document_meanlog = self.get_gamma_distribution_Eqmean_subset(self.theta_varfam, document_indices, log=True)
        ideal_point_loc = tf.gather(self.ideal_varfam.location, author_indices, axis=0)

        # E log(exp(verbosity)) from shape [num_authors] to [batch_size]
        author_verbosity_mean = tf.gather(self.get_Eqmean(self.exp_verbosity_varfam, log=True), author_indices, axis=0)
        # E log(beta)
        beta_Eqmeanlog = self.get_Eqmean(self.beta_varfam, log=True)

        ## Sum up the contributions to numerator on log-scale in an elegant way:
        aux_prob_log  = document_meanlog[:, :, tf.newaxis] + beta_Eqmeanlog[tf.newaxis, :, :]
        aux_prob_log += self.eta_varfam.location[tf.newaxis, :, :] * ideal_point_loc[:, :, tf.newaxis]
        # num_topics dimension got from etas despite being [1] or [num_topics] for ideal points
        aux_prob_log += author_verbosity_mean[:, tf.newaxis, tf.newaxis]

        # Before we proceed with exp() and normalizing over topics we perform trick ensuring numerical stability first.
        # Compute maximum over topics (second dimension) - shape: [batch_size, num_words]
        # Rescale to 3D - shape: [batch_size, num_topics, num_words]
        # Subtract the maxima over topics to obtain non-positive values to be exponentiated.
        aux_prob_log -= tf.reduce_max(aux_prob_log, axis=1)[:, tf.newaxis, :]  # -aux_prob_log_max

        # Now we can finally call the exp() function
        auxiliary_numerator = tf.exp(aux_prob_log)

        # Quantities are proportional across topics, rescale to sum to 1 over topics
        auxiliary_proportions = auxiliary_numerator / tf.reduce_sum(auxiliary_numerator, axis=1)[:, tf.newaxis, :]
        return auxiliary_proportions

    def get_cavi_sparse_auxiliary_counts(self, counts, document_indices, author_indices):
        """Perform CAVI update for auxiliary proportion variables. And multiply

        Args:
            counts: SPARSE count matrix.
            document_indices: Indices of documents in the batch.
                int[batch_size]
            author_indices: Indices of authors in the batch.
                int[batch_size]

        Returns:
            auxiliary_counts: The updated auxiliary proportions multiplied by the counts.
                A sparse tensor with shape [batch_size, num_topics, num_words].
        """
        ## Variational means on log-scale are needed.
        # First extract only the current batch-related quantities
        document_meanlog = self.get_gamma_distribution_Eqmean_subset(self.theta_varfam, document_indices, log=True)
        ideal_point_loc = tf.gather(self.ideal_varfam.location, author_indices, axis=0)

        # E log(exp(verbosity)) from shape [num_authors] to [batch_size]
        author_verbosity_mean = tf.gather(self.get_Eqmean(self.exp_verbosity_varfam, log=True), author_indices, axis=0)
        # E log(beta)
        beta_Eqmeanlog = self.get_Eqmean(self.beta_varfam, log=True)

        ## Create sparse matrices
        expanded_counts = tf.sparse.expand_dims(counts, axis=1)

        ## format where (d,v) is taken and duplicated num_topics times, then another (d,v)...
        extended_count_indices1 = tf.transpose(
            tf.tensor_scatter_nd_update(tf.transpose(tf.repeat(expanded_counts.indices, self.num_topics, axis=0)),
                                        [[1]],
                                        tf.tile(tf.range(self.num_topics, dtype=tf.int64),
                                                [tf.shape(expanded_counts.indices)[0]])[tf.newaxis, :])
        )
        # ## OR format where topic k is taken and then all pairs of (d,v) are stacked below each other
        # extended_count_indices2 = tf.transpose(
        #     tf.tensor_scatter_nd_update(tf.transpose(tf.tile(expanded_counts.indices, [self.num_topics, 1])),
        #                                 [[1]],
        #                                 tf.repeat(tf.range(self.num_topics, dtype=tf.int64),
        #                                           expanded_counts.indices.shape[0])[tf.newaxis, :])
        # )
        final_shape = tf.cast(tf.tensor_scatter_nd_update(tf.shape(expanded_counts), [[1]], [self.num_topics]), tf.int64)

        count_values = tf.repeat(counts.values, self.num_topics)

        ## Sum up the contributions to numerator on log-scale:
        kv_indices = tf.slice(extended_count_indices1, [0, 1], [tf.shape(extended_count_indices1)[0], 2])
        dk_indices = tf.slice(extended_count_indices1, [0, 0], [tf.shape(extended_count_indices1)[0], 2])
        d_indices  = tf.slice(extended_count_indices1, [0, 0], [tf.shape(extended_count_indices1)[0], 1])
        aux_prob_log_values  = tf.gather_nd(self.eta_varfam.location, indices=kv_indices)
        aux_prob_log_values *= tf.gather_nd(ideal_point_loc, indices=dk_indices)
        aux_prob_log_values += tf.gather_nd(document_meanlog, indices=dk_indices)
        aux_prob_log_values += tf.gather_nd(beta_Eqmeanlog, indices=kv_indices)
        aux_prob_log_values += tf.gather_nd(author_verbosity_mean, indices=d_indices)

        # Before we proceed with exp() and normalizing over topics we perform trick ensuring numerical stability first.
        # Compute maximum over topics (second dimension) - shape: [batch_size, num_words]
        # Rescale to 3D - shape: [batch_size, num_topics, num_words]
        # Subtract the maxima over topics to obtain non-positive values to be exponentiated.
        max_over_topic = tf.reduce_max(
            tf.reshape(aux_prob_log_values, [tf.shape(counts.indices)[0], self.num_topics]), axis=1)
        # Now we can finally call the exp() function on the shifted values
        auxiliary_numerator = tf.math.exp(aux_prob_log_values - tf.repeat(max_over_topic, self.num_topics))
        # Quantities are proportional across topics, rescale to sum to 1 over topics
        auxiliary_denominator = tf.reduce_sum(
            tf.reshape(auxiliary_numerator, [tf.shape(counts.indices)[0], self.num_topics]), axis=1)
        auxiliary_counts = tf.SparseTensor(
            indices=extended_count_indices1,
            values=auxiliary_numerator * count_values / tf.repeat(auxiliary_denominator, self.num_topics),
            dense_shape=final_shape
        )
        return auxiliary_counts

    def cavi_update_exp_verbosity_parameters(self, document_counts, expected_ideological_term,
                                             theta_Eqmean, beta_Eqmean, author_indices):
        """Perform CAVI update for exp_verbosity. Only if prior is gamma and varfam is also gamma.

        Args:
            document_counts: (sparse) total word-counts per each document in the batch.
                int[batch_size]
            expected_ideological_term: Expected value of the ideological term.
                float[batch_size, num_topics, num_words]
            theta_Eqmean: Variational means of theta for current batch.
                float[batch_size, num_topics]
            beta_Eqmean: Variational means of beta.
                float[num_topics, num_words]
            author_indices: Indices of authors in the batch
                int[batch_size]
        """
        # This is activated only if both prior and varfam are Gamma family.
        # Start with total word-counts in documents [batch_size].
        # document_counts = tf.reduce_sum(counts, axis=1) # in dense format
        # Count all expected rates and sum over words and topics.
        expected_rates = tf.reduce_sum(
            theta_Eqmean[:, :, tf.newaxis] * beta_Eqmean[tf.newaxis, :, :] * expected_ideological_term, axis=[1, 2])
        # shape: [batch_size, num_topics, num_words] --> sum over topics and words

        # Go through each document and add quantity to its author --> use tf.math.unsorted_segment_sum.
        # However, it requires all authors to be present in the batch, which cannot be guaranteed.
        # Hence, set num_segments to the number of authors
        author_expected_rates = tf.math.unsorted_segment_sum(expected_rates, author_indices,
                                                             num_segments=self.num_authors)
        author_word_count = tf.math.unsorted_segment_sum(document_counts, author_indices,
                                                         num_segments=self.num_authors)

        # Create the candidates for update, add also the minibatch_scaling constant multiplier.
        updated_exp_verbosity_shape = self.prior_hyperparameter["exp_verbosity"]["shape"] + \
                                      self.minibatch_scaling * author_word_count
        updated_exp_verbosity_rate = self.prior_hyperparameter["exp_verbosity"]["rate"] + \
                                     self.minibatch_scaling * author_expected_rates

        # Here we update parameters of all authors! Stochastic variational inference calls for updating
        # the variational parameters using a convex combination of the previous parameters and the updates.
        # We set the step size to be a decreasing sequence that satisfies the Robbins-Monro condition.
        global_exp_verbosity_shape = self.step_size * updated_exp_verbosity_shape + \
                                     (1 - self.step_size) * self.exp_verbosity_varfam.shape
        global_exp_verbosity_rate = self.step_size * updated_exp_verbosity_rate + \
                                    (1 - self.step_size) * self.exp_verbosity_varfam.rate

        # And now finally perform the update, dimensions [num_documents, num_topics] or [num_authors, num_topics] should match
        self.exp_verbosity_varfam.shape.assign(global_exp_verbosity_shape)
        self.exp_verbosity_varfam.rate.assign(global_exp_verbosity_rate)

    def cavi_update_theta_parameters(self, expected_ideological_term, theta_shape_shift,
                                     beta_Eqmean, exp_verbosity_Eqmean,
                                     document_indices, author_indices):
        """Perform CAVI update for theta parameters.

        Args:
            expected_ideological_term: Expected value of the ideological term.
                float[batch_size, num_topics, num_words]
            theta_shape_shift: Auxiliary proportions multiplied by counts and summed over words.
                float[batch_size, num_topics]
            beta_Eqmean: Variational mean of beta (objective topics).
                float[num_topics, num_words]
            exp_verbosity_Eqmean: Variational mean of exp_verbosity for current batch.
                float[batch_size]
            document_indices: Indices of documents in the batch.
                int[batch_size]
            author_indices: Indices of authors in the batch.
                int[batch_size]
        """
        # Shape: sum of y_{dv} * aux_prop_{dkv} over words to shape: [batch_size, num_topics]
        # theta_shape_shift = tf.reduce_sum(auxiliary_proportions * counts[:, tf.newaxis, :], axis=2)
        updated_theta_shape = self.prior_hyperparameter["theta"]["shape"] + theta_shape_shift

        # Rate: sum expected value of betas and ideological terms over words to shape: [batch_size, num_topics]
        theta_rate_shift = tf.reduce_sum(beta_Eqmean[tf.newaxis, :, :] * expected_ideological_term, axis=2)

        if self.prior_choice["theta"] == "Gfix":
            theta_rate_shift *= exp_verbosity_Eqmean[:, tf.newaxis]
            updated_theta_rate = self.prior_hyperparameter["theta"]["rate"] + theta_rate_shift

        # Otherwise, no verbosities enter the formula for theta_rate_shift.
        elif self.prior_choice["theta"] == "Gdrte":
            # But expected value of the rate has to be used instead of the fixed hyperparameter.
            expected_theta_rate = self.get_gamma_distribution_Eqmean_subset(self.theta_rate_varfam, document_indices)
            updated_theta_rate = expected_theta_rate[:, tf.newaxis] + theta_rate_shift

        elif self.prior_choice["theta"] == "Garte":
            # But expected value of the rate has to be used instead of the fixed hyperparameter.
            expected_theta_rate = self.get_gamma_distribution_Eqmean_subset(self.theta_rate_varfam, author_indices)
            updated_theta_rate = expected_theta_rate[:, tf.newaxis] + theta_rate_shift

        else:
            raise ValueError("Unrecognized prior_choice for theta, cannot CAVI update theta")

        # Update thetas in current batch - local updates without convex combination with previous values.
        global_theta_shape = tf.tensor_scatter_nd_update(self.theta_varfam.shape,
                                                         document_indices[:, tf.newaxis], updated_theta_shape)
        global_theta_rate = tf.tensor_scatter_nd_update(self.theta_varfam.rate,
                                                        document_indices[:, tf.newaxis], updated_theta_rate)
        # And now finally perform the update, dimensions [num_documents, num_topics] or [num_authors, num_topics] should match
        self.theta_varfam.shape.assign(global_theta_shape)
        self.theta_varfam.rate.assign(global_theta_rate)

    def cavi_update_theta_rate_parameters(self, theta_Eqmean, document_indices, author_indices):
        """Perform CAVI update for theta_rate parameters.

        Args:
            theta_Eqmean: Variational means of theta for current batch.
                float[batch_size, num_topics]
            document_indices: Indices of documents in the batch.
                int[batch_size]
            author_indices: Indices of authors in the batch.
                int[batch_size]
        """
        # Shape parameter does not have to be updated, it is always fixed.
        # These shapes were already initialized with this CAVI update.
        # Only rates for theta_rate have to be updated
        theta_rate_rate_shift = tf.reduce_sum(theta_Eqmean, axis=1)
        # shape: [batch_size, num_topics] --> sum the expected values over topics
        # now we have shape [batch_size]

        if self.prior_choice["theta"] == "Gdrte":
            updated_theta_rate_rate = self.prior_hyperparameter["theta_rate"]["rate"] + theta_rate_rate_shift
            # get original values self.theta_rate_varfam.rate and the updated ones
            # document_indices[:,tf.newaxis] transposes, e.g., [0,1,2] into [[0], [1], [2]]
            # Change theta_rates to CAVI update for documents in current batch.
            global_theta_rate_rate = tf.tensor_scatter_nd_update(self.theta_rate_varfam.rate,
                                                                 document_indices[:, tf.newaxis],
                                                                 updated_theta_rate_rate)
            # Actually, local (not global) updates --> no convex combination with previous values.
            # local update is of shape: [num_documents], as it should be
        elif self.prior_choice["theta"] == "Garte":
            # This one is complicated. Some authors may not have a document in the batch.
            # Some authors may even have more documents in the current batch.
            # We wish to sum by all authors first to obtain [num_authors].
            # Go through each document and add quantity to its author --> use tf.math.unsorted_segment_sum.
            # However, it requires all authors to be present in the batch, which cannot be guaranteed.
            # Hence, set num_segments to the number of authors
            author_doc_count = tf.math.unsorted_segment_sum(tf.fill([tf.shape(author_indices)[0]], 1), author_indices,
                                                            num_segments=self.num_authors)
            author_rate_shift = tf.math.unsorted_segment_sum(theta_rate_rate_shift, author_indices,
                                                             num_segments=self.num_authors)

            updated_theta_rate_shape = self.prior_hyperparameter["theta_rate"][
                                           "shape"] + self.minibatch_scaling * self.prior_hyperparameter["theta"][
                                           "shape"] * tf.cast(author_doc_count, tf.float32) * self.num_topics
            updated_theta_rate_rate = self.prior_hyperparameter["theta_rate"][
                                          "rate"] + self.minibatch_scaling * author_rate_shift

            # Here we update parameters of all authors! Stochastic variational inference calls for updating
            # the variational parameters using a convex combination of the previous parameters and the updates.
            # We set the step size to be a decreasing sequence that satisfies the Robbins-Monro condition.
            global_theta_rate_shape = self.step_size * updated_theta_rate_shape + (
                        1 - self.step_size) * self.theta_rate_varfam.shape
            global_theta_rate_rate = self.step_size * updated_theta_rate_rate + (
                        1 - self.step_size) * self.theta_rate_varfam.rate
            # Finally, perform the update for shape parameter.
            self.theta_rate_varfam.shape.assign(global_theta_rate_shape)
        else:
            raise ValueError("Unrecognized prior_choice for theta, no theta_rate update performed.")
        # Finally, perform the update for rate parameter.
        self.theta_rate_varfam.rate.assign(global_theta_rate_rate)


    def cavi_update_beta_parameters(self,
                                    expected_ideological_term,
                                    beta_shape_shift,
                                    theta_Eqmean,
                                    exp_verbosity_Eqmean):
        """Perform CAVI update for beta parameters.

        Args:
            expected_ideological_term: Expected value of the ideological term.
                float[batch_size, num_topics, num_words]
            beta_shape_shift: Auxiliary proportions multiplied by counts and summed over documents in the batch.
                float[num_topics, num_words]
            theta_Eqmean: Variational means of theta for current batch.
                float[batch_size, num_topics]
            exp_verbosity_Eqmean: Variational mean of exp_verbosity for current batch
                float[batch_size]
        """
        # Sum of y_{dv} * aux_prop_{dkv} over documents to shape: [num_topics, num_words]
        # beta_shape_shift = tf.reduce_sum(auxiliary_proportions * counts[:, tf.newaxis, :], axis=0)
        updated_beta_shape = self.prior_hyperparameter["beta"]["shape"] + self.minibatch_scaling * beta_shape_shift

        # Multiply theta_Eqmeans, exp_verbosity_Eqmean and expected ideological term
        # to shape: [batch_size, num_topics, num_words]
        # and sum over documents to [num_topics, num_words].
        beta_rate_shift = tf.reduce_sum(
            theta_Eqmean[:, :, tf.newaxis] * exp_verbosity_Eqmean[:, tf.newaxis, tf.newaxis] * expected_ideological_term,
            axis=0)

        # Get the beta_rate_Eqmean, which may be
        beta_rate_Eqmean = self.get_Eqmean(self.beta_rate_varfam)
        updated_beta_rate = beta_rate_Eqmean[tf.newaxis, :] + self.minibatch_scaling * beta_rate_shift
        # shape: [num_topics, num_words]

        # Here we update parameters of all authors! Stochastic variational inference calls for updating
        # the variational parameters using a convex combination of the previous parameters and the updates.
        # We set the step size to be a decreasing sequence that satisfies the Robbins-Monro condition.
        global_beta_shape = self.step_size * updated_beta_shape + (1 - self.step_size) * self.beta_varfam.shape
        global_beta_rate = self.step_size * updated_beta_rate + (1 - self.step_size) * self.beta_varfam.rate

        # And now finally perform the update.
        self.beta_varfam.shape.assign(global_beta_shape)
        self.beta_varfam.rate.assign(global_beta_rate)

    def cavi_update_beta_rate_parameters(self, beta_Eqmean):
        """Perform CAVI update for beta_rate parameters.

        Args:
            beta_Eqmean: Variational mean of beta (objective topics).
                float[num_topics, num_words]
        """

        if self.prior_choice["beta"] == "Gvrte":
            # Shape parameter does not have to be updated, it is always fixed.
            # These shapes were already initialized with this CAVI update.
            # Only rates for beta_rate have to be updated.
            # beta_Eqmean of shape: [num_topics, num_words].
            beta_rate_rate_shift = tf.reduce_sum(beta_Eqmean, axis=0)  # sum over topics
            # Now we have shape: [num_words].
            updated_beta_rate_rate = self.prior_hyperparameter["beta_rate"]["rate"] + beta_rate_rate_shift
            # Make a convex combination with previous values.
            global_beta_rate_rate = self.step_size * updated_beta_rate_rate + (
                        1 - self.step_size) * self.beta_rate_varfam.rate
            # Global update is of shape: [num_words], as it should be.
        else:
            raise ValueError("Unrecognized prior_choice for beta, no beta_rate update performed.")
        # Finally, perform the updates.
        self.beta_rate_varfam.rate.assign(global_beta_rate_rate)

    def cavi_update_eta_prec_parameters(self, ):
        """ Perform CAVI update for eta_prec parameters describing prior precision of etas.
        """
        # Shape parameter has a fixed update --> already initialized with it.
        # Only rate(s) parameter(s) require an update.
        # Update depends on the assumption for ideological topics etas.
        eta_prec_rate_shift = tf.reduce_sum((self.eta_varfam.location - self.prior_hyperparameter["eta"][
            "location"]) ** 2 + tf.math.square(self.eta_varfam.scale), axis=1)
        # shape: [num_topics, num_words] --> sum over words
        # now it has shape: [num_topics], as is expected

        # eta_prec_rate_Eqmean should be either
        # the fixed hyperparameter for the rate (if None var family)
        # or the variational mean of eta_prec_rate (if Gamma var family)
        eta_prec_rate_Eqmean = self.get_Eqmean(self.eta_prec_rate_varfam)
        updated_eta_prec_rate = eta_prec_rate_Eqmean + 0.5 * eta_prec_rate_shift
        # make a convex combination with previous values
        global_eta_prec_rate = self.step_size * updated_eta_prec_rate + (1 - self.step_size) * self.eta_prec_varfam.rate
        # global update is of shape: [num_topics], as it should be
        self.eta_prec_varfam.rate.assign(global_eta_prec_rate)

    def cavi_update_eta_prec_rate_parameters(self, ):
        """ Perform CAVI update for kappa parameters describing the triple gamma for etas.
        """
        # Shape parameter has a fixed update --> already initialized with it.
        # Only rate parameter requires an update.
        # Update depends on the assumption for ideological topics etas.
        eta_prec_Eqmean = self.get_Eqmean(self.eta_prec_varfam)
        updated_eta_prec_rate_rate = prior_hyperparameter["eta_prec_rate"]["rate"] + eta_prec_Eqmean
        # make a convex combination with previous values
        global_eta_prec_rate_rate = self.step_size * updated_eta_prec_rate_rate + (
                    1 - self.step_size) * self.eta_prec_rate_varfam.rate
        # global update is of shape: [num_topics], as it should be

        # Finally, perform the update.
        self.eta_prec_rate_varfam.rate.assign(global_eta_prec_rate_rate)

    def cavi_update_ideal_prec_parameters(self, ):
        """ Perform CAVI update for parameters describing precisions for the ideological positions.
        """
        # Shape parameter has a fixed update --> already initialized with it.
        # Only rate parameter requires an update.
        # Update depends on the assumption for ideal points, especially, the regression.

        if self.prior_choice["ideal_prec"] in ["Nprec", "Naprec"]:
            if self.prior_choice["ideal_mean"] == "Nfix":
                ideal_loc_squared = (self.ideal_varfam.location - prior_hyperparameter["ideal"]["location"]) ** 2
                X_scale = tf.fill([self.num_authors, self.iota_dim[0]], 0.0)
            elif self.prior_choice["ideal_mean"] == "Nreg":
                # We need (X_a)^T * mu_{iota,k}, hence, X * iota.varfam.location.
                # X     is      of shape: [num_authors, num_coef]
                # iota.location of shape: [num_topics or 1, num_coef] --> to be transposed
                # X * iota.loc  of shape: [num_authors, num_topics or 1], depends on iota_dim
                predictor = tf.linalg.matmul(self.X,
                                             tf.transpose(self.iota_varfam.location))
                # Now combined with ideal --> gets shape from that: [num_authors, num_topics or 1]
                # always the topic dimension: ideal >= iota (otherwise error should have been printed before)
                ideal_loc_squared = (self.ideal_varfam.location - predictor) ** 2
                # shape: [num_authors, num_topics or 1], depends on ideal_dim
                if self.iota_coef_jointly:
                    # x_a^T × iota_varfam.covariance × x_a, for each author a
                    # iota_varfam.covariance() of shape: [num_coef, num_coef]
                    # self.X of shape [num_authors, num_coef]
                    # X_scale = tf.linalg.diag_part(tf.transpose(self.X) @ self.iota_varfam.covariance() @ self.X)
                    A = self.X @ self.iota_varfam.scale_tril
                    X_scale = tf.linalg.diag_part(A @ tf.transpose(A))[:, tf.newaxis]
                    # shape: [num_authors, 1]
                else:
                    Xsquared = self.X * self.X  # shape: [num_authors, num_coef]
                    X_scale = tf.reduce_sum(
                        Xsquared[:, tf.newaxis, :] * tf.math.square(self.iota_varfam.scale)[tf.newaxis, :, :], axis=2)
                    # shape: [num_authors, num_topics or 1, num_coef], depends on iota_dim
                    # sum over coefficients --> [num_authors, num_topics or 1]


            else:
                raise ValueError("Unrecognized prior_choice[ideal_mean], cannot CAVI-update ideal_prec!")
            # ideal_loc_squared     is of shape: self.ideal_dim
            # X_scale               is of shape: [num_authors, self.iota_dim[0]]
            # ideal_varfam.scale    is of shape: self.ideal_dim

            # It may happen that the topic dimension is still [1] if ideal is not topic-specific
            if self.prior_choice["ideal_prec"] == "Nprec":
                # There is one precision for all authors --> sum over both topics and authors
                ideal_prec_rate_shift = tf.reduce_sum(
                    ideal_loc_squared + tf.math.square(self.ideal_varfam.scale) + X_scale)
            else:
                # Each author has his own precision --> sum just over topics
                ideal_prec_rate_shift = tf.reduce_sum(
                    ideal_loc_squared + tf.math.square(self.ideal_varfam.scale) + X_scale, axis=1)
            # shape: [num_authors, num_topics or 1] --> sum over topics (and possibly over authors)
            # now it has shape: [num_authors or 1], as is expected
            # Final update is shifted prior value
            updated_ideal_prec_rate = prior_hyperparameter["ideal_prec"]["rate"] + 0.5 * ideal_prec_rate_shift
            # make a convex combination with previous values
            global_ideal_prec_rate = self.step_size * updated_ideal_prec_rate + (
                    1 - self.step_size) * self.ideal_prec_varfam.rate
            # global update is of shape: [num_authors or 1], as it should be
        else:
            raise ValueError("Unrecognized prior_choice[ideal_prec], cannot CAVI-update ideal_prec!")
        # Finally, perform the updates.
        self.ideal_prec_varfam.rate.assign(global_ideal_prec_rate)

    def cavi_update_iota_parameters(self, iota_prec_Eqmean):
        """ Perform CAVI update for regression coefficients for the ideal points.
        Like a weighted regression with priors.

        Args:
            iota_prec_Eqmean: Variational mean of iota_prec.
                float[num_coef]
        """

        ## Right-hand side of the system of linear equations to be solved.
        # Reduce from shape [num_authors, num_coef, num_topics or 1] to [num_coef, num_topics or 1] by summing over authors.
        ideal_prec_Eqmean = self.get_Eqmean(self.ideal_prec_varfam)
        rhs_lk = tf.reduce_sum(
            ideal_prec_Eqmean[:, tf.newaxis, tf.newaxis] * self.ideal_varfam.location[:, tf.newaxis, :] * self.X[:, :,
                                                                                                          tf.newaxis],
            axis=0)
        # Add the prior mean value.
        rhs_lk += iota_prec_Eqmean[:, tf.newaxis] * self.iota_mean_varfam.location[:, tf.newaxis]
        # There are num_topics right-hand sides of length num_coef to be used. Each gives update for topic-specific coefficient.
        # rhs is of shape:  [num_coef, num_topics or 1], depending on ideal_dim,
        # but we want:      [num_coef, num_topics or 1], depending on iota_dim.
        # It may happen that it is currently [num_coef, num_topics], but we want [num_coef, 1]
        if self.prior_choice["ideal_dim"] == "ak" and self.prior_choice["iota_dim"] == "l":
            # The same as if: self.ideal_dim[1] > self.iota_dim[1]
            rhs = tf.reduce_sum(rhs_lk, axis=1)[:, tf.newaxis] # sum over topics
        else:
            rhs = rhs_lk

        ## The matrix yielding the system of linear equations to be solved.
        # Reduce from shape [num_authors, num_coef, num_coef] to [num_coef, num_coef] by summing over authors.
        matrix = tf.reduce_sum(ideal_prec_Eqmean[:, tf.newaxis, tf.newaxis] * self.XtX, axis=0)
        # Correct for the possible summation over topics.
        if self.prior_choice["ideal_dim"] == "ak" and self.prior_choice["iota_dim"] == "l":
            # The same as if: self.ideal_dim[1] > self.iota_dim[1]
            matrix *= self.num_topics # multiply by the number of topics
        # Add the prior mean value - diagonal matrix with iota_prec means.
        matrix += tf.linalg.diag(iota_prec_Eqmean)
        # matrix is of shape: [num_coef, num_coef]

        ## Solve the system of linear equations.
        # updated_location = tf.linalg.solve(matrix, rhs)             # shape: [num_coef, num_topics]
        # Or more efficiently using Cholesky decomposition, since we know the matrix is symmetric and positive definite.
        # print("Matrix:")
        # print(matrix)
        chol = tf.linalg.cholesky(matrix)
        # print("chol:")
        # print(chol)
        updated_location = tf.transpose(tf.linalg.cholesky_solve(chol, rhs))
        # needs to be transposed to obtain the expected order of dimensions.
        # shape: [num_topics or 1, num_coef], depending on iota_dim

        ## Global updates for location parameters
        global_location = self.step_size * updated_location + (1 - self.step_size) * self.iota_varfam.location
        self.iota_varfam.location.assign(global_location)

        ## Global updates for scale parameters
        if self.iota_coef_jointly:
            # chol is cholesky of matrix, we want cholesky of matrix^{-1}
            chol_inv = tf.linalg.inv(chol)
            matrix_inv = tf.linalg.matmul(chol_inv, chol_inv, transpose_a=True)
            chol_matrix_inv = tf.linalg.cholesky(matrix_inv)
            global_scale_tril = self.step_size * chol_matrix_inv + (1 - self.step_size) * self.iota_varfam.scale_tril
            self.iota_varfam.scale_tril.assign(global_scale_tril)
        else:
            # Updated scales are simply the reversed diagonal of the matrix.
            updated_scale = tf.math.pow(tf.linalg.diag_part(matrix), -0.5)
            # shape: [num_coef], but we want iota_dim --> use [tf.newaxis, :] in update below
            global_scale = self.step_size * updated_scale[tf.newaxis, :] + (1 - self.step_size) * self.iota_varfam.scale
            self.iota_varfam.scale.assign(global_scale)

    def cavi_update_iota_prec_parameters(self, ):
        """ Perform CAVI update for iota_prec parameters describing prior of iotas.
        Very analogical to eta_prec parameter.
        #todo Think about writing one function appliable for these two circumstances.
        """
        # Shape parameter has a fixed update --> already initialized with it.
        # Only rate(s) parameter(s) require an update.
        # Update depends on the assumption for the coefficients iota.

        # shape: [num_topics, num_coef]
        if self.iota_coef_jointly:
            iota_scale_kl = tf.linalg.diag_part(self.iota_varfam.scale_tril @ tf.transpose(self.iota_varfam.scale_tril))[tf.newaxis, :]
        else:
            iota_scale_kl = tf.math.square(self.iota_varfam.scale)
        iota_prec_rate_shift_kl = tf.math.square(
            self.iota_varfam.location - self.iota_mean_varfam.location[tf.newaxis, :]) + iota_scale_kl

        if self.iota_mean_varfam.family != 'deterministic':
            iota_prec_rate_shift_kl += tf.math.square(self.iota_mean_varfam.scale)[tf.newaxis, :]
        iota_prec_rate_shift = tf.reduce_sum(iota_prec_rate_shift_kl, axis=0)  # sum over topics (if there are any...)
        # now it has shape: [num_coef], as is expected

        # iota_prec_rate_Eqmean should be either
        # the fixed hyperparameter for the rate (if None var family)
        # or the variational mean of iota_prec_rate (if Gamma var family)
        iota_prec_rate_Eqmean = self.get_Eqmean(self.iota_prec_rate_varfam)
        updated_iota_prec_rate = iota_prec_rate_Eqmean + 0.5 * iota_prec_rate_shift
        # make a convex combination with previous values
        global_iota_prec_rate = self.step_size * updated_iota_prec_rate + (
                1 - self.step_size) * self.iota_prec_varfam.rate
        # global update is of shape: [num_coef], as it should be
        self.iota_prec_varfam.rate.assign(global_iota_prec_rate)

    def cavi_update_iota_prec_rate_parameters(self, iota_prec_Eqmean):
        """ Perform CAVI update for kappa parameters describing the triple gamma for iotas.
        Very analogical to eta_prec_rate parameter.
        #todo Think about writing one function appliable for these two circumstances.

        Args:
            iota_prec_Eqmean: Variational mean of iota_prec.
                float[num_coef]
        """
        # Shape parameter has a fixed update --> already initialized with it.
        # Only rate parameter requires an update.
        # Update depends on the assumption for regression coefficients iotas.

        # now it has shape: [num_coef], as is expected
        updated_iota_prec_rate_rate = prior_hyperparameter["iota_prec_rate"]["rate"] + iota_prec_Eqmean
        # make a convex combination with previous values
        global_iota_prec_rate_rate = self.step_size * updated_iota_prec_rate_rate + (
                1 - self.step_size) * self.iota_prec_rate_varfam.rate
        # global update is of shape: [num_coef], as it should be

        # Finally, perform the updates.
        self.iota_prec_rate_varfam.rate.assign(global_iota_prec_rate_rate)

    def cavi_update_iota_mean_parameters(self, iota_prec_Eqmean):
        """ Perform CAVI update for the prior means of the regression coefficients for the ideal points.
        Like a weighted mean.

        Args:
            iota_prec_Eqmean: Variational mean of iota_prec.
                float[num_coef]
        """
        ## Weighted sum of iota locations.
        # Sum over topics from shape [num_coef, num_topics or 1] to [num_coef]
        # Weigh by the iota_prec means.
        # Add scaled prior location (should be  0 / 1^2).
        prior_weight = 1.0 / tf.math.square(self.prior_hyperparameter["iota_mean"]["scale"])
        updated_location = tf.reduce_sum(self.iota_varfam.location, axis=0) * iota_prec_Eqmean + \
                           self.prior_hyperparameter["iota_mean"]["location"] * prior_weight

        ## Updated scales
        # self.iota_dim[0] is either number of topics or 1 (depending on topic-specificity of iotas)
        updated_scl = prior_weight + self.iota_dim[0] * iota_prec_Eqmean    # denominator
        updated_location /= updated_scl                           # final update of the location
        updated_scale = tf.math.pow(updated_scl, -0.5)

        ## Global updates are convex combination with the previous value.
        global_location = self.step_size * updated_location + (1 - self.step_size) * self.iota_mean_varfam.location
        global_scale = self.step_size * updated_scale + (1 - self.step_size) * self.iota_mean_varfam.scale

        ## Perform the updates.
        self.iota_mean_varfam.location.assign(global_location)
        self.iota_mean_varfam.scale.assign(global_scale)

    def perform_cavi_updates(self, inputs, outputs, step):
        """Perform CAVI updates for document intensities and objective topics.

        Args:
            inputs: A dictionary of input tensors.
            outputs: A sparse tensor containing word counts.
            step: The current training step.
        """
        self.batch_size = tf.shape(outputs)[0]
        self.print_non_finite_parameters("At start of perform_cavi_updates for step " + str(step))

        if self.aux_prob_sparse:
            # "Faster" SPARSE calculation
            start = time.time()
            auxiliary_counts = self.get_cavi_sparse_auxiliary_counts(outputs,
                                                                     inputs['document_indices'],
                                                                     inputs['author_indices'])
            beta_shape_shift = tf.sparse.reduce_sum(auxiliary_counts, axis=0)
            theta_shape_shift = tf.sparse.reduce_sum(auxiliary_counts, axis=2)
            document_counts = tf.sparse.reduce_sum(outputs, axis=1)
            end = time.time()
            print(end - start)
        else:
            # Original, "slower", memory-inefficient calculation
            start = time.time()
            counts = tf.sparse.to_dense(outputs)
            auxiliary_proportions = self.get_cavi_auxiliary_proportions(inputs['document_indices'],
                                                                        inputs['author_indices'])
            aux_counts = auxiliary_proportions * counts[:, tf.newaxis, :]
            beta_shape_shift = tf.reduce_sum(aux_counts, axis=0)
            theta_shape_shift = tf.reduce_sum(aux_counts, axis=2)
            document_counts = tf.reduce_sum(counts, axis=1)
            end = time.time()
            print(end - start)


        # The updates all use the following ideological term and mean of verbosities
        expected_ideological_term = self.get_ideological_term(inputs['author_indices'])
        self.check_and_print_non_finite(expected_ideological_term,
                                        "After updating expected_ideological_term for step " + str(step),
                                        name='expected_ideological_term')


        # We scale to account for the fact that we're only using a minibatch to
        # update the variational parameters of a global latent variable.
        self.minibatch_scaling = tf.cast(self.num_documents / self.batch_size, tf.dtypes.float32)
        self.step_size = tf.math.pow(tf.cast(step, tf.dtypes.float32) + 1, self.RobMon_exponent)
        # print(self.step_size)

        ## Auxiliary proportions
        # An auxiliary latent variable is required to perform the CAVI updates for the document intensities and objective topics.

        ### Get exp_verbosity and theta variational means
        exp_verbosity_Eqmean = tf.gather(self.get_Eqmean(self.exp_verbosity_varfam, log=False),
                                         inputs['author_indices'], axis=0)
        self.check_and_print_non_finite(exp_verbosity_Eqmean,
                                        "After updating exp_verbosity_Eqmean for step " + str(step),
                                        name='exp_verbosity_Eqmean')
        theta_Eqmean = self.get_gamma_distribution_Eqmean_subset(self.theta_varfam,
                                                                 inputs['document_indices'], log=False)
        self.check_and_print_non_finite(theta_Eqmean,
                                        "After updating theta_Eqmean for step " + str(step),
                                        name='theta_Eqmean')

        self.print_non_finite_parameters("After obtaining auxiliary variables for step " + str(step))

        ## Beta = objective topics + randomized hyperparameters
        # Update the objective topics beta and randomized hyperparameters of its prior.
        self.cavi_update_beta_parameters(expected_ideological_term,
                                         beta_shape_shift, theta_Eqmean, exp_verbosity_Eqmean)
        self.print_non_finite_parameters("After updating beta parameters for step " + str(step))
        print(self.beta_varfam.rate[2, 3150:3155])
        print(self.beta_varfam.rate[8, 5522:5524])

        beta_Eqmean = self.beta_varfam.shape / self.beta_varfam.rate
        if self.beta_rate_varfam.family != 'deterministic':
            self.cavi_update_beta_rate_parameters(beta_Eqmean)
            self.print_non_finite_parameters("After updating beta_rate parameters for step " + str(step))
        # else there is nothing to be updated!



        ## Theta = document intensities + randomized hyperparameters
        # Update the document intensities theta and randomized hyperparameters of its prior.
        self.cavi_update_theta_parameters(expected_ideological_term,
                                          theta_shape_shift, beta_Eqmean, exp_verbosity_Eqmean,
                                          inputs['document_indices'], inputs['author_indices'])
        self.print_non_finite_parameters("After updating theta parameters for step " + str(step))
        # Update theta_Eqmean after the change of the parameters
        # theta_Eqmean shall not be overwritten (proper tensorflow use)
        theta_Eqmean_updated = self.get_gamma_distribution_Eqmean_subset(self.theta_varfam,
                                                                         inputs['document_indices'], log=False)
        if self.theta_rate_varfam.family != 'deterministic':
            self.cavi_update_theta_rate_parameters(theta_Eqmean_updated,
                                                   inputs['document_indices'], inputs['author_indices'])
            self.print_non_finite_parameters("After updating theta_rate parameters for step " + str(step))
        # else there is nothing to be updated!


        ## Exp_verbosity
        # Update exp_verbosity parameters. CAVI is possible only if gamma prior and gamma variational family are used.
        # Under LogNormal prior and whatever variational family we cannot find CAVI updates in closed form.
        # Have to be updated via stochastic gradient approach instead.
        if self.prior_choice["exp_verbosity"] == "Gfix":
            self.cavi_update_exp_verbosity_parameters(document_counts, expected_ideological_term,
                                                      theta_Eqmean_updated, beta_Eqmean, inputs['author_indices'])
            self.print_non_finite_parameters("After updating exp_verbosity parameters for step " + str(step))

        ## Parameters for etas and ideals cannot be updated by CAVI due to complicated shape of expected_ideological_term.
        ## Hence, they are updated later by stochastic gradient approach.
        ## Nevertheless, randomized hyperparameters for their prior distribution can be updated.

        ## Randomized hyperparameters for Eta = ideological topics.
        if self.eta_prec_varfam.family != 'deterministic':
            self.cavi_update_eta_prec_parameters()
            self.print_non_finite_parameters("After updating eta_prec parameters for step " + str(step))
        # else there is nothing to be updated!

        if self.eta_prec_rate_varfam.family != 'deterministic':
            self.cavi_update_eta_prec_rate_parameters()
            self.print_non_finite_parameters("After updating eta_prec_rate parameters for step " + str(step))
        # else there is nothing to be updated!



        ## Randomized hyperparameters for Ideal = ideological positions.
        if self.ideal_prec_varfam.family != 'deterministic':
            self.cavi_update_ideal_prec_parameters()
            self.print_non_finite_parameters("After updating ideal_prec for step " + str(step))

        # Then the regression part of the model. First update iota_prec.
        if self.iota_prec_varfam.family != 'deterministic':
            self.cavi_update_iota_prec_parameters()
            self.print_non_finite_parameters("After updating iota_prec parameters for step " + str(step))
        # Compute iota_prec_Eqmean --> will be used several times in other updates.
        iota_prec_Eqmean = self.get_Eqmean(self.iota_prec_varfam)

        # Other updates can be done in an arbitrary order.
        if self.iota_varfam.family != 'deterministic':
            self.cavi_update_iota_parameters(iota_prec_Eqmean)
            self.print_non_finite_parameters("After updating iota parameters for step " + str(step))

        if self.iota_mean_varfam.family != 'deterministic':
            self.cavi_update_iota_mean_parameters(iota_prec_Eqmean)
            self.print_non_finite_parameters("After updating iota_mean parameters for step " + str(step))

        if self.iota_prec_rate_varfam.family != 'deterministic':
            self.cavi_update_iota_prec_rate_parameters(iota_prec_Eqmean)
            self.print_non_finite_parameters("After updating iota_prec_rate parameters for step " + str(step))


    def get_topic_means(self):
        """Get neutral and ideological topics from variational parameters.

        For each (k,v), we want to evaluate E[beta_kv], E[beta_kv * exp(eta_kv)],
        and E[beta_kv * exp(-eta_kv)], where the expectations are with respect to
        the variational distributions. Like the paper, beta refers to the objective
        topic and eta refers to the ideological topic.

        The exact form depends on the variational family (gamma or lognormal).

        Returns:
            negative_mean: The variational mean for the ideological topics with an ideal point of -1.
                float[num_topics, num_words]
            neutral_mean: The variational mean for the neutral topics (an ideal point of 0).
                float[num_topics, num_words]
            positive_mean: The variational mean for the ideological topics with an ideal point of +1.
                float[num_topics, num_words]
        """

        ideological_topic_loc = self.eta_varfam.location
        ideological_topic_scale = self.eta_varfam.scale
        objective_topic_shape = self.beta_varfam.shape
        objective_topic_rate = self.beta_varfam.rate

        neutral_mean = objective_topic_shape / objective_topic_rate
        positive_mean = ((objective_topic_shape / objective_topic_rate) * tf.math.exp(
            ideological_topic_loc +
            (ideological_topic_scale ** 2) / 2))
        negative_mean = ((objective_topic_shape / objective_topic_rate) * tf.math.exp(
            -ideological_topic_loc +
            (ideological_topic_scale ** 2) / 2))

        return negative_mean, neutral_mean, positive_mean


    def get_reconstruction_at_Eqmean(self, inputs, outputs, Eqmeans):
        """Evaluates the log probability of word counts given the variational means (Eqmean or location)
        of the model parameters.

        Args:
            inputs: document and author indices of shape [batch_size]
            outputs: sparse notation of [batch_size, num_words] matrix of word counts.
            Eqmeans: dictionary of Eqmeans.

        Returns:
            reconstruction: [float] a single value of Poisson log-prob evaluated at Eqmean
        """
        ### Compute rate for each document in batch.
        selected_thetas = tf.gather(Eqmeans["theta"], inputs['document_indices'], axis=0)
        selected_ideal_points = tf.gather(self.ideal_varfam.location, inputs['author_indices'],
                                          axis=0)  # [batch_size, num_topics or 1]

        # Compute ideological term
        selected_ideological_topic_samples = tf.exp(
            selected_ideal_points[:, :, tf.newaxis] *
            self.eta_varfam.location[tf.newaxis, :, :])  # [batch_size, num_topics, num_words]
        # num_topics dimension comes from etas regardless of dimension of ideal positions

        rate = tf.reduce_sum(
            selected_thetas[:, :, tf.newaxis] *
            Eqmeans["beta"][tf.newaxis, :, :] *
            selected_ideological_topic_samples[:, :, :],
            axis=1)  # sum over all topics (2nd dimension)

        # multiply by exp_verbosities (should be just ones if verbosities are not part of the model (prior_choice["exp_verbosity"] == "None"))
        if self.exp_verbosity_prior.family != 'deterministic':
            selected_author_verbosities = tf.gather(Eqmeans["exp_verbosity"], inputs['author_indices'], axis=0)
            rate *= selected_author_verbosities[:, tf.newaxis]  # [batch_size, num_words]
        # else multiply it with ones... not necessary to be performed

        count_distribution = tfp.distributions.Poisson(rate=rate)
        # reconstruction = log-likelihood of the word counts
        reconstruction = count_distribution.log_prob(tf.sparse.to_dense(outputs))
        reconstruction = tf.reduce_sum(reconstruction)

        return reconstruction

    def get_Eqmeans(self, ):
        """
        Computes Eqmean of several parameters (those where Eqmean is not location).

        Returns:
            Eqmeans: a dictionary of Eqmeans
        """
        ### Get Eqmeans for several parameters
        Eqmeans = {"theta": self.get_Eqmean(self.theta_varfam),
                   "theta_rate": self.get_Eqmean(self.theta_rate_varfam),
                   "beta": self.get_Eqmean(self.beta_varfam),
                   "beta_rate": self.get_Eqmean(self.beta_rate_varfam),
                   "eta_prec": self.get_Eqmean(self.eta_prec_varfam),
                   "eta_prec_rate": self.get_Eqmean(self.eta_prec_rate_varfam),
                   "iota_prec": self.get_Eqmean(self.iota_prec_varfam),
                   "iota_prec_rate": self.get_Eqmean(self.iota_prec_rate_varfam),
                   "ideal_prec": self.get_Eqmean(self.ideal_prec_varfam),
                   "exp_verbosity": self.get_Eqmean(self.exp_verbosity_varfam)
                   }
        return Eqmeans

    def get_variational_information_criteria(self, dataset, seed=None, nsamples=10):
        """Performs thorough approximation of the individual components of the ELBO.
        Then computes several variational versions of known information criteria.

        Args:
            dataset: sparse notation of [num_documents, num_words] matrix of word counts. Iterator enabled.
            seed: random generator seed for sampling the parameters needed for MC approximation
            nsamples: number of samples per parameters to be sampled,
                        high values result in more precise approximations,
                        but take more time to evaluate

        Returns:
            ELBO_MC: Monte Carlo approximation of the Evidence Lower BOund
            log_prior_MC: Monte Carlo approximation of the log_prior
            entropy_MC: Monte Carlo approximation of the entropy
            reconstruction_MC: Monte Carlo approximation of the reconstruction
            reconstruction_at_Eqmean: reconstruction evaluated at variational means
            effective_number_of_parameters: effective number of parameters
            VAIC: Variational Akaike Information Criterion
            VBIC: Variational Bayes Information Criterion
            seed: seed for random generator
        """
        ### First we need to approximate the ELBO and all its components.
        # Get individual Monte Carlo approximations of rates and log-likelihoods.
        # To spare memory, we have to do it batch by batch.
        entropy = []
        log_prior = []
        reconstruction = []
        reconstruction_at_Eqmean = []

        Eqmeans = self.get_Eqmeans()

        for step, batch in enumerate(iter(dataset)):
            inputs, outputs = batch
            empty_samples = self.get_empty_samples()
            samples, seed = self.get_samples_and_update_prior_customized(empty_samples, seed=seed, varfam=True,
                                                                         nsamples=nsamples)
            log_prior_batch = self.get_log_prior(samples)
            entropy_batch = self.get_entropy(samples, exact=self.exact_entropy)
            rate_batch = self.get_rates(samples, inputs['document_indices'], inputs['author_indices'])
            # Create the Poisson distribution with given rates
            count_distribution = tfp.distributions.Poisson(rate=rate_batch)
            # reconstruction = log-likelihood of the word counts
            reconstruction_batch = count_distribution.log_prob(tf.sparse.to_dense(outputs))

            entropy.append(tf.reduce_mean(entropy_batch).numpy())
            log_prior.append(tf.reduce_mean(log_prior_batch).numpy())
            reconstruction.append(tf.reduce_mean(tf.reduce_sum(reconstruction_batch, axis=[1, 2])).numpy())
            reconstruction_at_Eqmean.append(self.get_reconstruction_at_Eqmean(inputs, outputs, Eqmeans))

        # Entropy and log_prior is computed several times, but it is practically the same, just different samples.
        log_prior_MC = tf.reduce_mean(log_prior)                # mean over the same quantities in each batch
        entropy_MC = tf.reduce_mean(entropy)                    # mean over the same quantities in each batch
        reconstruction_MC = tf.reduce_sum(reconstruction)       # sum over all batches
        ELBO_MC = log_prior_MC + entropy_MC + reconstruction_MC

        # Reconstruction at Eqmean - sum over all batches
        reconstruction_at_Eqmean_sum = tf.reduce_sum(reconstruction_at_Eqmean)

        # Effective number of parameters
        effective_number_of_parameters = 2.0 * (reconstruction_at_Eqmean_sum - reconstruction_MC)

        ## Variational Akaike Information Criterion = VAIC
        #  AIC = -2*loglik(param_ML)             + 2*number_of_parameters
        #  DIC = -2*loglik(param_posterior_mean) + 2*effective_number_of_parameters
        # VAIC = -2*loglik(param_Eqmean)         + 2*effective_number_of_parameters
        VAIC = -2.0 * reconstruction_at_Eqmean_sum + 2.0 * effective_number_of_parameters

        ## Variational Bayes Information Criterion = VBIC
        #  BIC = -2*loglik(param_ML)             + number_of_parameters * log(sample_size)
        #  BIC = -2*loglik() (param integrated out) + 2*log_prior(param_ML) +...+ O(1/sample_size) (for linear regression)
        # VBIC = -2*ELBO + 2*log_prior
        # VBIC = -2*reconstruction - 2*entropy
        VBIC = -2.0 * ELBO_MC + 2.0 * log_prior_MC
        # todo Question the reasonability of VBIC!

        return ELBO_MC, log_prior_MC, entropy_MC, reconstruction_MC, reconstruction_at_Eqmean_sum, effective_number_of_parameters, VAIC, VBIC, seed

    def call(self, inputs, seed, nsamples):
        """Approximate terms in the ELBO with Monte-Carlo samples.

        Args:
            inputs: A dictionary of input tensors.
            seed: A seed for the random number generator.
            nsamples: A number of samples to approximate variational means with by Monte Carlo.

        Returns:
            rate: Sampled word count rates of shape [num_samples, batch_size, num_words].
            negative_log_prior: The negative log prior averaged over samples of shape [1].
            negative_entropy: The negative entropy averaged over samples of shape [1].
            seed: The updated seed.
        """
        empty_samples = self.get_empty_samples()
        samples, seed = self.get_samples_and_update_prior_customized(empty_samples, seed=seed, varfam=True,
                                                                     nsamples=nsamples)
        log_prior = self.get_log_prior(samples)
        entropy = self.get_entropy(samples, exact=self.exact_entropy)
        rate = self.get_rates(samples,
                              document_indices=inputs['document_indices'],
                              author_indices=inputs['author_indices'])
        negative_log_prior = -tf.reduce_mean(log_prior)
        negative_entropy = -tf.reduce_mean(entropy)
        return rate, negative_log_prior, negative_entropy, seed
