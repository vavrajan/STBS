# Import global packages
import tensorflow as tf
import tensorflow_probability as tfp

class VariationalFamily(tf.keras.layers.Layer):
    """Object to represent variational parameters."""

    def __init__(self,
                 family,
                 shape,
                 cavi=True,
                 restrict_scale=False,
                 fitted_location=None,
                 fitted_shape=None,
                 fitted_rate=None,
                 name=None,):
        """Initialize variational family.

        Args:
            family: A string repesenting the variational family, one of "gamma", "lognormal", "normal" and "None".
            shape: A list denoting the shape of the variational parameters.
            cavi: Whether the variational parameters will be maximized with CAVI rather than with gradient ascent.
                If not cavi, then the variable becomes trainable. This reduces the amount of taped gradients,
                which significantly speeds up the algorithm.
            restrict_scale: Should we restrict the scales into the interval (0, 1)?
            fitted_location: Initial values for location parameters. Works for normal, log-normal and None family.
            fitted_shape: The fitted shape parameter from Poisson Factorization,
                used only if pre-initializing with Poisson Factoriation.
            fitted_rate: The fitted rate parameter from Poisson Factorization,
                used only if pre-initializing with Poisson Factoriation.
            name: Name that should be given to the created variables.
        """
        super(VariationalFamily, self).__init__()
        self.cavi = cavi
        self.family = family
        ### Customized bijectors for a variable in (1, infty).
        ### Useful for shape of X ~ Gamma distribution when E 1/X is of interest.
        # self.chain = tfp.bijectors.Chain([tfp.bijectors.Exp(), tfp.bijectors.Softplus()], name="one_plus_exp")
        # self.chain = tfp.bijectors.Chain([tfp.bijectors.Shift(shift=1.0), tfp.bijectors.Exp()], name="one_plus_exp")
        # self.chain = tfp.bijectors.Chain(
        #     [tfp.bijectors.Shift(shift=1.0), tfp.bijectors.Softplus(), tfp.bijectors.Shift(shift=-1.0)],
        #     name="one_plus_softplus_minus_one")
        # self.covariance_chain = tfp.bijectors.Chain(
        #     tfp.bijectors.CholeskyOuterProduct(),
        #     tfp.bijectors.FillScaleTriL()
        # )
        if family in ['normal', 'lognormal']:
            self.restrict_scale = restrict_scale
            # Regardless of cavi, no transformation needed for the location parameter.
            if fitted_location is None:
                if fitted_shape is None or fitted_rate is None:
                    self.location = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=shape), name=name+'_loc',
                                                trainable=(not cavi))
                else:
                    # initialization from Poisson factorization
                    self.location = tf.Variable(tf.math.log(fitted_shape / fitted_rate),
                                                name=name+'_loc',
                                                trainable=(not cavi))
            else:
                self.location = tf.Variable(fitted_location, name=name + '_loc', trainable=(not cavi))

            if self.restrict_scale:
                # Use transformation from R to (0,1): x -> exp(x)/(1+exp(x)) = 1/(1+exp(-x))
                # Initialize scales always with 0.5
                self.scale = tfp.util.TransformedVariable(tf.fill(shape, 0.5),
                                                          bijector=tfp.bijectors.Sigmoid(),
                                                          name=name + '_scl',
                                                          trainable=(not cavi))
            else:
                # initialize scales always with 1.0
                if cavi:
                    # If we're doing CAVI, the scale doesn't need to be a transformed variable
                    # because it's optimized directly rather than with gradient descent.
                    self.scale = tf.Variable(tf.ones(shape), name=name+'_scl', trainable=(not cavi))
                else:
                    # Use transformation from R to (0,infty): x -> log(1+exp(x))     (almost linear for high x)
                    self.scale = tfp.util.TransformedVariable(tf.ones(shape),
                                                              bijector=tfp.bijectors.Softplus(),
                                                              name=name+'_scl',
                                                              trainable=(not cavi))
                # tf optimizer works with the unrestricted variables
                # but knows that the variable of interest is the transformed one.
        elif family == 'MVnormal':
            # Location
            if fitted_location is None:
                if fitted_shape is None or fitted_rate is None:
                    self.location = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=shape), name=name+'_loc',
                                                trainable=(not cavi))
                else:
                    # initialization from Poisson factorization
                    self.location = tf.Variable(tf.math.log(fitted_shape / fitted_rate),
                                                name=name+'_loc',
                                                trainable=(not cavi))
            else:
                self.location = tf.Variable(fitted_location, name=name + '_loc', trainable=(not cavi))
            # Covariance matrix - initialize with diagonal matrices
            # Choose one of the following:
            # self.covariance = tfp.util.TransformedVariable(
            #     tf.repeat(tf.linalg.diag(tf.ones(shape[1]))[tf.newaxis, :, :], shape[0], axis=0),
            #     bijector=self.covariance_chain,
            #     name=name+'_cov',
            #     trainable=(not cavi))
            #  or
            self.scale_tril = tfp.util.TransformedVariable(
                tf.linalg.diag(tf.ones(shape[1])), # There is only 1 matrix now!
                bijector=tfp.bijectors.FillScaleTriL(),
                name=name + '_scale_tril',
                trainable=(not cavi)
            )

        elif family == 'gamma':
            if fitted_shape is not None:
                # there are some initial values
                if cavi:
                    # If we're doing CAVI, the shape doesn't need to be a transformed variable
                    # because it's optimized directly rather than with gradient descent.
                    self.shape = tf.Variable(fitted_shape, name=name+'_shp', trainable=(not cavi))
                else:
                    self.shape = tfp.util.TransformedVariable(
                        fitted_shape,
                        bijector=tfp.bijectors.Softplus(), name=name+'_shp',
                        trainable=(not cavi))
            else:
                # initialize with some values
                if cavi:
                    self.shape = tf.Variable(
                        tf.exp(0.5 * tf.keras.initializers.RandomNormal()(shape=shape)),
                        name=name+'_shp',
                        trainable=(not cavi))
                else:
                    self.shape = tfp.util.TransformedVariable(
                        tf.ones(shape),
                        bijector=tfp.bijectors.Softplus(),
                        name=name+'_shp',
                        trainable=(not cavi))

            if fitted_rate is not None:
                if cavi:
                    self.rate = tf.Variable(fitted_rate, name=name+'_rte', trainable=(not cavi))
                else:
                    self.rate = tfp.util.TransformedVariable(
                        fitted_rate,
                        bijector=tfp.bijectors.Softplus(), name=name+'_rte',
                        trainable=(not cavi))
            else:
                if cavi:
                    self.rate = tf.Variable(
                        tf.exp(0.5 * tf.keras.initializers.RandomNormal()(shape=shape)),
                        name=name+'_rte',
                        trainable=(not cavi))
                else:
                    self.rate = tfp.util.TransformedVariable(
                        tf.ones(shape),
                        bijector=tfp.bijectors.Softplus(),
                        name=name+'_rte',
                        trainable=(not cavi))

        if family == 'normal':
            self.distribution = tfp.distributions.Normal(loc=self.location,
                                                         scale=self.scale)
        elif family == 'MVnormal':
            # Choose one:
            # self.distribution = tfp.distributions.MultivariateNormalFullCovariance(loc=self.location,
            #                                                                        covariance_matrix=self.covariance)
            # or
            self.distribution = tfp.distributions.MultivariateNormalTriL(loc=self.location,
                                                                         scale_tril=self.scale_tril)
        elif family == 'lognormal':
            self.distribution = tfp.distributions.LogNormal(loc=self.location,
                                                            scale=self.scale)
        elif family == 'gamma':
            self.distribution = tfp.distributions.Gamma(concentration=self.shape,
                                                        rate=self.rate)
        elif family == 'deterministic':
            self.location = tf.Variable(fitted_location, name=name+'_loc', trainable=False)
            self.distribution = tfp.distributions.Deterministic(loc=self.location)

        else:
            raise ValueError("Unrecognized variational family.")
        # NOTE: tf.keras requires formally recognizing TFP variables in order to
        # optimize them. See: https://github.com/tensorflow/probability/issues/946
        # todo Is it still necessary with newer version of tensorflow?
        if family in ['normal', 'MVnormal', 'lognormal', 'gamma']:
            self.recognized_variables = self.distribution.variables

    def get_entropy(self, samples, exact=False):
        """Compute entropy of samples from variational distribution.
        In most cases, Monte Carlo approximation leads to non-random gradients of the entropy.
        Therefore, why should we bother with stochastic approximation,
        if the exact entropies could be calculated instead?

        exact = True:
            Compute -E_q [ log q(phi) ] exactly as predefined entropy function.
            In most cases it should lead to non-random gradients of the entropy.
        exact = False
            Approximate -E_q [ log q(phi) ] by Monte Carlo approach:
                1/num_samples * sum_i(-log q(phi^i))
            where phi^i are variational parameters of i-th sample.
        """
        # Sum all but first axis.
        if samples is not None:
            if exact:
                entropy = tf.fill([samples.shape[0]], tf.reduce_sum(self.distribution.entropy()))
            else:
                log_prob_samples = self.distribution.log_prob(samples)
                entropy = -tf.reduce_sum(log_prob_samples, axis=tuple(range(1, len(log_prob_samples.shape))))
        else:
            # When samples are empty, it returns zeros of appropriate shape.
            entropy = tf.fill([samples.shape[0]], 0.0)

        return entropy

    def sample(self, num_samples, seed=None):
        """Sample from variational family using reparameterization."""
        seed, sample_seed = tfp.random.split_seed(seed)
        samples = self.distribution.sample(num_samples, seed=sample_seed)
        return samples, seed





class PriorFamily(tf.keras.layers.Layer):
    """Object to represent variational parameters."""

    def __init__(self,
                 family,
                 num_samples=1,
                 shape=None,
                 rate=None,
                 location=None,
                 scale=None):
        """Initialize variational family.

        Args:
            family: A string representing the variational family, one of:
                'gamma', 'lognormal', 'normal' or 'deterministic'.
            num_samples: Number of samples.
            shape: Shape parameter to be stored here.
            rate: Rate parameter to be stored here.
            location: Location parameter to be stored here.
                In case family='deterministic' it declares its deterministic value.
            scale: Scale parameter to be stored here.
        """
        super(PriorFamily, self).__init__()
        self.family = family
        if family == 'normal':
            self.location = tf.Variable(tf.repeat(location[tf.newaxis,...], num_samples, axis=0), trainable=False)
            self.scale = tf.Variable(tf.repeat(scale[tf.newaxis,...], num_samples, axis=0), trainable=False)
            self.distribution = tfp.distributions.Normal(loc=self.location,
                                                         scale=self.scale)
        elif family == 'lognormal':
            self.location = tf.Variable(tf.repeat(location[tf.newaxis, ...], num_samples, axis=0), trainable=False)
            self.scale = tf.Variable(tf.repeat(scale[tf.newaxis, ...], num_samples, axis=0), trainable=False)
            self.distribution = tfp.distributions.LogNormal(loc=self.location,
                                                            scale=self.scale)
        elif family == 'gamma':
            self.shape = tf.Variable(tf.repeat(shape[tf.newaxis, ...], num_samples, axis=0), trainable=False)
            self.rate = tf.Variable(tf.repeat(rate[tf.newaxis, ...], num_samples, axis=0), trainable=False)
            self.distribution = tfp.distributions.Gamma(concentration=self.shape,
                                                        rate=self.rate)
        elif family == 'deterministic':
            self.location = tf.Variable(tf.repeat(location[tf.newaxis, ...], num_samples, axis=0), trainable=False)
            self.distribution = tfp.distributions.Deterministic(loc=self.location)
        else:
            raise ValueError("Unrecognized prior family.")
        # if family in ['normal', 'lognormal', 'gamma']:
        #     self.recognized_variables = self.distribution.variables

    def get_log_prior(self, samples, exact=False):
        """Compute log prior contribution to the ELBO.
        If exact     --> compute exact E_q [log_prior]. (FOR NOW UNAVAILABLE)
        If not exact --> approximate using Monte Carlo samples. """
        # Sum all but first axis.
        if samples is not None:
            if exact:
                # todo Exact computation of log_prior. Requires a lot of Eqmeans, Eqmeanrevs, Eqmeanlogs...
                #  Needs to cooperate with Variational family.
                # For now Monte Carlo approximation.
                log_prob_samples = self.distribution.log_prob(samples)
                log_prior = tf.reduce_sum(log_prob_samples, axis=tuple(range(1, len(log_prob_samples.shape))))
            else:
                log_prob_samples = self.distribution.log_prob(samples)
                log_prior = tf.reduce_sum(log_prob_samples, axis=tuple(range(1, len(log_prob_samples.shape))))
        else:
            log_prior = tf.fill([samples.shape[0]], 0.0)
        return log_prior

    def sample(self, num_samples, seed=None):
        """Sample from prior family."""
        seed, sample_seed = tfp.random.split_seed(seed)
        samples = self.distribution.sample(num_samples, seed=sample_seed)
        return samples, seed
