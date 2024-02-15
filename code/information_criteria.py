# Import global packages
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def get_variational_information_criteria(model, dataset, seed=None, nsamples=10):
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

    Eqmeans = model.get_Eqmeans()

    for step, batch in enumerate(iter(dataset)):
        inputs, outputs = batch
        rate_batch, log_prior_batch, entropy_batch, seed = model(inputs, seed, nsamples)
        entropy.append(-entropy_batch)
        log_prior.append(-log_prior_batch)

        # Create the Poisson distribution with given rates
        count_distribution = tfp.distributions.Poisson(rate=rate_batch)
        # reconstruction = log-likelihood of the word counts
        reconstruction_batch = count_distribution.log_prob(tf.sparse.to_dense(outputs))
        reconstruction.append(tf.reduce_mean(tf.reduce_sum(reconstruction_batch, axis=[1, 2])))

        reconstruction_at_Eqmean.append(model.get_reconstruction_at_Eqmean(inputs, outputs, Eqmeans))

    # todo Entropy and log_prior is computed several times, but it is practically the same, just different samples.
    #  Think about simplification. However, this would require different function than model().
    log_prior_MC = tf.reduce_mean(log_prior)  # mean over the same quantities in each batch
    entropy_MC = tf.reduce_mean(entropy)  # mean over the same quantities in each batch
    reconstruction_MC = tf.reduce_sum(reconstruction)  # sum over all batches
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
