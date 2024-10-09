# Import global packages
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def train_step(model, inputs, outputs, optim, seed, step=None):
    """Perform a single training step.

    Args:
        model: The STBS.
        inputs: A dictionary of input tensors.
        outputs: A sparse tensor containing word counts.
        optim: An optimizer.
        seed: The random seed.
        step: The current step.

    Returns:
        total_loss: The total loss for the minibatch (the negative ELBO, sampled with Monte-Carlo).
        reconstruction_loss: The reconstruction loss (negative log-likelihood), sampled for the minibatch.
        log_prior_loss: The negative log prior.
        entropy_loss: The negative entropy.
    """
    # Perform CAVI updates.
    model.perform_cavi_updates(inputs, outputs, step)
    # Approximate the ELBO and tape the gradients.
    with tf.GradientTape() as tape:
        predictions, log_prior_loss, entropy_loss, seed = model(inputs, seed, model.num_samples)
        count_distribution = tfp.distributions.Poisson(rate=predictions)
        count_log_likelihood = tf.reduce_sum(count_distribution.log_prob(tf.sparse.to_dense(outputs)), axis=[1, 2])
        # Adjust for the fact that we're only using a minibatch.
        reconstruction_loss = -tf.reduce_mean(count_log_likelihood) * model.minibatch_scaling
        total_loss = reconstruction_loss + log_prior_loss + entropy_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    model.print_non_finite_parameters("After applying stochastic gradient updates for step " + str(step))

    return total_loss, reconstruction_loss, log_prior_loss, entropy_loss, seed