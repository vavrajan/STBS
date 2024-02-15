# Import global packages
import os
import time

from absl import app
from absl import flags

import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp


def print_topics(neutral_mean, negative_mean, positive_mean, vocabulary):
    """Get neutral and ideological topics to be used for Tensorboard.

    Args:
        neutral_mean: The mean of the neutral topics.
            float[num_topics, num_words]
        negative_mean: The mean of the negative topics.
            float[num_topics, num_words]
        positive_mean: The mean of the positive topics.
            float[num_topics, num_words]
        vocabulary: A list of the vocabulary.
            float[num_words]

    Returns:
        topic_strings: A list of the negative, neutral, and positive topics.
    """
    num_topics, _ = neutral_mean.shape
    words_per_topic = 10
    top_neutral_words = np.argsort(-neutral_mean, axis=1)
    top_negative_words = np.argsort(-negative_mean, axis=1)
    top_positive_words = np.argsort(-positive_mean, axis=1)
    topic_strings = []
    for topic_idx in range(num_topics):
        neutral_start_string = "Neutral {}:".format(topic_idx)
        neutral_row = [vocabulary[word] for word in
                       top_neutral_words[topic_idx, :words_per_topic]]
        neutral_row_string = ", ".join(neutral_row)
        neutral_string = " ".join([neutral_start_string, neutral_row_string])

        positive_start_string = "Positive {}:".format(topic_idx)
        positive_row = [vocabulary[word] for word in
                        top_positive_words[topic_idx, :words_per_topic]]
        positive_row_string = ", ".join(positive_row)
        positive_string = " ".join([positive_start_string, positive_row_string])

        negative_start_string = "Negative {}:".format(topic_idx)
        negative_row = [vocabulary[word] for word in
                        top_negative_words[topic_idx, :words_per_topic]]
        negative_row_string = ", ".join(negative_row)
        negative_string = " ".join([negative_start_string, negative_row_string])
        topic_strings.append("  \n".join([negative_string, neutral_string, positive_string]))
    return np.array(topic_strings)


def print_ideal_points(ideal_point_loc, author_map):
    """Print ideal point ordering for Tensorboard."""
    # Ideological positions for the first topic
    # todo: summary over all topics!
    return ", ".join(author_map[np.argsort(ideal_point_loc[:,0])])



def log_static_features(model, vocabulary, author_map, step):
    """Log static features to Tensorboard."""
    negative_mean, neutral_mean, positive_mean = model.get_topic_means()
    ideal_point_list = print_ideal_points(model.ideal_varfam.location.numpy(), author_map)
    topics = print_topics(neutral_mean, negative_mean, positive_mean, vocabulary)
    tf.summary.text("ideal_points", ideal_point_list, step=step)
    tf.summary.text("topics", topics, step=step)

    # Exp verbosities
    if model.exp_verbosity_varfam.family != 'deterministic':
        if model.prior_choice["exp_verbosity"] == "Gfix":
            tf.summary.histogram("params/exp_verbosity_shape", model.exp_verbosity_varfam.shape, step=step)
            tf.summary.histogram("params/exp_verbosity_rate", model.exp_verbosity_varfam.rate, step=step)
        elif model.prior_choice["exp_verbosity"] == "LNfix":
            tf.summary.histogram("params/exp_verbosity_location", model.exp_verbosity_varfam.location, step=step)
            tf.summary.histogram("params/exp_verbosity_scale", model.exp_verbosity_varfam.scale, step=step)
        else:
            raise ValueError("Unrecognized prior choice for exp verbosities.")

    # Theta parameters
    tf.summary.histogram("params/theta_shape", model.theta_varfam.shape, step=step)
    tf.summary.histogram("params/theta_rate", model.theta_varfam.rate, step=step)
    if model.theta_rate_varfam.family != 'deterministic':
        tf.summary.histogram("params/theta_rate_shape", model.theta_rate_varfam.shape, step=step)
        tf.summary.histogram("params/theta_rate_rate", model.theta_rate_varfam.rate, step=step)

    # Beta parameters
    tf.summary.histogram("params/beta_shape", model.beta_varfam.shape, step=step)
    tf.summary.histogram("params/beta_rate", model.beta_varfam.rate, step=step)
    if model.beta_rate_varfam.family != 'deterministic':
        tf.summary.histogram("params/beta_rate_shape", model.beta_rate_varfam.shape, step=step)
        tf.summary.histogram("params/beta_rate_rate", model.beta_rate_varfam.rate, step=step)

    # Eta parameters
    tf.summary.histogram("params/eta_location", model.eta_varfam.location, step=step)
    tf.summary.histogram("params/eta_scale", model.eta_varfam.scale, step=step)
    if model.eta_prec_varfam.family != 'deterministic':
        tf.summary.histogram("params/eta_prec_shape", model.eta_prec_varfam.shape, step=step)
        tf.summary.histogram("params/eta_prec_rate", model.eta_prec_varfam.rate, step=step)
    if model.eta_prec_rate_varfam.family != 'deterministic':
        tf.summary.histogram("params/eta_prec_rate_shape", model.eta_prec_rate_varfam.shape, step=step)
        tf.summary.histogram("params/eta_prec_rate_rate", model.eta_prec_rate_varfam.rate, step=step)

    # Ideal positions parameters
    tf.summary.histogram("params/ideal_location", model.ideal_varfam.location, step=step)
    tf.summary.histogram("params/ideal_scale", model.ideal_varfam.scale, step=step)
    if model.iota_varfam.family != 'deterministic':
        tf.summary.histogram("params/iota_location", model.iota_varfam.location, step=step)
        if model.iota_varfam.family == 'MVnormal':
            tf.summary.histogram("params/iota_scale_tril", model.iota_varfam.scale_tril, step=step)
        else:
            tf.summary.histogram("params/iota_scale", model.iota_varfam.scale, step=step)
        if model.iota_prec_varfam.family != 'deterministic':
            tf.summary.histogram("params/iota_prec_shape", model.iota_prec_varfam.shape, step=step)
            tf.summary.histogram("params/iota_prec_rate", model.iota_prec_varfam.rate, step=step)
        if model.iota_prec_rate_varfam.family != 'deterministic':
            tf.summary.histogram("params/iota_prec_rate_shape", model.iota_prec_rate_varfam.shape, step=step)
            tf.summary.histogram("params/iota_prec_rate_rate", model.iota_prec_rate_varfam.rate, step=step)
        if model.prior_choice["iota_mean"] == "Nlmean":
            tf.summary.histogram("params/iota_mean_location", model.iota_mean_varfam.location, step=step)
            tf.summary.histogram("params/iota_mean_scale", model.iota_mean_varfam.scale, step=step)
    if model.ideal_prec_varfam.family != 'deterministic':
        tf.summary.histogram("params/ideal_prec_shape", model.ideal_prec_varfam.shape, step=step)
        tf.summary.histogram("params/ideal_prec_rate", model.ideal_prec_varfam.rate, step=step)
