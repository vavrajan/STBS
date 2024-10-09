## Import global packages
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from absl import app
from absl import flags


## Import local modules
# Necessary to import local modules from specific directory
# todo Can the appropriate directory containing STBS be added to sys list of paths by some other means?
import sys
# first directory here is the one where analysis_cluster is located
# this is ./STBS/analysis/analysis_cluster
# So add ./ to the list so that it can find ./STBS.code....
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from STBS.code.check_prior import get_and_check_prior_choice, get_and_check_prior_hyperparameter
from STBS.code.input_pipeline import build_input_pipeline
from STBS.code.create_X import create_X
from STBS.code.STBS import STBS
from STBS.code.train_step import train_step
from STBS.code.information_criteria import get_variational_information_criteria
from STBS.code.utils import print_topics, print_ideal_points, log_static_features
from STBS.code.plotting_functions import create_all_general_descriptive_figures, create_all_figures_specific_to_data
from STBS.code.influential_speeches import find_most_influential_speeches

## FLAGS
flags.DEFINE_string("data_name", default="hein-daily", help="Data source being used.")
flags.DEFINE_string("addendum", default="114", help="String to be added to data name."
                                                    "For example, for senate speeches the session number.")
flags.DEFINE_string("checkpoint_name", default="checkpoint_name", help="Directory for saving checkpoints.")
flags.DEFINE_boolean("load_checkpoint", default=True,
                     help="Should checkpoints be loaded? If not, then the existed are overwritten.")
flags.DEFINE_enum("counts_transformation",
                  default="nothing",
                  enum_values=["nothing", "binary", "sqrt", "log"],
                  help="Transformation used on counts data.")
flags.DEFINE_boolean("pre_initialize_parameters",
                     default=False,
                     help="Whether to use pre-initialized document and topic "
                          "intensities (with Poisson factorization).")
flags.DEFINE_integer("seed", default=123456789, help="Random seed to be used.")
flags.DEFINE_integer("num_epochs", default=1000, help="Number of epochs to perform.")
flags.DEFINE_integer("save_every", default=5, help="How often should we save checkpoints?")
flags.DEFINE_integer("computeIC_every", default=0,
                     help="How often should we compute more precise approximation of the ELBO "
                          "and compute variational Information Criteria as a by-product?"
                          "If <=0, then do not compute at all.")
flags.DEFINE_integer("max_steps", default=1000000, help="Number of training steps to run.")
flags.DEFINE_integer("print_steps", default=500, help="Number of steps to print and save results.")
flags.DEFINE_integer("num_top_speeches", default=10, help="Number of top speeches to be saved, "
                                                          "if zero speech processing is completely skipped.")
flags.DEFINE_enum("how_influential", default="theta_then_loglik_ratio_test",
                  enum_values=["theta", "theta_then_loglik_ratio_test", "loglik_ratio_test"],
                  help="The method for selection of the most influential speeches:"
                       "theta = Compute variational means of theta parameters and choose documents maximizing it."
                       "theta_then_loglik_ratio_test = Choose a batch maximizing variational means of theta (see above)."
                       "    Then, compute a loglik ratio test statistic using just them."
                       "loglik_ratio_test = Compute loglik-ratio statistic from all documents and choose the document "
                       "    with the highest loglik ratio test statistic. ")

# Method of estimation flags:
flags.DEFINE_integer("num_topics", default=30, help="Number of topics.")
flags.DEFINE_float("learning_rate", default=0.01, help="Adam learning rate.")
flags.DEFINE_float("RobMon_exponent", default=-0.51, lower_bound=-1, upper_bound=-0.5,
                   help="Robbins-Monroe algorithm for stochastic gradient optimization requires the coefficients a_n "
                        "for convex combination of new update and old value to satisfy the following conditions:"
                        "(a) sum a_n = infty,"
                        "(b) sum a_n^2 < infty."
                        "We consider a_n = n^RobMon_exponent. Then,"
                        "(a) holds if RobMon_exponent  >= -1 and"
                        "(b) holds if 2*RobMon_exponent < -1,"
                        "which yields the restriction RobMon_exponent in [-1, -0.5). "
                        "It holds: n^{-1} < n^{RobMon_exponent} < n^{-0.5}."
                        "Therefore, value close to -1 puts higher strength on the old value."
                        "Value close to -0.5 prefers the newly learned direction. "
                        "For example, n=50 --> "
                        "a_n = 0.020 if exponent=-1,"
                        "a_n = 0.065 if exponent=-0.7,"
                        "a_n = 0.141 if exponent=-0.5")
flags.DEFINE_integer("batch_size", default=512, help="Batch size.")
flags.DEFINE_integer("num_samples", default=1, help="Number of samples to use for ELBO approximation.")
flags.DEFINE_integer("num_samplesIC", default=1,
                     help="Number of samples to use for detailed ELBO approximation + IC evaluation.")
flags.DEFINE_boolean("exact_entropy", default=False,
                     help="If True, entropy is calculated precisely instead of Monte Carlo approximation. "
                          "Fow now, cannot be used together with GIG family.")
flags.DEFINE_boolean("geom_approx", default=True,
                     help="Should the ideological term be approximated by geometric means (True) "
                          "or computed exactly (False) under restriction of scales into (0,1).")
flags.DEFINE_boolean("aux_prob_sparse", default=False,
                     help="Should we work with counts and auxiliary proportions as with sparse matrices (True/False)?")
flags.DEFINE_boolean("iota_coef_jointly", default=False,
                     help="If True, joint Multivariate Normal distribution is used as variational family "
                          "for regression coefficients iota, "
                          "otherwise traditional mean-field family that assumes independence is used.")
flags.DEFINE_string("covariates", default="None",
                    help="The string that declares the model formula for the model matrix X."
                         "Your choice should be implemented in create_X.py.")

# Prior structure of the model setting:
flags.DEFINE_enum("theta", default="Gfix", enum_values=["Gfix", "Gdrte", "Garte"],
                  help="Prior choice for document intensities theta:"
                       "Gfix=Gamma prior with fixed hyperparameter values,"
                       "Gdrte=Gamma prior with document-specific rates,"
                       "Garte=Gamma prior with author-specific rates.")
flags.DEFINE_enum("exp_verbosity", default="LNfix", enum_values=["None", "LNfix", "Gfix"],
                  help="Prior choice for exponential verbosity terms:"
                       "None=Deterministic 1.0 values, only if thetas have flexible rates,"
                       "LNfix=Log-normal prior with fixed hyperparameters, estimable only with stochastic gradient,"
                       "Gfix=Gamma prior with fixed hyperparameters, estimated by CAVI updates.")
flags.DEFINE_enum("beta", default="Gfix", enum_values=["Gfix", "Gvrte"],
                  help="Prior choice for neutral topics beta:"
                       "Gfix=Gamma prior with fixed hyperparameter values,"
                       "Gvrte=Gamma prior with word-specific rates.")
flags.DEFINE_enum("eta", default="Nfix", enum_values=["Nfix", "NkprecG", "NkprecF"],
                  help="Prior choice for topic corrections eta:"
                       "Nfix=Normal prior with fixed hyperparameter values,"
                       "NkprecG=Normal prior with topic-specific precisions,"
                       "NkprecF=Normal prior with topic-specific precisions with flexible rates.")
flags.DEFINE_enum("ideal_dim", default="a", enum_values=["a", "ak"],
                  help="Dimensions of the author ideological positions"
                       "a=Only one set of positions, one for each author,"
                       "ak=Each author has a set of positions, one for each topic.")
flags.DEFINE_enum("ideal_mean", default="Nfix", enum_values=["Nfix", "Nreg"],
                  help="Prior mean for the ideological positions:"
                       "Nfix=Locations fixed to zero,"
                       "Nreg=Locations predicted by a linear combination of regressors (given by iota).")
flags.DEFINE_enum("ideal_prec", default="Nfix", enum_values=["Nfix", "Nprec", "Naprec"],
                  help="Prior variance structure for ideological positions:"
                       "Nfix=Scales fixed,"
                       "Nprec=One precision common to all regressed ideological positions,"
                       "Naprec=Author-specific precisions (ideal_prec).")
flags.DEFINE_enum("iota_dim", default="l", enum_values=["l", "kl"],
                  help="Dimensions of the regression coefficients for ideological positions:"
                       "l=There is only one set of coefficients,"
                       "lk=Each topic has its own set of coefficients.")
flags.DEFINE_enum("iota_mean", default="None", enum_values=["None", "Nlmean"],
                  help="Prior mean for the regression coefficients iota:"
                       "None=Locations fixed, iota_mean does not appear in the model,"
                       "Nlmean=Each coefficient has its own prior mean.")
flags.DEFINE_enum("iota_prec", default="Nfix", enum_values=["Nfix", "NlprecG", "NlprecF"],
                  help="Prior choice for the variance structure of the regression coefficients:"
                       "Nfix=Normal prior with fixed hyperparameter values,"
                       "NlprecG=Normal prior with coefficient-specific precisions,"
                       "NlprecF=Normal prior with coefficient-specific precisions with flexible rates.")

# Model prior hyperparameters:
flags.DEFINE_float("theta_shp", default=0.3, help="Theta prior shape")
flags.DEFINE_float("theta_rte", default=0.3, help="Theta prior rate")
flags.DEFINE_float("theta_rate_shp", default=0.3, help="Theta_rate prior shape")
flags.DEFINE_float("theta_rate_mean", default=0.3, help="Theta_rate prior mean")

flags.DEFINE_float("beta_shp", default=0.3, help="Beta prior shape")
flags.DEFINE_float("beta_rte", default=0.3, help="Beta prior rate")
flags.DEFINE_float("beta_rate_shp", default=0.3, help="Beta_rate prior shape")
flags.DEFINE_float("beta_rate_mean", default=0.3, help="Beta_rate prior mean")

flags.DEFINE_float("exp_verbosity_loc", default=0.0, help="Exp_verbosity prior location")
flags.DEFINE_float("exp_verbosity_scl", default=1.0, help="Exp_verbosity prior scale")
flags.DEFINE_float("exp_verbosity_shp", default=0.3, help="Exp_verbosity prior shape")
flags.DEFINE_float("exp_verbosity_rte", default=0.3, help="Exp_verbosity prior rate")

flags.DEFINE_float("eta_loc", default=0.0, help="Eta prior location")
flags.DEFINE_float("eta_scl", default=1.0, help="Eta prior scale")
flags.DEFINE_float("eta_kappa", default=10.0, help="Kappa hyperparameter for eta precisions")
flags.DEFINE_float("eta_prec_shp", default=0.3, help="eta_prec prior shape")
flags.DEFINE_float("eta_prec_rte", default=0.3, help="eta_prec prior rate")
flags.DEFINE_float("eta_prec_rate_shp", default=0.3, help="eta_prec_rate prior shape")
flags.DEFINE_float("eta_prec_rate_rte", default=0.3, help="eta_prec_rate prior rate")

flags.DEFINE_float("ideal_loc", default=0.0, help="Ideal prior location")
flags.DEFINE_float("ideal_scl", default=1.0, help="Ideal prior scale")
flags.DEFINE_float("ideal_prec_shp", default=0.3, help="Ideal (unknown) precision shape")
flags.DEFINE_float("ideal_prec_rte", default=0.3, help="Ideal (unknown) precision rate")

flags.DEFINE_float("iota_loc", default=0.0, help="Iota prior location")
flags.DEFINE_float("iota_scl", default=1.0, help="Iota prior scale")
flags.DEFINE_float("iota_kappa", default=10.0, help="Kappa hyperparameter for iota precisions")
flags.DEFINE_float("iota_prec_shp", default=0.3, help="iota_prec prior shape")
flags.DEFINE_float("iota_prec_rte", default=0.3, help="iota_prec prior rate")
flags.DEFINE_float("iota_prec_rate_shp", default=0.3, help="iota_prec_rate prior shape")
flags.DEFINE_float("iota_prec_rate_rte", default=0.3, help="iota_prec_rate prior rate")
flags.DEFINE_float("iota_mean_loc", default=0.0, help="Iota_mean prior location")
flags.DEFINE_float("iota_mean_scl", default=1.0, help="Iota_mean prior scale")

FLAGS = flags.FLAGS


### Used memory
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory use:', memoryUse)


def main(argv):
    del argv
    # Hyperparameters for triple gamma prior for eta adjustments:
    prior_choice = get_and_check_prior_choice(FLAGS)
    prior_hyperparameter = get_and_check_prior_hyperparameter(FLAGS)

    tf.random.set_seed(FLAGS.seed)
    random_state = np.random.RandomState(FLAGS.seed)

    ### Setting up directories
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, 'data', FLAGS.data_name)
    fit_dir = os.path.join(source_dir, 'pf-fits')
    data_dir = os.path.join(source_dir, 'clean')
    save_dir = os.path.join(source_dir, 'fits', FLAGS.checkpoint_name)
    fig_dir = os.path.join(source_dir, 'figs', FLAGS.checkpoint_name)
    txt_dir = os.path.join(source_dir, 'txts', FLAGS.checkpoint_name)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)

    if os.path.exists(os.path.join(fig_dir, 'orig_hist_counts.png')):
        # If the descriptive histograms of the document-term matrix exist
        # --> do not plot them again, because it takes some time.
        use_fig_dir = None
    else:
        # Create the descriptive histograms of the document-term matrix
        # when loading the data within 'build_input_pipeline()'.
        use_fig_dir = fig_dir

    print("STBS analysis of " + FLAGS.data_name + "_" + FLAGS.addendum + " dataset.")

    ### Import clean datasets
    (dataset, permutation, all_author_indices, vocabulary, author_map, author_info) = build_input_pipeline(
        FLAGS.data_name, data_dir, FLAGS.batch_size, random_state, use_fig_dir, FLAGS.counts_transformation,
        FLAGS.addendum
    )
    num_documents = len(permutation)
    num_authors = author_info.shape[0]
    num_words = len(vocabulary)

    print("Number of documents: " + str(num_documents))
    print("Number of authors: " + str(num_authors))
    print("Number of words: " + str(num_words))

    ### Create the regression matrix X and initial values for ideological positions
    if prior_choice['ideal_dim'] == 'ak':
        ideal_topic_dim = FLAGS.num_topics
    elif prior_choice['ideal_dim'] == 'a':
        ideal_topic_dim = 1
    else:
        raise ValueError('Unknown selection of dimensions for ideological positions.')
    X, initial_ideal_location = create_X(FLAGS.data_name, author_info, FLAGS.covariates, ideal_topic_dim)
    print("Number of regression coefficients: " + str(X.shape[1]))

    ### Initilization by Poisson factorization
    ### Requires to run 'poisson_factorization.py' first to save document and topic shapes and rates.
    if FLAGS.pre_initialize_parameters:
        # Run 'poisson_factorization.py' first to store the initial values.
        add = str(FLAGS.num_topics) + str(FLAGS.addendum)
        fitted_document_shape = np.load(os.path.join(fit_dir, "document_shape_K" + add + ".npy")).astype(np.float32)
        fitted_document_rate = np.load(os.path.join(fit_dir, "document_rate_K" + add + ".npy")).astype(np.float32)
        fitted_topic_shape = np.load(os.path.join(fit_dir, "topic_shape_K" + add + ".npy")).astype(np.float32)
        fitted_topic_rate = np.load(os.path.join(fit_dir, "topic_rate_K" + add + ".npy")).astype(np.float32)
    else:
        fitted_document_shape = None
        fitted_document_rate = None
        fitted_topic_shape = None
        fitted_topic_rate = None

    ### Model initialization
    optim = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    model = STBS(num_documents,
                 FLAGS.num_topics,
                 num_words,
                 num_authors,
                 FLAGS.num_samples,
                 X,
                 all_author_indices,
                 initial_ideal_location=initial_ideal_location,
                 fitted_document_shape=fitted_document_shape,
                 fitted_document_rate=fitted_document_rate,
                 fitted_objective_topic_shape=fitted_topic_shape,
                 fitted_objective_topic_rate=fitted_topic_rate,
                 prior_hyperparameter=prior_hyperparameter,
                 prior_choice=prior_choice,
                 batch_size=FLAGS.batch_size,
                 RobMon_exponent=FLAGS.RobMon_exponent,
                 exact_entropy=FLAGS.exact_entropy,
                 geom_approx=FLAGS.geom_approx,
                 aux_prob_sparse=FLAGS.aux_prob_sparse,
                 iota_coef_jointly=FLAGS.iota_coef_jointly)

    ### Model training preparation
    # Add start epoch so checkpoint state is saved.
    model.start_epoch = tf.Variable(-1, trainable=False)

    if os.path.exists(checkpoint_dir) and FLAGS.load_checkpoint:
        pass
    else:
        # If we're not loading a checkpoint, overwrite the existing directory with saved results.
        if os.path.exists(save_dir):
            print("Deleting old log directory at {}".format(save_dir))
            tf.io.gfile.rmtree(save_dir)

    # We keep track of the seed to make sure the random number state is the same whether or not we load the model.
    _, seed = tfp.random.split_seed(FLAGS.seed)
    checkpoint = tf.train.Checkpoint(optimizer=optim,
                                     net=model,
                                     seed=tf.Variable(seed))
    manager = tf.train.CheckpointManager(checkpoint,
                                         checkpoint_dir,
                                         max_to_keep=1)

    checkpoint.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        # Load from saved checkpoint, keeping track of the seed.
        seed = checkpoint.seed
        # Since the dataset shuffles at every epoch and we'd like the runs to be
        # identical whether or not we load a checkpoint, we need to make sure the
        # dataset state is consistent. This is a hack but it will do for now.
        # Here's the issue: https://github.com/tensorflow/tensorflow/issues/48178
        for e in range(model.start_epoch.numpy() + 1):
            _ = iter(dataset)
            if FLAGS.computeIC_every > 0:
                if e % FLAGS.computeIC_every == 0:
                    _ = iter(dataset)

        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    summary_writer = tf.summary.create_file_writer(save_dir)
    summary_writer.set_as_default()
    start_time = time.time()
    start_epoch = model.start_epoch.numpy()

    model_state = {'ELBO': [], 'entropy': [], 'log_prior': [], 'reconstruction': [],
                   'epoch': [], 'batch': [], 'step': []}
    batches_per_epoch = len(dataset)

    if FLAGS.computeIC_every > 0:
        epoch_data = {'ELBO_MC': [], 'entropy_MC': [], 'log_prior_MC': [], 'reconstruction_MC': [],
                      'reconstruction_at_Eqmean': [], 'effective_number_of_parameters': [], 'VAIC': [], 'VBIC': [],
                      'epoch': []}
    step = 0
    for epoch in range(start_epoch + 1, FLAGS.num_epochs):
        for batch_index, batch in enumerate(iter(dataset)):
            step = batches_per_epoch * epoch + batch_index
            inputs, outputs = batch
            (total_loss, reconstruction_loss, log_prior_loss, entropy_loss, seed) = train_step(
                model, inputs, outputs, optim, seed, tf.constant(step))
            checkpoint.seed.assign(seed)
            model_state['ELBO'].append(-total_loss.numpy())
            model_state['entropy'].append(-entropy_loss.numpy())
            model_state['log_prior'].append(-log_prior_loss.numpy())
            model_state['reconstruction'].append(-reconstruction_loss.numpy())
            model_state['epoch'].append(epoch)
            model_state['batch'].append(batch_index)
            model_state['step'].append(step)

        sec_per_step = (time.time() - start_time) / (step + 1)
        sec_per_epoch = (time.time() - start_time) / (epoch - start_epoch)
        print(f"Epoch: {epoch} ELBO: {-total_loss.numpy()}")
        print(f"Entropy: {-entropy_loss.numpy()} Log-prob: {-log_prior_loss.numpy()} "
              f"Reconstruction: {-reconstruction_loss.numpy()}")
        print("({:.3f} sec/step, {:.3f} sec/epoch)".format(sec_per_step, sec_per_epoch))
        memory()

        if FLAGS.computeIC_every > 0:
            if epoch % FLAGS.computeIC_every == 0:
                ELBOstart = time.time()
                ## Using decorated @tf.function (for some reason requires too much memory)
                ELBO_MC, log_prior_MC, entropy_MC, reconstruction_MC, reconstruction_at_Eqmean, effective_number_of_parameters, VAIC, VBIC, seed = get_variational_information_criteria(
                    model, dataset, seed=seed, nsamples=FLAGS.num_samplesIC)
                ## Using a method of TBIP model
                # ELBO_MC, log_prior_MC, entropy_MC, reconstruction_MC, reconstruction_at_Eqmean, effective_number_of_parameters, VAIC, VBIC, seed = model.get_variational_information_criteria(
                #     dataset, seed=seed, nsamples=FLAGS.num_samplesIC)
                ELBOtime = time.time() - ELBOstart

                epoch_data['ELBO_MC'].append(ELBO_MC.numpy())
                epoch_data['entropy_MC'].append(entropy_MC.numpy())
                epoch_data['log_prior_MC'].append(log_prior_MC.numpy())
                epoch_data['reconstruction_MC'].append(reconstruction_MC.numpy())
                epoch_data['reconstruction_at_Eqmean'].append(reconstruction_at_Eqmean.numpy())
                epoch_data['effective_number_of_parameters'].append(effective_number_of_parameters.numpy())
                epoch_data['VAIC'].append(VAIC.numpy())
                epoch_data['VBIC'].append(VBIC.numpy())
                epoch_data['epoch'].append(epoch)

                print(f"Epoch: {epoch} ELBO: {ELBO_MC.numpy()}")
                print(
                    f"Entropy: {entropy_MC.numpy()} Log-prob: {log_prior_MC.numpy()} Reconstruction: {reconstruction_MC.numpy()}")
                print(
                    f"Reconstruction at Eqmean: {reconstruction_at_Eqmean.numpy()} Effective number of parameters: {effective_number_of_parameters.numpy()}")
                print(f"VAIC: {VAIC.numpy()} VBIC: {VBIC.numpy()}")
                print("({:.3f} sec/ELBO and IC evaluation".format(ELBOtime))

        # Log to tensorboard at the end of every `save_every` epochs.
        if epoch % FLAGS.save_every == 0:
            tf.summary.scalar("loss", total_loss, step=step)
            tf.summary.scalar("elbo/entropy", -entropy_loss, step=step)
            tf.summary.scalar("elbo/log_prior", -log_prior_loss, step=step)
            tf.summary.scalar("elbo/count_log_likelihood", -reconstruction_loss, step=step)
            tf.summary.scalar("elbo/elbo", -total_loss, step=step)
            log_static_features(model, vocabulary, author_map, step)
            summary_writer.flush()

            # Save checkpoint too.
            model.start_epoch.assign(epoch)
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

            # All model parameters can be accessed by loading the checkpoint, similar
            # to the logic at the beginning of this function. Since that may be
            # too much hassle, we also save the ideal point model parameters to a
            # separate file. You can save additional model parameters if you'd like.
            param_save_dir = os.path.join(save_dir, "params")
            if not os.path.exists(param_save_dir):
                os.makedirs(param_save_dir)
            np.save(os.path.join(param_save_dir, "all_author_indices"), model.all_author_indices.numpy())
            # theta
            pd.DataFrame(model.theta_varfam.shape.numpy()).to_csv(os.path.join(param_save_dir, "theta_shp.csv"))
            pd.DataFrame(model.theta_varfam.rate.numpy()).to_csv(os.path.join(param_save_dir, "theta_rte.csv"))
            np.save(os.path.join(param_save_dir, "theta_shp"), model.theta_varfam.shape.numpy())
            np.save(os.path.join(param_save_dir, "theta_rte"), model.theta_varfam.rate.numpy())
            if model.theta_rate_varfam.family != 'deterministic':
                pd.DataFrame(model.theta_rate_varfam.shape.numpy()).to_csv(
                    os.path.join(param_save_dir, "theta_rate_shp.csv"))
                pd.DataFrame(model.theta_rate_varfam.rate.numpy()).to_csv(
                    os.path.join(param_save_dir, "theta_rate_rte.csv"))
                np.save(os.path.join(param_save_dir, "verbosity"),
                        model.get_Eqmean(model.theta_rate_varfam, log=True).numpy())
            # beta
            pd.DataFrame(model.beta_varfam.shape.numpy()).to_csv(os.path.join(param_save_dir, "beta_shp.csv"))
            pd.DataFrame(model.beta_varfam.rate.numpy()).to_csv(os.path.join(param_save_dir, "beta_rte.csv"))
            if model.beta_rate_varfam.family != 'deterministic':
                pd.DataFrame(model.beta_rate_varfam.shape.numpy()).to_csv(
                    os.path.join(param_save_dir, "beta_rate_shp.csv"))
                pd.DataFrame(model.beta_rate_varfam.rate.numpy()).to_csv(
                    os.path.join(param_save_dir, "beta_rate_rte.csv"))
            # eta
            pd.DataFrame(model.eta_varfam.location.numpy()).to_csv(os.path.join(param_save_dir, "eta_loc.csv"))
            pd.DataFrame(model.eta_varfam.scale.numpy()).to_csv(os.path.join(param_save_dir, "eta_scl.csv"))
            if model.eta_prec_varfam.family != 'deterministic':
                pd.DataFrame(model.eta_prec_varfam.shape.numpy()).to_csv(
                    os.path.join(param_save_dir, "eta_prec_shp.csv"))
                pd.DataFrame(model.eta_prec_varfam.rate.numpy()).to_csv(
                    os.path.join(param_save_dir, "eta_prec_rte.csv"))
            # ideal
            np.save(os.path.join(param_save_dir, "ideal_point_location"), model.ideal_varfam.location.numpy())
            np.save(os.path.join(param_save_dir, "ideal_point_scale"), model.ideal_varfam.scale.numpy())
            pd.DataFrame(model.ideal_varfam.location.numpy()).to_csv(os.path.join(param_save_dir, "ideal_loc.csv"),
                                                                     index=False)
            pd.DataFrame(model.ideal_varfam.scale.numpy()).to_csv(os.path.join(param_save_dir, "ideal_scl.csv"),
                                                                  index=False)
            # regression coefficients
            if model.prior_choice["ideal_mean"] == "Nreg":
                np.save(os.path.join(param_save_dir, "iota_location"), model.iota_varfam.location.numpy())
                pd.DataFrame(model.iota_varfam.location.numpy()).to_csv(os.path.join(param_save_dir, "iota_loc.csv"),
                                                                        index=False)
                if model.iota_coef_jointly:
                    np.save(os.path.join(param_save_dir, "iota_scale_tril"), model.iota_varfam.scale_tril.numpy())
                    pd.DataFrame(model.iota_varfam.scale_tril.numpy()).to_csv(
                        os.path.join(param_save_dir, "iota_scale_tril.csv"), index=False)
                else:
                    np.save(os.path.join(param_save_dir, "iota_scale"), model.iota_varfam.scale.numpy())
                    pd.DataFrame(model.iota_varfam.scale.numpy()).to_csv(os.path.join(param_save_dir, "iota_scale.csv"),
                                                                         index=False)
            if model.iota_mean_varfam.family != 'deterministic':
                np.save(os.path.join(param_save_dir, "iota_mean_location"), model.iota_mean_varfam.location.numpy())
                np.save(os.path.join(param_save_dir, "iota_mean_scale"), model.iota_mean_varfam.scale.numpy())
            if model.exp_verbosity_varfam.family == 'lognormal':
                np.save(os.path.join(param_save_dir, "exp_verbosity_loc"), model.exp_verbosity_varfam.location.numpy())
                np.save(os.path.join(param_save_dir, "exp_verbosity_scl"), model.exp_verbosity_varfam.scale.numpy())
            elif model.exp_verbosity_varfam.family == 'gamma':
                np.save(os.path.join(param_save_dir, "exp_verbosity_shp"), model.exp_verbosity_varfam.shape.numpy())
                np.save(os.path.join(param_save_dir, "exp_verbosity_rte"), model.exp_verbosity_varfam.rate.numpy())

    ### Saving the ELBO values
    model_state = pd.DataFrame(model_state)
    model_state.to_csv(os.path.join(save_dir, 'model_state.csv'))

    ### Plotting the ELBO evolution in time plots
    for var in ['ELBO', 'entropy', 'log_prior', 'reconstruction']:
        # All steps
        plt.plot(model_state['step'], model_state[var])
        plt.ylabel(var)
        plt.xlabel('Step')
        plt.savefig(os.path.join(fig_dir, var + '.png'))
        plt.close()
        # Averages over epochs
        avg = model_state[var].to_numpy()
        avg = avg.reshape((FLAGS.num_epochs - start_epoch - 1, batches_per_epoch))
        avg = np.mean(avg, axis=1)
        plt.plot(range(start_epoch + 1, FLAGS.num_epochs), avg)
        plt.ylabel(var)
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(fig_dir, 'avg_' + var + '.png'))
        plt.close()

    if FLAGS.computeIC_every > 0:
        ### Saving the better approximation of ELBO+IC values
        epoch_data = pd.DataFrame(epoch_data)
        epoch_data.to_csv('epoch_data.csv')

        for var in ['ELBO_MC', 'entropy_MC', 'log_prior_MC', 'reconstruction_MC', 'reconstruction_at_Eqmean',
                    'effective_number_of_parameters', 'VAIC', 'VBIC']:
            plt.plot(epoch_data['epoch'], epoch_data[var])
            plt.ylabel(var)
            plt.xlabel('Epoch')
            # plt.show()
            plt.savefig(os.path.join(fig_dir, var + '.png'))
            plt.close()

    ### Other figures
    create_all_general_descriptive_figures(model, fig_dir, author_map, vocabulary,
                                           nwords_eta_beta_vs_ideal=20, nwords=20, ntopics=5)
    create_all_figures_specific_to_data(model, FLAGS.data_name, FLAGS.covariates, fig_dir, all_author_indices,
                                        author_map, author_info, vocabulary,
                                        nwords=10, ntopics=5,
                                        selected_topics=[5, 9, 11, 13, 15])

    ### Top influential speeches for each topic
    if FLAGS.num_top_speeches > 0:
        find_most_influential_speeches(model, FLAGS.data_name, data_dir, source_dir, txt_dir, FLAGS.addendum,
                                       FLAGS.how_influential, FLAGS.batch_size, FLAGS.num_top_speeches)


if __name__ == '__main__':
    app.run(main)
