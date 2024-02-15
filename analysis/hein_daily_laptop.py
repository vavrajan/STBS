### Generate some example to use TBIP on

# Import global packages
import os
import time

import pandas as pd
from absl import app
from absl import flags

import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
import matplotlib.pyplot as plt


# Import local modules
from STBIP.code.var_and_prior_family import VariationalFamily, PriorFamily
from STBIP.code.train_step import train_step
from STBIP.code.utils import print_topics, print_ideal_points, log_static_features
from STBIP.code.plotting_functions import create_all_general_descriptive_figures, create_all_figures_specific_to_data
from STBIP.code.input_pipeline import build_input_pipeline
from STBIP.code.stbip_model import STBIP
from STBIP.code.create_X import create_X

from sklearn.feature_extraction.text import CountVectorizer


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
#
# ### Classical TBIP setting
# prior_choice = {
#     "theta": "Gfix",        # Gfix=Gamma fixed, Gdrte=Gamma d-rates, Garte=Gamma a-rates
#     "exp_verbosity": "LNfix",# None=deterministic 1, LNfix=LogN(0,1), Gfix=Gamma fixed
#     "beta": "Gfix",         # Gfix=Gamma fixed, Gvrte=Gamma v-rates
#     "eta": "Nfix",          # Nfix=N(0,1),
#                             # NkprecG=N(0,1/eta_prec_k) and eta_prec_k=Gfix,
#                             # NkprecF=N(0,1/eta_prec_k) and eta_prec_k=G(.,eta_prec_rate_k)
#     "ideal_dim": "a",       # "ak" - author and topic-specific ideological positions, "a" just author-specific locations
#     "ideal_mean": "Nfix",   # Nfix=N(0,.), Nreg=author-level regression N(x^T * iota, .)
#     "ideal_prec": "Nfix",   # Nfix=N(.,1), Nprec=N(.,1/ideal_prec), Naprec=N(., 1/ideal_prec_{author})
#     "iota_dim": "l",        # "lk" - coefficient and topic-specific ideological positions (cannot when ideal_dim="a"), "l" just coefficient-specific locations
#     "iota_prec": "None",    # Nfix=N(.,1),
#                             # NlprecG=N(.,1/iota_prec_l) and iota_prec_l=Gfix,,
#                             # NlprecF=N(.,1/iota_prec_l) and iota_prec_l=G(.,iota_prec_rate_l),
#                             # None=iotas do not exist in the model (if ideal_mean=="Nfix")
#     "iota_mean": "None",    # None=iotas are centred to a fixed value, Nlmean=each regressor has its own mean across all topics
# }

### Another setting
prior_choice = {
    "theta": "Garte",       # Gfix=Gamma fixed, Gdrte=Gamma d-rates, Garte=Gamma a-rates
    "exp_verbosity": "None",# None=deterministic 1, LNfix=LogN(0,1), Gfix=Gamma fixed
    "beta": "Gvrte",        # Gfix=Gamma fixed, Gvrte=Gamma v-rates
    "eta": "NkprecF",       # Nfix=N(0,1),
                            # NkprecG=N(0,1/eta_prec_k) and eta_prec_k=Gfix,
                            # NkprecF=N(0,1/eta_prec_k) and eta_prec_k=G(.,eta_prec_rate_k)
    "ideal_dim": "ak",      # "ak" - author and topic-specific ideological positions, "a" just author-specific locations
    "ideal_mean": "Nreg",   # Nfix=N(0,.), Nreg=author-level regression N(x^T * iota, .)
    "ideal_prec": "Naprec", # Nfix=N(.,1), Nprec=N(.,1/ideal_prec), Naprec=N(., 1/ideal_prec_{author})
    # "iota_dim": "lk",       # "lk" - coefficient and topic-specific ideological positions (cannot when ideal_dim="a"), "l" just coefficient-specific locations
    "iota_dim": "kl",       # "kl" - coefficient and topic-specific ideological positions (cannot when ideal_dim="a"), "l" just coefficient-specific locations
    "iota_prec": "NlprecF", # Nfix=N(.,1),
                            # NlprecG=N(.,1/iota_prec_l) and iota_prec_l=Gfix,,
                            # NlprecF=N(.,1/iota_prec_l) and iota_prec_l=G(.,iota_prec_rate_l),
                            # None=iotas do not exist in the model (if ideal_mean=="Nfix")
    "iota_mean": "Nlmean",  # None=iotas are centred to a fixed value, Nlmean=each regressor has its own mean across all topics
}

# Initial random seed for parameter initialization.
seed = 314159
data_name = "hein-daily"
addendum = "114"
addendum = '97'
checkpoint_name = "first_use"
load_checkpoint = False
batch_size = 64
counts_transformation = "nothing"
learning_rate = 0.01
num_topics = 30
num_samples = 1
exact_entropy = True
geom_approx = False
aux_prob_sparse = False
num_epochs = 2
save_every = 1
covariates = "party"

# data_name = 'fomc'
# addendum = ''
# num_topics = 2
# covariates = 'gender+title+year+flength+flaughter'
# covariates = 'None'
# prior_choice['ideal_mean'] = 'Nfix'
# prior_choice['ideal_prec'] = 'Nfix'
# prior_choice['iota_dim'] = 'l'
# prior_choice['iota_prec'] = 'Nfix'
# prior_choice['iota_mean'] = 'None'


tf.random.set_seed(seed)
random_state = np.random.RandomState(seed)
#project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
#source_dir = os.path.join(project_dir, "TBIP_colab\\data\\{}".format(data_name))
#project_dir = 'home/jvavra/TBIP/'
### My laptop
project_dir = 'C:\\Users\\jvavra\\OneDrive - WU Wien\\Documents\\TBIP_colab'
source_dir = os.path.join(project_dir, "data\\{}".format(data_name))
### Cluster
# project_dir = ''
# project_dir = os.getcwd()
# source_dir = project_dir+'/data/'+data_name+'/'

# As described in the docstring, the data directory must have the following
# files: counts.npz, author_indices.npy, vocabulary.txt, author_map.txt.
### My laptop
data_dir = os.path.join(source_dir, "clean")
save_dir = os.path.join(source_dir, "fits\\{}".format(checkpoint_name))
fig_dir = os.path.join(source_dir, "figs\\{}".format(checkpoint_name))
### Cluster
# data_dir = source_dir+'clean/'
# save_dir = source_dir+'fits/'+checkpoint_name+'/'
# fig_dir = source_dir+'figs/'+checkpoint_name+'/'
#
# if not(os.path.exists(fig_dir)):
#     # The fig path does not exist --> create it
#     os.mkdir(fig_dir)
if os.path.exists(fig_dir + 'orig_log_hist_counts.png'):
    use_fig_dir = None
else:
    use_fig_dir = fig_dir

(dataset, permutation, all_author_indices, vocabulary, author_map, author_info) = build_input_pipeline(
        data_name, data_dir, batch_size, random_state, use_fig_dir, counts_transformation, addendum
)
num_documents = len(permutation)
num_words = len(vocabulary)
num_authors = author_info.shape[0]

# Import infodata about senators.
# author_info = np.load(os.path.join(data_dir, "author_info" + FLAGS.addendum + ".npy"))
# author_info = pd.read_csv(os.path.join(data_dir, "author_info" + addendum + ".csv"))
# author_info = pd.read_csv(os.path.join(data_dir, "author_detailed_info" + addendum + ".csv"))
if prior_choice['ideal_dim'] == 'ak':
    ideal_topic_dim = num_topics
elif prior_choice['ideal_dim'] == 'a':
    ideal_topic_dim = 1
else:
    raise ValueError('Unknown selection of dimensions for ideological positions.')
X, initial_ideal_location = create_X(data_name, author_info, covariates, ideal_topic_dim)

fitted_document_shape = None
fitted_document_rate = None
fitted_topic_shape = None
fitted_topic_rate = None

optim = tf.optimizers.Adam(learning_rate=learning_rate)

model = STBIP(num_documents,
              num_topics,
              num_words,
              num_authors,
              num_samples,
              X,
              all_author_indices,
              initial_ideal_location=initial_ideal_location,
              prior_hyperparameter=prior_hyperparameter,
              prior_choice=prior_choice,
              batch_size=batch_size,
              exact_entropy=exact_entropy,
              geom_approx=geom_approx,
              aux_prob_sparse=aux_prob_sparse,
              iota_coef_jointly=True)



# Add start epoch so checkpoint state is saved.
model.start_epoch = tf.Variable(-1, trainable=False)

checkpoint_dir = os.path.join(save_dir, "checkpoints")
if os.path.exists(checkpoint_dir) and load_checkpoint:
    pass
else:
    # If we're not loading a checkpoint, overwrite the existing directory with saved results.
    if os.path.exists(save_dir):
        print("Deleting old log directory at {}".format(save_dir))
        tf.io.gfile.rmtree(save_dir)

# We keep track of the seed to make sure the random number state is the same whether or not we load a model.
_, seed = tfp.random.split_seed(seed)
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
    for _ in range(model.start_epoch.numpy() + 1):
        _ = iter(dataset)
        _ = iter(dataset) # second one for more elaborate evaluation of the ELBO
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

summary_writer = tf.summary.create_file_writer(save_dir)
summary_writer.set_as_default()
start_time = time.time()
start_epoch = model.start_epoch.numpy()

model_state = {'ELBO': [], 'entropy': [], 'log_prior': [], 'reconstruction': [], 'epoch': [], 'batch': [], 'step': []}
epoch_data = {'ELBO_MC': [], 'entropy_MC': [], 'log_prior_MC': [], 'reconstruction_MC': [],
              'reconstruction_at_Eqmean': [], 'effective_number_of_parameters': [], 'VAIC': [], 'VBIC': [],
              'epoch': []}

for epoch in range(start_epoch + 1, num_epochs):
    for batch_index, batch in enumerate(iter(dataset)):
        batches_per_epoch = len(dataset)
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
    print(f"Entropy: {-entropy_loss.numpy()} Log-prob: {-log_prior_loss.numpy()} Reconstruction: {-reconstruction_loss.numpy()}")
    print("({:.3f} sec/step, {:.3f} sec/epoch)".format(sec_per_step, sec_per_epoch))
    # print("Epoch: {:>3d} ELBO: {:.3f} Entropy: {:.1f} ({:.3f} sec/step, "
    #       "{:.3f} sec/epoch)".format(epoch,
    #                                  -total_loss.numpy(),
    #                                  -entropy_loss.numpy(),
    #                                  sec_per_step,
    #                                  sec_per_epoch))
    ELBOstart = time.time()
    ELBO_MC, log_prior_MC, entropy_MC, reconstruction_MC, reconstruction_at_Eqmean, effective_number_of_parameters, VAIC, VBIC, seed = model.get_variational_information_criteria(dataset, seed=seed, nsamples=num_samples)
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
    print(f"Entropy: {entropy_MC.numpy()} Log-prob: {log_prior_MC.numpy()} Reconstruction: {reconstruction_MC.numpy()}")
    print(f"Reconstruction at Eqmean: {reconstruction_at_Eqmean.numpy()} Effective number of parameters: {effective_number_of_parameters.numpy()}")
    print(f"VAIC: {VAIC.numpy()} VBIC: {VBIC.numpy()}")
    print("({:.3f} sec/ELBO and IC evaluation".format(ELBOtime))


    # Log to tensorboard at the end of every `save_every` epochs.
    if epoch % save_every == 0:
        tf.summary.scalar("loss", total_loss, step=step)
        tf.summary.scalar("elbo/entropy", -entropy_loss, step=step)
        tf.summary.scalar("elbo/log_prior", -log_prior_loss, step=step)
        tf.summary.scalar("elbo/count_log_likelihood", -reconstruction_loss, step=step)
        tf.summary.scalar("elbo/elbo", -total_loss, step=step)
        log_static_features(model, vocabulary, author_map, step)
        summary_writer.flush()

        # Save checkpoint too.
        model.start_epoch.assign(epoch)
        #model.save(save_dir+"model.keras", save_format="tf")
        save_path = manager.save() #Value Error: _shared_name return self.name[:self.name.index(":")] ValueError: substring not found
        print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

        # All model parameters can be accessed by loading the checkpoint, similar
        # to the logic at the beginning of this function. Since that may be
        # too much hassle, we also save the ideal point model parameters to a
        # separate file. You can save additional model parameters if you'd like.
        param_save_dir = os.path.join(save_dir, "params/")
        if not os.path.exists(param_save_dir):
            os.makedirs(param_save_dir)
        np.save(os.path.join(param_save_dir, "ideal_point_location_"+str(epoch)), model.ideal_varfam.location.numpy())
        np.save(os.path.join(param_save_dir, "ideal_point_scale_"+str(epoch)), model.ideal_varfam.scale.numpy())
        np.save(os.path.join(param_save_dir, "eta_location_" + str(epoch)), model.eta_varfam.location.numpy())
        np.save(os.path.join(param_save_dir, "eta_scale_" + str(epoch)), model.eta_varfam.scale.numpy())
        # theta
        df = pd.DataFrame(model.theta_varfam.shape.numpy())
        df.to_csv(os.path.join(param_save_dir, "theta_shp.csv"))
        df = pd.DataFrame(model.theta_varfam.rate.numpy())
        df.to_csv(os.path.join(param_save_dir, "theta_rte.csv"))

model_state=pd.DataFrame(model_state)
model_state.to_csv('model_state.csv')

epoch_data=pd.DataFrame(epoch_data)
epoch_data.to_csv('epoch_data.csv')

for var in ['ELBO', 'entropy', 'log_prior', 'reconstruction']:
    # All steps
    plt.plot(model_state['step'], model_state[var])
    plt.ylabel(var)
    plt.xlabel('Step')
    # plt.show()
    plt.savefig(os.path.join(fig_dir, var+'.png'))
    plt.close()
    # Averages over epochs
    avg = model_state[var].to_numpy()
    avg = avg.reshape((num_epochs-start_epoch-1, batches_per_epoch))
    avg = np.mean(avg, axis=1)
    plt.plot(range(start_epoch+1, num_epochs), avg)
    plt.ylabel(var)
    plt.xlabel('Epoch')
    # plt.show()
    plt.savefig(os.path.join(fig_dir, 'avg_'+var+'.png'))
    plt.close()

for var in ['ELBO_MC', 'entropy_MC', 'log_prior_MC', 'reconstruction_MC', 'reconstruction_at_Eqmean',
            'effective_number_of_parameters', 'VAIC', 'VBIC']:
    plt.plot(epoch_data['epoch'], epoch_data[var])
    plt.ylabel(var)
    plt.xlabel('Epoch')
    # plt.show()
    plt.savefig(os.path.join(fig_dir, var+'.png'))
    plt.close()

### Other figures
create_all_general_descriptive_figures(model, fig_dir, author_map, vocabulary,
                                           nwords_eta_beta_vs_ideal=20, nwords=10, ntopics=5,
                                           selected_topics=[5, 9, 11, 13, 15])
create_all_figures_specific_to_data(model, data_name, covariates, fig_dir, all_author_indices,
                                    author_map, author_info, vocabulary,
                                    nwords=10, ntopics=5)
