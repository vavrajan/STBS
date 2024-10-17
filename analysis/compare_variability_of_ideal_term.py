import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.join(os.getcwd(), 'code'))
from plotting_functions import plot_labels

### Setting up directories
data_name = 'hein-daily'
addendum = '114'
num_topics = 25
covariates = 'all'

project_dir = os.getcwd()
source_dir = os.path.join(project_dir, 'data', data_name)
data_dir = os.path.join(source_dir, 'clean')
ak_dir = os.path.join(source_dir, 'fits', 'STBS_ideal_ak_' + covariates + addendum + '_K'+str(num_topics), 'params')
a_dir = os.path.join(source_dir, 'fits', 'STBS_ideal_a_' + covariates + addendum + '_K'+str(num_topics), 'params')
fig_dir = os.path.join(source_dir, 'figs')

### Load rho variances
ak_rho_shp = pd.read_csv(os.path.join(ak_dir, "eta_prec_shp.csv")).to_numpy()[:, 1]
ak_rho_rte = pd.read_csv(os.path.join(ak_dir, "eta_prec_rte.csv")).to_numpy()[:, 1]
a_rho_shp = pd.read_csv(os.path.join(a_dir, "eta_prec_shp.csv")).to_numpy()[:, 1]
a_rho_rte = pd.read_csv(os.path.join(a_dir, "eta_prec_rte.csv")).to_numpy()[:, 1]

### Load ideological positions
ideal_ak_loc = pd.read_csv(os.path.join(ak_dir, "ideal_loc.csv")).to_numpy()
ideal_a_loc = pd.read_csv(os.path.join(a_dir, "ideal_loc.csv")).to_numpy()

### Load polarities eta
eta_ak_loc = pd.read_csv(os.path.join(ak_dir, "eta_loc.csv")).to_numpy()
eta_a_loc = pd.read_csv(os.path.join(a_dir, "eta_loc.csv")).to_numpy()

term_ak = tf.constant(eta_ak_loc)[tf.newaxis, :, :] * tf.constant(ideal_ak_loc)[:, :, tf.newaxis]
term_a = tf.constant(eta_a_loc)[tf.newaxis, :, :] * tf.constant(ideal_a_loc)[:, :, tf.newaxis]

Fip = 'Fixed ideological position'
Tsip = 'Topic-specific ideological position'

Eq_rho_to_var = {
    Fip: a_rho_rte / (a_rho_shp - 1.0),
    Tsip: ak_rho_rte / (ak_rho_shp - 1.0),
}

ideal_variability = {
    Fip: np.repeat(np.var(ideal_a_loc, axis=0), num_topics),
    Tsip: np.var(ideal_ak_loc, axis=0),
}

rho_ideal_variability = {
    Fip: Eq_rho_to_var[Fip] * ideal_variability[Fip],
    Tsip: Eq_rho_to_var[Tsip] * ideal_variability[Tsip],
}

eta_ideal_variability = {
    Fip: tf.math.reduce_variance(term_a, axis=[0, 2]),
    Tsip: tf.math.reduce_variance(term_ak, axis=[0, 2]),
}

pd.DataFrame(eta_ideal_variability[Fip].numpy()).to_csv(os.path.join(a_dir, "eta_ideal_a_variability.csv"))
pd.DataFrame(eta_ideal_variability[Tsip].numpy()).to_csv(os.path.join(ak_dir, "eta_ideal_ak_variability.csv"))

x = np.arange(num_topics)  # the label locations
width = 0.25  # the width of the bars
multiplier = -0.5

fig, ax = plt.subplots(layout='constrained', figsize=(10, 5))
for attribute, measurement in Eq_rho_to_var.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width)
    # ax.bar_label(rects, padding=3)
    multiplier += 1
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Variance')
ax.set_xlabel('Topic')
ax.set_title('Eta variances')
ax.legend(Eq_rho_to_var.keys(), loc='upper center', ncols=2)
ax.set_ylim(0, 1.2)
# plt.show()
plt.savefig(os.path.join(fig_dir, 'Eq_rho_to_var_a_vs_ak.png'))

multiplier = -0.5
fig, ax = plt.subplots(layout='constrained', figsize=(10, 5))
for attribute, measurement in ideal_variability.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width)
    # ax.bar_label(rects, padding=3)
    multiplier += 1
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Variance')
ax.set_xlabel('Topic')
ax.set_title('Variability of ideological positions')
ax.legend(ideal_variability.keys(), loc='upper center', ncols=2)
ax.set_ylim(0, 1.2)
# plt.show()
plt.savefig(os.path.join(fig_dir, 'ideal_variability_a_vs_ak.png'))

multiplier = -0.5
fig, ax = plt.subplots(layout='constrained', figsize=(10, 5))
for attribute, measurement in rho_ideal_variability.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width)
    # ax.bar_label(rects, padding=3)
    multiplier += 1
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Variance')
ax.set_xlabel('Topic')
ax.set_title('Variability of the ideological term')
ax.legend(rho_ideal_variability.keys(), loc='upper center', ncols=2)
ax.set_ylim(0, 1.2)
# plt.show()
plt.savefig(os.path.join(fig_dir, 'rho_ideal_variability_a_vs_ak.png'))

multiplier = -0.5
fig, ax = plt.subplots(layout='constrained', figsize=(10, 5))
for attribute, measurement in eta_ideal_variability.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width)
    # ax.bar_label(rects, padding=3)
    multiplier += 1
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Variance')
ax.set_xlabel('Topic')
ax.set_title('Variability of the ideological term')
ax.legend(eta_ideal_variability.keys(), loc='upper center', ncols=2)
ax.set_ylim(0, 0.15)
# plt.show()
plt.savefig(os.path.join(fig_dir, 'eta_ideal_variability_a_vs_ak.png'))