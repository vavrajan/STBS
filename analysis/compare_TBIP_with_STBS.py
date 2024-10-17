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
STBS = 'STBS_ideal_ak_party'
# STBS = 'STBS_ideal_ak_all_no_int'
# STBS = 'STBS_ideal_ak_all'
party_by_mean = True # if False, then party means are determined based on the estimated regression coefficients
weighted_mean = True # weights created by averaging thetas

project_dir = os.getcwd()
source_dir = os.path.join(project_dir, 'data', data_name)
data_dir = os.path.join(source_dir, 'clean')
fit_dir = os.path.join(source_dir, 'fits')
TBIP_dir = os.path.join(fit_dir, 'TBIP_' + addendum + '_K'+str(num_topics), 'params')
STBS_dir = os.path.join(fit_dir, STBS + addendum + '_K'+str(num_topics), 'params')
fig_dir = os.path.join(source_dir, 'figs')

tab_dir = os.path.join(source_dir, 'tabs')
if not os.path.exists(tab_dir):
    os.mkdir(tab_dir)

### Load verbosities, ideal points, iotas and thetas
TBIPverb = np.load(os.path.join(TBIP_dir, "exp_verbosity_loc.npy"))
STBSverb = np.load(os.path.join(STBS_dir, "verbosity.npy"))

TBIPloc = np.load(os.path.join(TBIP_dir, "ideal_point_location.npy"))
STBSloc = np.load(os.path.join(STBS_dir, "ideal_point_location.npy"))

TBIPscl = np.load(os.path.join(TBIP_dir, "ideal_point_scale.npy"))
STBSscl = np.load(os.path.join(STBS_dir, "ideal_point_scale.npy"))

iotas = np.load(os.path.join(STBS_dir, "iota_location.npy"))

STBS_theta_shp = np.load(os.path.join(STBS_dir, "theta_shp.npy"))
STBS_theta_rte = np.load(os.path.join(STBS_dir, "theta_rte.npy"))

all_author_indices = np.load(os.path.join(STBS_dir, "all_author_indices.npy"))

# author_info = pd.read_csv(os.path.join(data_dir, "author_info114.csv"))
# author_info = pd.read_csv(os.path.join(data_dir, "author_detailed_info114.csv"))
author_info = pd.read_csv(os.path.join(data_dir, "author_detailed_info_with_religion114.csv"))
file = open(os.path.join(data_dir, "author_map114.txt"))
lines = file.readlines()
author = np.array(lines)
author_map = np.char.replace(author, '\n', '')
num_authors = len(author_map)

# correlations = [tfp.stats.correlation(tf.gather(TBIPloc, 0, axis=1), tf.gather(STBSloc, k, axis=1)) for k in range(num_topics)]
correlations = tfp.stats.correlation(tf.repeat(TBIPloc, num_topics, axis=1), STBSloc)[0]
correlations = tf.math.round(100*correlations) / 100
# correlations = tfp.stats.correlation(TBIPloc[:, 0], STBSloc)

with open(os.path.join(tab_dir, 'ideal_points_TBIP_vs_'+STBS+'.tex'), 'w') as file:
    #file.write(str(correlations) + '\n')
    file.write('\\begin{tabular}{ccr}\n')
    file.write('Topic & Ideological positions & Correlation \\\\\n')
    file.write('\\toprule\n')
    file.write(' & \\includegraphics[]{fig/ideal_points.png} & \\\\\n')
    file.write('\\midrule\n')
    for k in range(num_topics):
        file.write(str(k) + ' & \\includegraphics[]{fig/ideal_points_' + str(k) + '.png} & ' + str(correlations.numpy()[k]) + ' \\\\\n')
    file.write('\\end{tabular}\n')


### Plotting verbosities against each other
author_category = author_info['party']
path = os.path.join(fig_dir, 'verbosities_TBIP_vs_Garte_K'+str(num_topics)+'.png')
title = 'Comparison of verbosities'
# labels = [author_map[i] for i in range(len(STBSverb))]
labels = [author_info['surname'][i].title() for i in range(len(STBSverb))]
colorcat = {"R": "red", "D": "blue", "I": "grey"}
colors = [colorcat[author_category[i]] for i in range(len(STBSverb))]
size = (10, 6.5)
plot_labels(TBIPverb, STBSverb, labels, path=path, title=title, colors=colors, size=size,
            xlab='Classical TBIP verbosity term', ylab='Theta rates on log-scale')

def plot_heatmap(table, title, fig_path, ind0, ind1):
    # Define the plot
    fig, ax = plt.subplots(figsize=(13, 7))
    # Set the font size and the distance of the title from the plot
    plt.title(title, fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5, 1.05])
    table = tf.gather(table, ind0, axis=0)
    table = tf.gather(table, ind1, axis=1)
    # Hide ticks for X & Y axis
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove the axes
    ax.axis('off')
    # Use the heatmap function from the seaborn package
    sns.heatmap(table, fmt="", cmap='plasma', linewidths=0.30, ax=ax)
    # Display the heatmap
    # plt.show()
    # Save the heatmap
    plt.gcf().set_size_inches((10, 13))
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

### Plotting ideal points for classical TBIP and then topic-specific ideals
Eqtheta = STBS_theta_shp / STBS_theta_rte
weights = tf.math.unsorted_segment_mean(Eqtheta, all_author_indices, num_authors)
weights_row = weights / tf.reduce_sum(weights, axis=0)[tf.newaxis, :]
weights_column = weights / tf.reduce_sum(weights, axis=1)[:, tf.newaxis]
weights_total = weights / tf.reduce_sum(weights)
ind0 = range(0, num_authors)
ind1 = range(0, num_topics)
plot_heatmap(weights_row, "Row-wise scaled weights", os.path.join(fig_dir, 'weights_row.png'), ind0, ind1)
plot_heatmap(weights_column, "Column-wise scaled weights", os.path.join(fig_dir, 'weights_column.png'), ind0, ind1)
plot_heatmap(weights_total, "Total scaled weights", os.path.join(fig_dir, 'weights_total.png'), ind0, ind1)
# reorganization by categories
#ind0 = author_info['party'].argsort()
# print(ind0)
# print(tf.gather(author_map, ind0, axis=0))
#plot_heatmap(weights_total, "Weights grouped by party", fig_dir + 'weights_by_party.png', ind0, ind1)

for cat in ['party', 'gender', 'region', 'generation', 'exper_cong', 'RELIGION']:
    ind0 = author_info[cat].argsort()
    plot_heatmap(weights_total, "Weights grouped by "+cat, os.path.join(fig_dir, 'weights_by_'+cat+'.png'), ind0, ind1)

for transposed in [True, False]:
    if transposed:
        label_t = "_transposed"
    else:
        label_t = ""
    # weights.shape: [num_authors, num_topics]
    if party_by_mean:
        if weighted_mean:
            # weighing ideal locations:
            wloc = STBSloc * weights_row
            # Party means
            xD = tf.reduce_sum(tf.gather(wloc, tf.where(author_info['party'] == 'D'), axis=0), axis=0)[0, :]
            xR = tf.reduce_sum(tf.gather(wloc, tf.where(author_info['party'] == 'R'), axis=0), axis=0)[0, :]
            how_party = 'party_by_weighted_average'
        else:
            # Party means
            xD = tf.reduce_mean(tf.gather(STBSloc, tf.where(author_info['party'] == 'D'), axis=0), axis=0)[0, :]
            xR = tf.reduce_mean(tf.gather(STBSloc, tf.where(author_info['party'] == 'R'), axis=0), axis=0)[0, :]
            how_party = 'party_by_average'
    else:
        xD = iotas[:, 0]
        xR = iotas[:, 0] + iotas[:, 1]
        how_party = 'party_by_iota'

    xD1 = tf.reduce_mean(tf.gather(TBIPloc, tf.where(author_info['party'] == 'D'), axis=0), axis=0)
    xR1 = tf.reduce_mean(tf.gather(TBIPloc, tf.where(author_info['party'] == 'R'), axis=0), axis=0)
    # ... or even better, predicted by regression coefficients
    permutation = tf.argsort(tf.math.abs(xR - xD), direction='DESCENDING')
    xD = tf.gather(xD, permutation)
    xR = tf.gather(xR, permutation)
    markercat = {"R": "o", "D": "o", "I": "+"}
    #colorcat = {"R": "orange", "D": "lightblue", "I": "grey"}
    colorcat = {"R": "tomato", "D": "cornflowerblue", "I": "grey"}
    topics = -tf.range(num_topics)-1
    # topicslab = tf.range(num_topics)+1
    # topicslab = topicslab.numpy()
    # topicslab = topicslab.astype(str)
    topicslab = permutation.numpy().astype(str)
    if transposed:
        plt.hlines(0.0, -num_topics-1, 1.5, colors='grey', linestyles='--')
        for i in range(TBIPloc.shape[0]):
            plt.scatter(y=TBIPloc[i], x=1.0, color=colorcat[author_info['party'][i]], marker=markercat[author_info['party'][i]])
        plt.vlines(0.5, -1.4, 1.9, colors='grey', linestyles='-')
    else:
        plt.vlines(0.0, -num_topics - 1, 1.5, colors='grey', linestyles='--')
        for i in range(TBIPloc.shape[0]):
            plt.scatter(x=TBIPloc[i], y=1.0, color=colorcat[author_info['party'][i]], marker=markercat[author_info['party'][i]])
        plt.hlines(0.5, -1.4, 1.9, colors='grey', linestyles='-')
    if weighted_mean:
        meanlocs = tf.reduce_sum(STBSloc * weights_column, axis=1)
        if transposed:
            meanlab = 'WA'
        else:
            meanlab = 'WghtAvrg'
    else:
        meanlocs = tf.reduce_mean(STBSloc, axis=1)
        if transposed:
            meanlab = 'Avg'
        else:
            meanlab = 'Averaged'
    correlation = tfp.stats.correlation(TBIPloc, meanlocs[:, tf.newaxis])[0]
    xD0 = tf.reduce_mean(tf.gather(meanlocs, tf.where(author_info['party'] == 'D'), axis=0), axis=0)
    xR0 = tf.reduce_mean(tf.gather(meanlocs, tf.where(author_info['party'] == 'R'), axis=0), axis=0)
    locs = tf.gather(STBSloc, permutation, axis=1)
    if transposed:
        for i in range(TBIPloc.shape[0]):
            plt.scatter(y=meanlocs[i], x=0.0, color=colorcat[author_info['party'][i]], marker=markercat[author_info['party'][i]])
        plt.vlines(-0.5, -1.4, 1.9, colors='grey', linestyles='-')
        for i in range(TBIPloc.shape[0]):
            plt.scatter(y=locs[i, :], x=topics,
                        color=colorcat[author_info['party'][i]], marker=markercat[author_info['party'][i]])
        # plt.axis(False)
        plt.xticks(np.append(topics.numpy(), [0, 1]), np.append(topicslab, [meanlab, 'TBIP']))
        plt.ylim((-1.5, 2.0))
        plt.ylabel('Ideological position')
    else:
        for i in range(TBIPloc.shape[0]):
            plt.scatter(x=meanlocs[i], y=0.0, color=colorcat[author_info['party'][i]], marker=markercat[author_info['party'][i]])
        plt.hlines(-0.5, -1.4, 1.9, colors='grey', linestyles='-')
        for i in range(TBIPloc.shape[0]):
            plt.scatter(x=locs[i, :], y=topics,
                        color=colorcat[author_info['party'][i]], marker=markercat[author_info['party'][i]])
        #plt.axis(False)
        plt.yticks(np.append(topics.numpy(), [0, 1]), np.append(topicslab, [meanlab, 'TBIP']))
        plt.xlim((-1.5, 2.0))
        plt.xlabel('Ideological position')
    plt.box(False)
    plt.margins(x=0, y=0)
    ## Correlation between TBIP and averaged positions
    cor = tf.round(1e3*correlation).numpy() / 1e3
    label = "Pearson corr.: " + str(cor[0])
    if transposed:
        plt.text(-2.8, 1.8, label, style='italic', bbox={'facecolor': 'white', 'pad': 10})
    else:
        plt.text(1.2, 0.5, label, style='italic', bbox={'facecolor': 'white', 'pad': 10})
    # ## Taking some specific authors
    # # Chuck Schumer (D) and Mitch McConnel (R)
    # colparty = {'I': 'grey', 'D': 'blue', 'R': 'red'}
    # # print(author_info['party'])
    # for a in ['Charles Schumer (D)', 'Mitch Mcconnell (R)']:
    #     i = tf.where(author_map == a)[0]
    #     # print(i)
    #     # print(author_info['party'][i])
    #     ideala1 = tf.reshape(tf.gather(TBIPloc, i, axis=0), (1))
    #     ideala2 = tf.reshape(tf.gather(meanlocs, i, axis=0), (1))
    #     ideala3 = tf.reshape(tf.gather(locs, i, axis=0), (num_topics))
    #     ideala = np.append([ideala1.numpy(), ideala2.numpy()], ideala3.numpy())
    #     k = np.append(np.array([1, 0]), topics)
    #     plt.plot(ideala, k, linestyle='-', color=colparty[author_info['party'][i].values[0]])
    # Party means
    if transposed:
        plt.vlines(topics, xD, xR, colors='black')
        plt.scatter(y=[xD, xR], x=[topics, topics],
                    color=np.append(np.repeat('darkblue', num_topics), np.repeat('darkred', num_topics)),
                    marker='x')
        plt.vlines(0, xD0, xR0, colors='black')
        plt.scatter(y=[xD0, xR0], x=[0, 0], color=['darkblue', 'darkred'], marker='x')
        plt.vlines(1, xD1, xR1, colors='black')
        plt.scatter(y=[xD1, xR1], x=[1, 1], color=['darkblue', 'darkred'], marker='x')
        plt.gcf().set_size_inches((11, 5))
    else:
        plt.hlines(topics, xD, xR, colors='black')
        plt.scatter([xD, xR], [topics, topics],
                    color=np.append(np.repeat('darkblue', num_topics), np.repeat('darkred', num_topics)),
                    marker='x')
        plt.hlines(0, xD0, xR0, colors='black')
        plt.scatter([xD0, xR0], [0, 0], color=['darkblue', 'darkred'], marker='x')
        plt.hlines(1, xD1, xR1, colors='black')
        plt.scatter([xD1, xR1], [1, 1], color=['darkblue', 'darkred'], marker='x')
        plt.gcf().set_size_inches((7.5, 8))

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(fig_dir, 'ideal_points_'+STBS+'_vs_TBIP_'+how_party+label_t+'_K'+str(num_topics)+'.png'),
                bbox_inches='tight')
    plt.close()



### Mixture densities instead of dots
party_by_mean = False
if party_by_mean:
    # Party means
    xD = tf.reduce_mean(tf.gather(STBSloc, tf.where(author_info['party'] == 'D'), axis=0), axis=0)[0, :]
    xR = tf.reduce_mean(tf.gather(STBSloc, tf.where(author_info['party'] == 'R'), axis=0), axis=0)[0, :]
    how_party = 'party_by_average'
else:
    xD = iotas[:, 0]
    xR = iotas[:, 0] + iotas[:, 1]
    how_party = 'party_by_iota'
permutation = tf.argsort(tf.math.abs(xR - xD), direction='DESCENDING')
xD = tf.gather(xD, permutation)
xR = tf.gather(xR, permutation)
darkcolorcat = {"R": "red", "D": "blue", "I": "black"}
colorcat = {"R": "orange", "D": "lightblue", "I": "grey"}
xmin = -2.5
xmax = 2.5
xdif = (xmax - xmin) / 500
xs = tf.range(xmin, xmax + xdif, delta=xdif)
plt.vlines(0.0, -num_topics, 2, colors='grey', linestyles='--')
for key in ['R', 'D']: #colorcat.keys():
    if key == 'R':
        currentx = xR
    else:
        currentx = xD
    ### classical TBIP
    TBIPlockey = tf.gather(TBIPloc, tf.where(author_info['party'] == key), axis=0)
    TBIPlockey = tf.reshape(TBIPlockey, tf.reduce_prod(TBIPlockey.shape))
    TBIPsclkey = tf.gather(TBIPscl, tf.where(author_info['party'] == key), axis=0)
    TBIPsclkey = tf.reshape(TBIPsclkey, tf.reduce_prod(TBIPlockey.shape))
    mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=tf.fill(TBIPlockey.shape[0], 1/TBIPlockey.shape[0])),
        components_distribution=tfp.distributions.Normal(loc=TBIPlockey, scale=TBIPsclkey)
        )
    ys = mixture.prob(xs)
    ys /= tf.reduce_max(ys)
    ys *= 0.9
    plt.plot(xs, ys + 1.0, color=darkcolorcat[key])

    ### STBSarchical TBIP over all topics
    STBSlockey = tf.gather(STBSloc, tf.where(author_info['party'] == key), axis=0)
    STBSlockey = tf.reshape(STBSlockey, tf.reduce_prod(STBSlockey.shape))
    STBSsclkey = tf.gather(STBSscl, tf.where(author_info['party'] == key), axis=0)
    STBSsclkey = tf.reshape(STBSsclkey, tf.reduce_prod(STBSsclkey.shape))
    mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=tf.fill(STBSlockey.shape[0], 1 / STBSlockey.shape[0])),
        components_distribution=tfp.distributions.Normal(loc=STBSlockey, scale=STBSsclkey)
    )
    ys = mixture.prob(xs)
    ys /= tf.reduce_max(ys)
    ys *= 0.9
    plt.plot(xs, ys + 0.0, color=darkcolorcat[key])

    ### topics individually
    kcount = 0
    for k in permutation:
        STBSlockey = tf.gather(STBSloc[:, k], tf.where(author_info['party'] == key), axis=0)
        STBSlockey = tf.reshape(STBSlockey, tf.reduce_prod(STBSlockey.shape))
        STBSsclkey = tf.gather(STBSscl[:, k], tf.where(author_info['party'] == key), axis=0)
        STBSsclkey = tf.reshape(STBSsclkey, tf.reduce_prod(STBSsclkey.shape))
        mixture = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=tf.fill(STBSlockey.shape[0], 1 / STBSlockey.shape[0])),
            components_distribution=tfp.distributions.Normal(loc=STBSlockey, scale=STBSsclkey)
        )
        ys = mixture.prob(xs)
        ys /= tf.reduce_max(ys)
        ys *= 0.9
        plt.plot(xs, ys - kcount - 1.0, color=darkcolorcat[key])
        # plt.scatter(mixture.mean(), - kcount - 1.0 + 0.3, color=darkcolorcat[key], marker='x')
        plt.scatter(currentx[kcount], - kcount - 1.0 + 0.3, color=darkcolorcat[key], marker='x')
        kcount += 1
plt.hlines(tf.cast(topics, tf.float32)+0.3, xD, xR, colors='black')
# plt.scatter([xD, xR], tf.cast([topics, topics], tf.float32)+0.3,
#             color=np.append(np.repeat('blue', num_topics), np.repeat('red', num_topics)),
#             marker='x')
plt.hlines(0.3, xD0, xR0, colors='black')
plt.scatter([xD0, xR0], [0.3, 0.3], color=['blue', 'red'], marker='x')
plt.hlines(1.3, xD1, xR1, colors='black')
plt.scatter([xD1, xR1], [1.3, 1.3], color=['blue', 'red'], marker='x')
topicslab = permutation.numpy().astype(str)
plt.yticks(np.append(topics.numpy(), [0, 1]), np.append(topicslab, ['Averaged', 'TBIP']))
plt.box(False)
plt.margins(x=0, y=0)
plt.xlabel('Ideological position')
plt.tight_layout()
plt.gcf().set_size_inches((7.5, 15))
#plt.show()
plt.savefig(os.path.join(fig_dir, 'ideal_points_as_distribution_'+STBS+'_vs_TBIP_'+how_party+'_K'+str(num_topics)+'.png'),
            bbox_inches='tight')
plt.close()






### Mixture densities instead of dots
party_by_mean = False
if party_by_mean:
    # Party means
    xD = tf.reduce_mean(tf.gather(STBSloc, tf.where(author_info['party'] == 'D'), axis=0), axis=0)[0, :]
    xR = tf.reduce_mean(tf.gather(STBSloc, tf.where(author_info['party'] == 'R'), axis=0), axis=0)[0, :]
    how_party = 'party_by_average'
else:
    xD = iotas[:, 0]
    xR = iotas[:, 0] + iotas[:, 1]
    how_party = 'party_by_iota'
permutation = tf.argsort(tf.math.abs(xR - xD), direction='DESCENDING')
xD = tf.gather(xD, permutation)
xR = tf.gather(xR, permutation)
darkcolorcat = {"R": "red", "D": "blue", "I": "black"}
colorcat = {"R": "orange", "D": "lightblue", "I": "grey"}
xmin = -2.5
xmax = 2.5
xdif = (xmax - xmin) / 500
xs = tf.range(xmin, xmax + xdif, delta=xdif)
plt.vlines(0.0, -11.0, 2, colors='grey', linestyles='--')
for key in ['R', 'D']: #colorcat.keys():
    if key == 'R':
        currentx = xR[0:5]
    else:
        currentx = xD[0:5]
    ### classical TBIP
    TBIPlockey = tf.gather(TBIPloc, tf.where(author_info['party'] == key), axis=0)
    TBIPlockey = tf.reshape(TBIPlockey, tf.reduce_prod(TBIPlockey.shape))
    TBIPsclkey = tf.gather(TBIPscl, tf.where(author_info['party'] == key), axis=0)
    TBIPsclkey = tf.reshape(TBIPsclkey, tf.reduce_prod(TBIPlockey.shape))
    mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=tf.fill(TBIPlockey.shape[0], 1/TBIPlockey.shape[0])),
        components_distribution=tfp.distributions.Normal(loc=TBIPlockey, scale=TBIPsclkey)
        )
    ys = mixture.prob(xs)
    ys /= tf.reduce_max(ys)
    ys *= 0.9
    plt.plot(xs, ys + 1.0, color=darkcolorcat[key])

    ### STBSarchical TBIP over all topics
    STBSlockey = tf.gather(STBSloc, tf.where(author_info['party'] == key), axis=0)
    STBSlockey = tf.reshape(STBSlockey, tf.reduce_prod(STBSlockey.shape))
    STBSsclkey = tf.gather(STBSscl, tf.where(author_info['party'] == key), axis=0)
    STBSsclkey = tf.reshape(STBSsclkey, tf.reduce_prod(STBSsclkey.shape))
    mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=tf.fill(STBSlockey.shape[0], 1 / STBSlockey.shape[0])),
        components_distribution=tfp.distributions.Normal(loc=STBSlockey, scale=STBSsclkey)
    )
    ys = mixture.prob(xs)
    ys /= tf.reduce_max(ys)
    ys *= 0.9
    plt.plot(xs, ys + 0.0, color=darkcolorcat[key])

    ### topics individually
    kcount = 0
    for k in permutation[0:5]:
        STBSlockey = tf.gather(STBSloc[:, k], tf.where(author_info['party'] == key), axis=0)
        STBSlockey = tf.reshape(STBSlockey, tf.reduce_prod(STBSlockey.shape))
        STBSsclkey = tf.gather(STBSscl[:, k], tf.where(author_info['party'] == key), axis=0)
        STBSsclkey = tf.reshape(STBSsclkey, tf.reduce_prod(STBSsclkey.shape))
        mixture = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=tf.fill(STBSlockey.shape[0], 1 / STBSlockey.shape[0])),
            components_distribution=tfp.distributions.Normal(loc=STBSlockey, scale=STBSsclkey)
        )
        ys = mixture.prob(xs)
        ys /= tf.reduce_max(ys)
        ys *= 0.9
        plt.plot(xs, ys - kcount - 1.0, color=darkcolorcat[key])
        # plt.scatter(mixture.mean(), - kcount - 1.0 + 0.3, color=darkcolorcat[key], marker='x')
        plt.scatter(currentx[kcount], - kcount - 1.0 + 0.3, color=darkcolorcat[key], marker='x')
        kcount += 1
plt.hlines(tf.cast(-tf.range(5)-1, tf.float32)+0.3, xD[0:5], xR[0:5], colors='black')
# DOTS

# Continue
for key in ['R', 'D']: #colorcat.keys():
    if key == 'R':
        currentx = xR[-5:]
    else:
        currentx = xD[-5:]
    ### classical TBIP
    TBIPlockey = tf.gather(TBIPloc, tf.where(author_info['party'] == key), axis=0)
    TBIPlockey = tf.reshape(TBIPlockey, tf.reduce_prod(TBIPlockey.shape))
    TBIPsclkey = tf.gather(TBIPscl, tf.where(author_info['party'] == key), axis=0)
    TBIPsclkey = tf.reshape(TBIPsclkey, tf.reduce_prod(TBIPlockey.shape))
    mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=tf.fill(TBIPlockey.shape[0], 1/TBIPlockey.shape[0])),
        components_distribution=tfp.distributions.Normal(loc=TBIPlockey, scale=TBIPsclkey)
        )
    ys = mixture.prob(xs)
    ys /= tf.reduce_max(ys)
    ys *= 0.9
    plt.plot(xs, ys + 1.0, color=darkcolorcat[key])

    ### STBSarchical TBIP over all topics
    STBSlockey = tf.gather(STBSloc, tf.where(author_info['party'] == key), axis=0)
    STBSlockey = tf.reshape(STBSlockey, tf.reduce_prod(STBSlockey.shape))
    STBSsclkey = tf.gather(STBSscl, tf.where(author_info['party'] == key), axis=0)
    STBSsclkey = tf.reshape(STBSsclkey, tf.reduce_prod(STBSsclkey.shape))
    mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=tf.fill(STBSlockey.shape[0], 1 / STBSlockey.shape[0])),
        components_distribution=tfp.distributions.Normal(loc=STBSlockey, scale=STBSsclkey)
    )
    ys = mixture.prob(xs)
    ys /= tf.reduce_max(ys)
    ys *= 0.9
    plt.plot(xs, ys + 0.0, color=darkcolorcat[key])

    ### topics individually
    kcount = 6
    for k in permutation[-5:]:
        STBSlockey = tf.gather(STBSloc[:, k], tf.where(author_info['party'] == key), axis=0)
        STBSlockey = tf.reshape(STBSlockey, tf.reduce_prod(STBSlockey.shape))
        STBSsclkey = tf.gather(STBSscl[:, k], tf.where(author_info['party'] == key), axis=0)
        STBSsclkey = tf.reshape(STBSsclkey, tf.reduce_prod(STBSsclkey.shape))
        mixture = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=tf.fill(STBSlockey.shape[0], 1 / STBSlockey.shape[0])),
            components_distribution=tfp.distributions.Normal(loc=STBSlockey, scale=STBSsclkey)
        )
        ys = mixture.prob(xs)
        ys /= tf.reduce_max(ys)
        ys *= 0.9
        plt.plot(xs, ys - kcount - 1.0, color=darkcolorcat[key])
        # plt.scatter(mixture.mean(), - kcount - 1.0 + 0.3, color=darkcolorcat[key], marker='x')
        plt.scatter(currentx[kcount-6], - kcount - 1.0 + 0.3, color=darkcolorcat[key], marker='x')
        kcount += 1
plt.hlines(tf.cast(-tf.range(6, 11)-1, tf.float32)+0.3, xD[-5:], xR[-5:], colors='black')
# plt.scatter([xD, xR], tf.cast([topics, topics], tf.float32)+0.3,
#             color=np.append(np.repeat('blue', num_topics), np.repeat('red', num_topics)),
#             marker='x')
plt.hlines(0.3, xD0, xR0, colors='black')
plt.scatter([xD0, xR0], [0.3, 0.3], color=['blue', 'red'], marker='x')
plt.hlines(1.3, xD1, xR1, colors='black')
plt.scatter([xD1, xR1], [1.3, 1.3], color=['blue', 'red'], marker='x')
topicslab = permutation.numpy().astype(str)
sometopics = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11]
topicslab5 = topicslab[0:5]
topicslab5 = np.append(topicslab5, ['...'])
topicslab5 = np.append(topicslab5, topicslab[-5:])
plt.yticks(np.append(sometopics, [0, 1]), np.append(topicslab5, ['Averaged', 'TBIP']))
plt.box(False)
plt.margins(x=0, y=0)
plt.xlabel('Ideological position')
plt.tight_layout()
plt.gcf().set_size_inches((7.5, 8))
#plt.show()
plt.savefig(os.path.join(fig_dir, 'ideal_points_as_distribution_best_worst_'+STBS+'_vs_TBIP_'+how_party+'_K'+str(num_topics)+'.png'),
            bbox_inches='tight')
plt.close()




