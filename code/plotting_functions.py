# Import global packages
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from datetime import datetime


def change_format(val):
    val = str(val)
    date = datetime.strptime(val, "%Y%M%d")
    new_date = date.strftime("%Y-%M-%d")
    return new_date

def nice_nrow_ncol(nfig, increasing=True, n=20, maxratio=3):
    """Proposes ideal number of columns and rows for the total number of figures.
    | 1 | 2 |...|ncol|
    | 2 |   |   |    |
    |...|   |   |    |
    |nrow|  |   |    |

    Total number of cells should be higher than the number of figures.

    Args:
        nfig: Total number of figures to be covered
        increasing: True --> return the smaller value first
                    False--> return the higher value first
        n: number of trials
    Returns:
        nrow: [int]
        ncol: [int]
    """
    sq = np.sqrt(nfig)
    sqlow, squpp = int(np.floor(sq)), int(np.ceil(sq))
    upper = np.empty([], dtype=int)
    lower = np.empty([], dtype=int)
    remainder = np.array(999999, dtype=int)
    x = squpp
    y = squpp
    i = 0
    while (i < n) and (y >= 1) and (x/y <= maxratio):
        while (y*x >= nfig):
            y -= 1
        lower = np.append(lower, y+1)
        upper = np.append(upper, x)
        remainder = np.append(remainder, (y+1)*x-nfig)
        x += 1
        i += 1

    # find the first pair [x, y] minimizing the remainder
    i = remainder.argmin()
    if increasing:
        nrow, ncol = lower[i], upper[i]
    else:
        nrow, ncol = upper[i], lower[i]

    return nrow, ncol

def plot_several_bar_plots(figures, file, nrows = 1, ncols=1, width=0.8, size=(15, 15)):
    """Plot a dictionary of barplots. Midpoints and heights are expected.

    Args:
        figures: <title, figure> dictionary containing midpoints and heights
        file: where to save the figure [str]
        ncols: number of columns of subplots wanted in the display [integer]
        nrows: number of rows of subplots wanted in the figure [integer]
        width: width of the plotted bars [float]
    """
    if (ncols == 1) and (nrows == 1):
        title = 'Topic 0'
        mids, heights = figures[title]
        plt.bar(mids, height=heights, width=width)
        plt.title(title)
    else:
        fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
        ylim = 0
        for ind,title in enumerate(figures):
            mids, heights = figures[title]
            ylim = np.maximum(ylim, np.max(heights))
        for ind,title in enumerate(figures):
            mids, heights = figures[title]
            axeslist.ravel()[ind].bar(mids, height=heights, width=width)
            axeslist.ravel()[ind].set_ylim(0, ylim)
            axeslist.ravel()[ind].set_title(title)
    plt.gcf().set_size_inches(size)
    #plt.figure(figsize=(15, 15))
    plt.tight_layout()
    plt.savefig(file)
    plt.close()

def hist_values(values, grid):
    """Compute the counts of values in bins given by the grid.

    Args:
        values: vector of values to create a histogram
        grid: grid of bounds for the bins

    Returns:
        mids: midpoints of the bins
        counts: number of topics-specific locations in the bins
    """
    lower = grid[:-1]
    upper = grid[1:]
    count = np.full(lower.shape, 0.0)
    for i, x in enumerate(lower):
        count[i] = tf.reduce_sum(
            tf.dtypes.cast(values > lower[i], tf.int32) * tf.dtypes.cast(values <= upper[i], tf.int32))
    return ((lower + upper) / 2, count)

def hist_author(model, author_map, path,
                what="loc", grid_from=-2.5, grid_to=2.5, nbins=20, size=(15, 15)):
    """ Histograms of ideological positions for each author separately.
    If positions are topic-specific we obtain histograms of values for different topics.
    If positions are fixed for all topics we obtain just bars of height one.

    Args:
        model: TBIP class object
        author_map: vector[num_authors] containing names of the authors (labels for histograms)
        path: directory + file_name.png
        what: either "loc" or "scl" to designate what values should be plotted
        grid_from: the lowest value of the grid
        grid_to: the highest value of the grid
        nbins: number of bins
    """
    if what == "loc":
        xx = model.ideal_varfam.location
    elif what == "scl":
        xx = model.ideal_varfam.scale
    else:
        raise ValueError("Unrecognized choice of 'what' in hist_author plotting function. Use either 'loc' or 'scl'.")
    grid = np.linspace(grid_from, grid_to, nbins+1)
    figures = {author_map[i]: hist_values(xx[i, :], grid) for i in range(model.num_authors)}
    nrow, ncol = nice_nrow_ncol(model.num_authors)
    plot_several_bar_plots(figures, path, nrow, ncol, grid[1] - grid[0], size)

def hist_log_theta_means(model, path, grid_from=-2.5, grid_to=2.5, nbins=20, size=(15,15)):
    """ Histograms of log theta means for each topic separately.

    Args:
        model: TBIP class object
        path: directory + file_name.png
        grid_from: the lowest value of the grid
        grid_to: the highest value of the grid
        nbins: number of bins

    """
    grid = np.linspace(grid_from, grid_to, nbins + 1)
    Eqmeanlog = model.get_Eqmean(model.theta_varfam, log=False)
    figures = {'Topic '+str(k): hist_values(Eqmeanlog[:, k], grid) for k in range(model.num_topics)}
    nrow, ncol = nice_nrow_ncol(model.num_topics)
    plot_several_bar_plots(figures, path, nrow, ncol, grid[1] - grid[0], size)

def hist_max_theta_over_k(model, path, nbins=20, size=(15,15)):
    """ Histograms of log theta means for each topic separately.

    Args:
        model: TBIP class object
        path: directory + file_name.png
        nbins: number of bins
    """
    Eqmean = model.get_Eqmean(model.theta_varfam, log=True)
    # Scale wrt the sum of all thetas for the document OR sum to scale it back
    Eqmeansum = tf.reduce_mean(Eqmean, axis=1)
    Eqmeanscaled = Eqmean / Eqmeansum[:, tf.newaxis]
    # the maximal value of theta
    Eqmean_max_over_k = tf.reduce_max(Eqmeanscaled, axis=1)
    # which topic was maximal
    Eqmean_which_k_max = tf.argmax(Eqmeanscaled, axis=1)
    fig, axs = plt.subplots(1, 2, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(Eqmean_max_over_k, bins=nbins)
    axs[0].set_title('Maximal scaled theta')
    axs[1].hist(Eqmean_which_k_max, bins=model.num_topics)
    axs[1].set_title('Topic of maximal theta')
    plt.gcf().set_size_inches(size)
    plt.savefig(path)
    plt.close()

def hist_log_beta_means(model, path, grid_from=-2.5, grid_to=2.5, nbins=20, size=(15,15)):
    """ Histograms of log beta means for each topic separately.

    Args:
        model: TBIP class object
        path: directory + file_name.png
        grid_from: the lowest value of the grid
        grid_to: the highest value of the grid
        nbins: number of bins

    """
    grid = np.linspace(grid_from, grid_to, nbins + 1)
    Eqmeanlog = model.get_Eqmean(model.beta_varfam, log=True)
    figures = {'Topic '+str(k): hist_values(Eqmeanlog[k, :], grid) for k in range(model.num_topics)}
    nrow, ncol = nice_nrow_ncol(model.num_topics)
    plot_several_bar_plots(figures, path, nrow, ncol, grid[1] - grid[0], size)


def hist_eta_locations(model, path, grid_from=-2.5, grid_to=2.5, nbins=20, size=(15,15)):
    """ Histograms of eta locations for each topic separately.

    Args:
        model: TBIP class object
        path: directory + file_name.png
        grid_from: the lowest value of the grid
        grid_to: the highest value of the grid
        nbins: number of bins

    """
    grid = np.linspace(grid_from, grid_to, nbins + 1)
    figures = {'Topic '+str(k): hist_values(model.eta_varfam.location[k, :], grid) for k in range(model.num_topics)}
    nrow, ncol = nice_nrow_ncol(model.num_topics)
    plot_several_bar_plots(figures, path, nrow, ncol, grid[1] - grid[0], size)

def hist_iota_locations(model, path, grid_from=-2.5, grid_to=2.5, nbins=20, size=(5,10)):
    """ Histograms of iota locations for each topic separately.
    If iotas are topic-specific, we obtain histogram over different topics.
    If iotas are fixed for all topics, we obtain histograms with single bar of height 1.

    Args:
        model: TBIP class object
        path: directory + file_name.png
        grid_from: the lowest value of the grid
        grid_to: the highest value of the grid
        nbins: number of bins

    """
    if model.iota_varfam.family != 'deterministic':
        grid = np.linspace(grid_from, grid_to, nbins + 1)
        figures = {'Coefficient '+str(l): hist_values(model.iota_varfam.location[:, l], grid) for l in range(model.iota_varfam.location.shape[1])}
        nrow, ncol = nice_nrow_ncol(model.iota_varfam.location.shape[1])
        plot_several_bar_plots(figures, path, nrow, ncol, grid[1] - grid[0], size)

def plot_labels(x, y, labels, path, title, colors, size = (15,15), xlab = "X", ylab = "Y"):
    """ Plots the labels on [x,y] positions.

    Args:
        x: float[n] positions on x-axis
        y: float[n] positions on y-axis
        labels: str[n] text labels to be written on [x,y] positions
        path: where to save the plot (including the name and .png)
        title: title of the plot
        colors: str[n] colors for labels
        size: figure sizes in inches
        xlab: label for the x-axis
        ylab: lable for the y-axis

    """
    xlim = (min(x), max(x))
    xrange = xlim[1] - xlim[0]
    xlim += 0.05 * np.array([-1, 1]) * xrange
    ylim = (min(y), max(y))
    yrange = ylim[1] - ylim[0]
    ylim += 0.05 * np.array([-1, 1]) * yrange

    plt.figure(figsize=size)
    plt.xlabel(xlab)
    plt.xlim(xlim)
    plt.ylabel(ylab)
    plt.ylim(ylim)
    plt.title(title)
    plt.axhline(0, linestyle='--', color='grey')
    plt.axvline(0, linestyle='--', color='grey')
    for i in range(x.shape[0]):
        plt.text(x[i], y[i], labels[i], color=colors[i], horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    #plt.show()
    plt.savefig(path)
    plt.close()

def plot_words_beta_vs_eta(model, fig_dir, k, vocabulary, size = (15,15)):
    """ Plot vocabulary onto 2D plot based on log betas and etas.

    Args:
        model: TBIP class object
        fig_dir: directory where to save the figure
        k: topic number
        vocabulary: array of words
        size: figure sizes in inches
    """
    x = model.eta_varfam.location[k, :].numpy()
    y = model.get_gamma_distribution_Eqmean_subset(model.beta_varfam, k, log=True).numpy()

    plot_labels(x, y, vocabulary,
                path=os.path.join(fig_dir, 'words_beta_vs_eta_topic_'+str(k)+'.png'),
                title='Topic '+str(k)+': log beta means vs eta',
                colors=np.full(x.shape, 'black'),
                size=size,
                xlab='Ideological correction eta',
                ylab='Neutral topics log beta')

def plot_location_jittered(model, x, fig_dir, k, labels, author_category, size = (15,15)):
    if model.prior_choice['ideal_dim'] == "a" and k!=0:
        raise ValueError("When plotting verbosity vs ideal location only topic k=0 is allowed "
                         "if positions are not topic-specific.")
    if model.prior_choice['ideal_dim'] == "ak":
        path = os.path.join(fig_dir, 'ideal_' + str(k) + '_' + author_category.name + '.png')
        title = 'Topic ' + str(k) + ': ideological position'
    else:
        path = os.path.join(fig_dir, 'ideal_' + author_category.name + '.png')
        title = 'Ideological position'

    # Location
    y = tfp.distributions.Uniform().sample(len(x))

    colordict = {}
    for level in author_category.drop_duplicates():
        # random color
        colordict[level] = mcolors.CSS4_COLORS[np.random.choice(list(mcolors.CSS4_COLORS.keys()), 1)[0]]
    colors = [colordict[author_category[i]] for i in range(model.num_authors)]
    plot_labels(x, y, labels, path=path, title=title, colors=colors, size=size,
                xlab='Ideological position', ylab='Jitter')

def plot_coef_jittered(model, x, fig_dir, k, labels, author_category, size = (15,15)):
    if model.prior_choice['iota_dim'] == "kl":
        path = os.path.join(fig_dir, 'iota_' + str(k) + '_' + author_category + '.png')
        title = 'Topic ' + str(k) + ': iota coefficient'
    else:
        path = os.path.join(fig_dir, 'iota_' + author_category + '.png')
        title = 'iota coefficient'

    # Location
    y = tfp.distributions.Uniform().sample(len(x))

    colors = []
    for level in labels:
        # random color
        colors.append(mcolors.CSS4_COLORS[np.random.choice(list(mcolors.CSS4_COLORS.keys()), 1)[0]])
    plot_labels(x, y, labels, path=path, title=title, colors=colors, size=size,
                xlab='Ideological position', ylab='Jitter')

def plot_verbosity_vs_location(model, fig_dir, k, all_author_indices, author_map, author_category, size = (15,15)):
    """ Plot author-specific verbosity against his topic-specific location.
    The term verbosity refers to different quantities depending on the model choice:
        a) theta == Gfix --> model contains exp_verbosity --> use Eqmeanlog of exp_verbosity
        b) theta == Gdrte --> document-specific theta rates --> use averages of theta_rate Eqmeanlog across all documents written by the same author
        c) theta == Garte --> author-specific theta rates --> use Eqmeanlog of theta_rate

    Args:
        model: TBIP class object
        fig_dir: directory where to save the figure
        k: topic number (only 0 is allowed when locations are not topic-specific)
        all_author_indices: int[num_documents] indices of authors of all the documents
        author_map: str[num_authors] containing names of the authors
        author_category: str[num_authors] containing label of category of each author for colouring purposes
        size: figure sizes in inches

    """
    if model.prior_choice['ideal_dim'] == "a" and k!=0:
        raise ValueError("When plotting verbosity vs ideal location only topic k=0 is allowed "
                         "if positions are not topic-specific.")
    # Location
    x = model.ideal_varfam.location[:, k].numpy()
    # Verbosity - depends on the model choice

    if model.prior_choice['theta'] == "Gfix":
        # ... then exp_verbosity is in the model --> take Eqmeanlog
        y = model.get_Eqmean(model.exp_verbosity_varfam, log=True)
    elif model.prior_choice['theta'] == "Gdrte":
        # ... document-specific rates adjust for different lengths of documents
        # We have to summarize over documents written by the same author
        # Go through each document and add quantity to its author
        author_doc_count = tf.math.unsorted_segment_sum(tf.fill(len(all_author_indices), 1), all_author_indices,
                                                        num_segments=model.num_authors)
        author_doc_count = tf.cast(author_doc_count, dtype=tf.float32)
        Eqmeanlog = model.get_Eqmean(model.theta_rate_varfam, log=True)
        author_rate = tf.math.unsorted_segment_sum(Eqmeanlog, all_author_indices, num_segments=model.num_authors)

        # y-axis will be average Eqmeanlog of theta rates over documents written by that author
        # Since higher theta rate results in lower intensities theta, we take a negative value.
        y = - author_rate / author_doc_count
    elif model.prior_choice['theta'] == "Garte":
        # ... author-specific rates --> take Eqmeanlog of theta rates
        # Since higher theta rate results in lower intensities theta, we take a negative value.
        y = - model.get_Eqmean(model.theta_rate_varfam, log=True)
    else:
        raise ValueError("The prior choice for theta: '"+model.prior_choice['theta']+"', is not recognized.")

    if model.prior_choice['ideal_dim'] == "ak":
        path = os.path.join(fig_dir, 'authors_verbosity_vs_ideal_' + str(k) + '_' + author_category.name + '.png')
        title = 'Topic ' + str(k) + ': verbosity vs ideological position'
    else:
        path = os.path.join(fig_dir, 'authors_verbosity_vs_ideal_' + author_category.name + '.png')
        title = 'Verbosity vs ideological position'

    if max(np.vectorize(len)(author_category)) > 2:
        labels = [author_map[i] for i in range(model.num_authors)]
    else:
        labels = [author_map[i] + ' ' + author_category[i] for i in range(model.num_authors)]
    colorcat = {"R": "red", "D": "blue", "I": "grey",
                "M": "blue", "F": "red",
                "Midwest": "green", "Northeast": "blue", "Southeast": "red", "South": "teal", "West": "orange",
                "Silent": "red", "Boomers": "orange", "Gen X": "green",
                "(0, 1]": "green", "(1, 5]": "orange", "(1, 10]": "orange", "(5, 100]": "red", "(10, 100]": "red",
                "Catholic": "red", "Presbyterian": "orange", "Baptist": "yellow", "Jewish": "cyan",
                "Unspecified/Other (Protestant)": "black", "Methodist": "purple", "Lutheran": "green", "Mormon": "blue",
                "Anglican / Episcopal": "grey", "Don’t Know/Refused": "grey", "Congregationalist": "grey",
                "Nondenominational Christian": "grey", "Buddhist": "grey",
                "2005": "red", "2006": "orange", "2007": "green",
                "chairman": "blue", "vice chairman": "purple", "governor": "black", "ms_or_mr": "green",
                "short": "green", "medium": "orange", "long": "red",
                "relaxed": "green", "serious": "red",
    }
    colors = [colorcat[author_category[i]] for i in range(model.num_authors)]
    plot_labels(x, y, labels, path=path, title=title, colors=colors, size=size,
                xlab='Ideological position', ylab='Verbosity')

def plot_beta_eta_vs_ideal_top_N_words(model, fig_dir, N, k, vocabulary, size = (15,15)):
    """ Plot the objective topics (beta) corrected by the ideological term (eta * ideal),
    while changing the value of ideological positions.
    Choose N words achieving the highest values on the interval bounded by min and max ideal locations.

    Args:
        model: TBIP class object
        fig_dir: directory where to save the figure
        N: number of top words
        k: topic number
        vocabulary: str[num_words] all used words
        size: figure sizes in inches

    """
    betas = model.get_gamma_distribution_Eqmean_subset(model.beta_varfam, k, log=True)
    etas = model.eta_varfam.location[k, :]

    #xlim = tf.Variable([tf.reduce_min(model.ideal_varfam.location), tf.reduce_max(model.ideal_varfam.location)])
    # or rather fixed boundaries
    xlim = tf.Variable([-1.2, 1.2])
    xlimin = tf.Variable([-1.0, 1.0])
    y = betas[:, tf.newaxis] + etas[:, tf.newaxis] * xlim[tf.newaxis, :]
    yin = betas[:, tf.newaxis] + etas[:, tf.newaxis] * xlimin[tf.newaxis, :]
    # Now choose those words which appear highest
    ymax = tf.maximum(y[:, 0], y[:, 1])
    values, indices = tf.nn.top_k(ymax, N)
    word_indices = indices.numpy()

    ymean = 0.5*(y + yin)
    xmean = 0.5*(xlim+xlimin)

    # random positions for text labels
    #unifdist = tfp.distributions.Uniform(xlim[0], xlim[1])
    unifdist = tfp.distributions.Uniform(-0.2, 0.2)
    xpos = unifdist.sample(N)
    ypos = tf.gather(betas, word_indices) + tf.gather(etas, word_indices)*xpos

    ylim = (tf.reduce_min(tf.gather(y, word_indices, axis=0)),
            tf.reduce_max(tf.gather(y, word_indices, axis=0)))
    yrange = ylim[1] - ylim[0]
    ylim += 0.05 * np.array([-1, 1]) * yrange

    plt.figure(figsize=size)
    plt.xlabel('Ideological position')
    plt.xlim(xlim)
    plt.ylabel('Corrected objective topic (log scale)')
    plt.ylim(ylim)
    plt.title('Top '+str(N)+' words: topic '+str(k))
    #plt.axhline(0, linestyle='--', color='grey')
    plt.axvline(-1, linestyle='--', color='grey')
    plt.axvline(0, linestyle='--', color='grey')
    plt.axvline(1, linestyle='--', color='grey')
    for i, wi in enumerate(word_indices):
        # random color
        color = mcolors.CSS4_COLORS[np.random.choice(list(mcolors.CSS4_COLORS.keys()), 1)[0]]
        plt.plot(xlim.numpy(), y[wi, :].numpy(), linestyle='-', color=color)
        plt.text(xpos[i].numpy(), ypos[i].numpy(), vocabulary[wi], color=color)
        plt.text(xmean[0].numpy(), ymean[wi, 0].numpy(), vocabulary[wi], color=color)
        plt.text(xmean[1].numpy(), ymean[wi, 1].numpy(), vocabulary[wi], color=color)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(fig_dir, 'top'+str(N)+'words_vs_ideal_topic_'+str(k)+'.png'))
    plt.close()

def barplot_ordered_labels_top(x, label, path, size=(15,15)):
    xsort = -np.sort(-x)
    xord = np.argsort(-x)
    labup = 0.02*np.max(x)
    plt.figure(figsize=size)
    #plt.axis('off')
    plt.xticks([])
    plt.bar(range(len(x)), height=xsort, align='edge')
    for i in range(len(x)):
        plt.text(i, xsort[i]+labup, label+str(xord[i]))
    plt.tight_layout()
    # plt.show()
    plt.savefig(path)
    plt.close()

def plot_wordclouds(model, path, nwords, topics, vocabulary, logscale = True, justeta = False, size=(10,2.5)):
    title = {"neg": "Negative", "neu": "Neutral", "pos": "Positive"}
    plt.figure()
    x = tf.Variable([-1.0, 0.0, 1.0])

    for ik, k in enumerate(topics):
        if justeta:
            if logscale:
                betas = tf.zeros(model.num_words)
            else:
                betas = tf.ones(model.num_words)
        else:
            betas = model.get_gamma_distribution_Eqmean_subset(model.beta_varfam, k, log=logscale)
        etas = tf.gather(model.eta_varfam.location, k, axis=0)
        if logscale:
            y = betas[:, tf.newaxis] + etas[:, tf.newaxis] * x[tf.newaxis, :]
            # Problem is that on log scale y can have negative values.
            # And negative values are not suitable for 'frequency' parameter of WordClouds.
            # Therefore, subtract the minimum and add something positive (5 % of the range).
            ymin = tf.reduce_min(y)
            ymax = tf.reduce_max(y)
            y = y - 1.05*ymin + 0.05*ymax
        else:
            etascl = tf.gather(model.eta_varfam.scale, k, axis=0)
            y = betas[:, tf.newaxis] * tf.math.exp(
                etas[:, tf.newaxis] * x[tf.newaxis, :] + 0.5 * (x[tf.newaxis, :] ** 2) * (etascl[:, tf.newaxis] ** 2)
            )
        dictionary = {}
        dictionary["neg"] = {vocabulary[v]: y.numpy()[v, 0] for v in range(vocabulary.shape[0])}
        if not justeta:
            dictionary["neu"] = {vocabulary[v]: y.numpy()[v, 1] for v in range(vocabulary.shape[0])}
        dictionary["pos"] = {vocabulary[v]: y.numpy()[v, 2] for v in range(vocabulary.shape[0])}
        ncolumn = len(dictionary.keys())

        for ind, name in enumerate(dictionary.keys()):
            if ik == 0:
                plt.subplot(len(topics), ncolumn, ik * ncolumn + ind + 1).set_title(title[name])
            else:
                plt.subplot(len(topics), ncolumn, ik * ncolumn + ind + 1).set_title('')
            plt.plot()
            word_cloud = WordCloud(collocations=False, background_color='white', max_words=nwords)
            word_cloud.generate_from_frequencies(frequencies=dictionary[name])
            plt.imshow(word_cloud, interpolation='bilinear')
            plt.xticks([])
            plt.yticks([])
            if ind == 0:
                plt.ylabel('Topic '+str(k))
    plt.gcf().set_size_inches(size)
    plt.tight_layout()
    plt.show()
    plt.savefig(path)
    plt.close()

def plot_wordclouds_slides(model, path, nwords, k, vocabulary, logscale = True, size=(10,2.5)):
    plt.figure()
    x = tf.Variable([-1.0, 0.0, 1.0])
    betathreshold = 0.01

    betas = model.get_gamma_distribution_Eqmean_subset(model.beta_varfam, k, log=logscale)
    etas = model.eta_varfam.location[k, :]
    yeta = etas[:, tf.newaxis] * x[tf.newaxis, :]
    if logscale:
        y = betas[:, tf.newaxis] + yeta
        # Problem is that on log scale y can have negative values.
        # And negative values are not suitable for 'frequency' parameter of WordClouds.
        # Therefore, subtract the minimum and add something positive (5 % of the range).
        ymin = tf.reduce_min(y)
        ymax = tf.reduce_max(y)
        y = y - 1.05*ymin + 0.05*ymax
        title = {"neg": "E log(beta) - eta", "neu": "E log(beta)", "pos": "E log(beta) + eta",
                 "negeta": "E -eta under log(beta) > "+str(betathreshold), "poseta": "E +eta under log(beta) > "+str(betathreshold)
                 }
    else:
        etascl = model.eta_varfam.scale[k, :]
        yeta = tf.math.exp(yeta + 0.5 * (x[tf.newaxis, :] ** 2) * (etascl[:, tf.newaxis] ** 2))
        y = betas[:, tf.newaxis] * yeta
        title = {"neg": "E beta * exp(- eta)", "neu": "E beta", "pos": "E beta * exp(+ eta)",
                 "negeta": "E exp(-eta) under log(beta) > "+str(betathreshold), "poseta": "E exp(+eta) under log(beta) > "+str(betathreshold)
                 }
    dicty = {}
    dicty["neg"] = {vocabulary[v]: y.numpy()[v, 0] for v in range(vocabulary.shape[0])}
    dicty["neu"] = {vocabulary[v]: y.numpy()[v, 1] for v in range(vocabulary.shape[0])}
    dicty["pos"] = {vocabulary[v]: y.numpy()[v, 2] for v in range(vocabulary.shape[0])}
    dictyeta = {}
    dictyeta["negeta"] = {vocabulary[v]: yeta.numpy()[v, 0]*(betas.numpy()[v] > betathreshold) for v in range(vocabulary.shape[0])}
    dictyeta["poseta"] = {vocabulary[v]: yeta.numpy()[v, 2]*(betas.numpy()[v] > betathreshold) for v in range(vocabulary.shape[0])}
    ncolumn = 3

    for ind, name in enumerate(dicty.keys()):
        plt.subplot(2, ncolumn, ind + 1).set_title(title[name])
        plt.plot()
        word_cloud = WordCloud(collocations=False, background_color='white', max_words=nwords)
        word_cloud.generate_from_frequencies(frequencies=dicty[name])
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.xticks([])
        plt.yticks([])

    plt.subplot(2, ncolumn, ncolumn + 1).set_title(title['negeta'])
    plt.plot()
    word_cloud = WordCloud(collocations=False, background_color='white', max_words=nwords)
    word_cloud.generate_from_frequencies(frequencies=dictyeta['negeta'])
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, ncolumn, ncolumn + 2).set_title('')
    plt.plot()
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.text(0, 0, 'Topic '+str(k), fontsize=24, horizontalalignment='center', verticalalignment='center')

    plt.subplot(2, ncolumn, ncolumn + 3).set_title(title['poseta'])
    plt.plot()
    word_cloud = WordCloud(collocations=False, background_color='white', max_words=nwords)
    word_cloud.generate_from_frequencies(frequencies=dictyeta['poseta'])
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.xticks([])
    plt.yticks([])

    plt.gcf().set_size_inches(size)
    plt.tight_layout()
    plt.show()
    plt.savefig(path)
    plt.close()

def plot_wordclouds_parties(model, path, nwords, topics, vocabulary, iota_based = True, logscale = True, size=(7,2.5)):
    title = {"D": "Democrats", "R": "Republicans"}
    plt.figure()
    for ik, k in enumerate(topics):
        # Prediction of ideological position for each political party from iotas.
        # Used to be:
        # iota[0] - intercept - independent (I) senators
        # iota[1] - difference between independent (I) and democratic (D) senators
        # iota[2] - difference between independent (I) and republican (R) senators
        # But now:
        # iota[0] - intercept - Democratic (D) senators
        # iota[1] - difference between Democratic (D) and Republican (R) senators
        # iota[2] - difference between Democratic (D) and Independent (I) senators
        if iota_based:
            if model.prior_choice['iota_dim'] == "kl":
                x = model.iota_varfam.location[k, 0:2]
            else:
                x = model.iota_varfam.location[0, 0:2]
            update = tf.reduce_sum(x)
            x = tf.tensor_scatter_nd_update(x[:, tf.newaxis], [[1]], update[tf.newaxis, tf.newaxis])
            x = tf.reshape(x, 2)
        else:
            # use party averages of ideal point locations
            if model.prior_choice['ideal_dim'] == "ak":
                x = model.ideal_varfam.location[:, k]
            else:
                x = model.ideal_varfam.location[:, 0]
                # 0=D, 1=R, -1=I
            party_index = 1*model.X[:, 0] -1*model.X[:, 1]
            x = tf.math.unsorted_segment_mean(x, party_index, num_segments=2)


        if logscale:
            betas = model.get_gamma_distribution_Eqmean_subset(model.beta_varfam, k, log=True)
            etas = model.eta_varfam.location[k, :]
            y = betas[:, tf.newaxis] + etas[:, tf.newaxis] * x[tf.newaxis, :]
            # Problem is that on log scale y can have negative values.
            # And negative values are not suitable for 'frequency' parameter of WordClouds.
            # Therefore, subtract the minimum and add something positive (5 % of the range).
            ymin = tf.reduce_min(y)
            ymax = tf.reduce_max(y)
            y = y - 1.05*ymin + 0.05*ymax
        else:
            betas = model.get_gamma_distribution_Eqmean_subset(model.beta_varfam, k, log=False)
            etas = model.eta_varfam.location[k, :]
            etascl = model.eta_varfam.scale[k, :]
            y = betas[:, tf.newaxis] * tf.math.exp(
                etas[:, tf.newaxis] * x[tf.newaxis, :] + 0.5 * (x[tf.newaxis, :] ** 2) * (etascl[:, tf.newaxis] ** 2)
            )
        dictionary = {
            "D": {vocabulary[v]: y.numpy()[v, 0] for v in range(vocabulary.shape[0])},
            "R": {vocabulary[v]: y.numpy()[v, 1] for v in range(vocabulary.shape[0])},
        }
        for ind, name in enumerate(dictionary.keys()):
            if ik == 0:
                plt.subplot(len(topics), 2, ik * 2 + ind + 1).set_title(title[name])
            else:
                plt.subplot(len(topics), 2, ik * 2 + ind + 1).set_title('')
            plt.plot()
            word_cloud = WordCloud(collocations=False, background_color='white', max_words=nwords)
            word_cloud.generate_from_frequencies(frequencies=dictionary[name])
            plt.imshow(word_cloud, interpolation='bilinear')
            plt.xticks([])
            plt.yticks([])
            if ind == 0:
                plt.ylabel('Topic '+str(k))
    plt.gcf().set_size_inches(size)
    plt.tight_layout()
    # plt.show()
    plt.savefig(path)
    plt.close()

def plot_ideal_points_thin(model, fig_dir, k, author_category, xlim=(-1.2, 1.2), size=(2.5, 0.25)):
    locs = model.ideal_varfam.location[:, k]
    xmin = tf.reduce_mean(tf.gather(locs, tf.where(author_category == 'D')))
    xmax = tf.reduce_mean(tf.gather(locs, tf.where(author_category == 'R')))
    markercat = {"R": "x", "D": "o", "I": "+",
                 "M": "x", "F": "o",
                 "Midwest": "o", "Northeast": "o", "Southeast": "o", "South": "o", "West": "o",
                 "Silent": "o", "Boomers": "o", "Gen X": "o",
                 "(0, 1]": "o", "(1, 5]": "o", "(1, 10]": "o", "(5, 100]": "o", "(10, 100]": "o",
                 "Catholic": "o", "Presbyterian": "o", "Baptist": "o", "Jewish": "o",
                 "Unspecified/Other (Protestant)": "o", "Methodist": "o", "Lutheran": "o", "Mormon": "o",
                 "Anglican / Episcopal": "o", "Don’t Know/Refused": "o", "Congregationalist": "o",
                 "Nondenominational Christian": "o", "Buddhist": "o",
                 "2005": "x", "2006": "x", "2007": "x",
                 "chairman": "x", "vice chairman": "x", "governor": "x", "ms_or_mr": "x",
                 "short": "x", "medium": "x", "long": "x",
                "relaxed": "x", "serious": "x",
                 }
    colorcat = {"R": "red", "D": "blue", "I": "grey",
                "M": "blue", "F": "red",
                "Midwest": "green", "Northeast": "blue", "Southeast": "red", "South": "teal", "West": "orange",
                "Silent": "red", "Boomers": "orange", "Gen X": "green",
                "(0, 1]": "green", "(1, 5]": "orange", "(1, 10]": "orange", "(5, 100]": "red", "(10, 100]": "red",
                "Catholic": "red", "Presbyterian": "orange", "Baptist": "yellow", "Jewish": "cyan",
                "Unspecified/Other (Protestant)": "black", "Methodist": "purple", "Lutheran": "green", "Mormon": "blue",
                "Anglican / Episcopal": "grey", "Don’t Know/Refused": "grey", "Congregationalist": "grey",
                "Nondenominational Christian": "grey", "Buddhist": "grey",
                "2005": "red", "2006": "orange", "2007": "green",
                "chairman": "blue", "vice chairman": "purple", "governor": "black", "ms_or_mr": "green",
                "short": "green", "medium": "orange", "long": "red",
                "relaxed": "green", "serious": "red",
                }
    # colors = [colorcat[author_category[i]] for i in range(model.num_authors)]
    # markers = [markercat[author_category[i]] for i in range(model.num_authors)]
    # plt.scatter(x=locs, y=tf.zeros(locs.shape), color=colors, marker=markers)
    for i in range(model.num_authors):
        plt.scatter(x=locs[i], y=0.0, color=colorcat[author_category[i]], marker=markercat[author_category[i]])
    plt.axis(False)
    plt.xlim(xlim)
    plt.box(False)
    plt.margins(x=0, y=0)
    plt.vlines(0.0, -1, 1, colors='grey', linestyles='--')
    plt.hlines(0.0, xmin, xmax, colors='black')
    plt.scatter([xmin, xmax], [0.0, 0.0], color='black', marker='x')
    plt.tight_layout()
    #plt.subplots_adjust(left=0.1, right=0.1, top=0.1, bottom=0.1)
    plt.gcf().set_size_inches(size)
    #plt.show()
    plt.savefig(os.path.join(fig_dir, 'ideal_points_' + str(k) + '.png'))
    plt.close()

def plot_ideal_points_as_distribution(model, fig_dir, k, author_category, xlim=(-2.5, 2.5), size=(10, 5)):
    locs = model.ideal_varfam.location[:, k]
    scls = model.ideal_varfam.scale[:, k]
    markercat = {"R": "x", "D": "x", "I": "+",
                 "M": "x", "F": "x",
                 "Midwest": "o", "Northeast": "o", "Southeast": "o", "South": "o", "West": "o",
                 "Silent": "o", "Boomers": "o", "Gen X": "o",
                 "(0, 1]": "o", "(1, 5]": "o", "(1, 10]": "o", "(5, 100]": "o", "(10, 100]": "o",
                 "Catholic": "o", "Presbyterian": "o", "Baptist": "o", "Jewish": "o",
                 "Unspecified/Other (Protestant)": "o", "Methodist": "o", "Lutheran": "o", "Mormon": "o",
                 "Anglican / Episcopal": "o", "Don’t Know/Refused": "o", "Congregationalist": "o",
                 "Nondenominational Christian": "o", "Buddhist": "o",
                 "2005": "x", "2006": "x", "2007": "x",
                 "chairman": "x", "vice chairman": "x", "governor": "x", "ms_or_mr": "x",
                 "short": "x", "medium": "x", "long": "x",
                 "relaxed": "x", "serious": "x",
                 }
    darkcolorcat = {"R": "red", "D": "blue", "I": "black",
                    "M": "blue", "F": "red",
                    "Midwest": "green", "Northeast": "blue", "Southeast": "red", "South": "teal", "West": "orange",
                    "Silent": "red", "Boomers": "orange", "Gen X": "green",
                    "(0, 1]": "green", "(1, 5]": "orange", "(1, 10]": "orange", "(5, 100]": "red", "(10, 100]": "red",
                    "Catholic": "red", "Presbyterian": "orange", "Baptist": "yellow", "Jewish": "cyan",
                    "Unspecified/Other (Protestant)": "black", "Methodist": "purple", "Lutheran": "green",
                    "Mormon": "blue",
                    "Anglican / Episcopal": "grey", "Don’t Know/Refused": "grey", "Congregationalist": "grey",
                    "Nondenominational Christian": "grey", "Buddhist": "grey",
                    "2005": "red", "2006": "orange", "2007": "green",
                    "chairman": "blue", "vice chairman": "purple", "governor": "black", "ms_or_mr": "green",
                    "short": "green", "medium": "orange", "long": "red",
                    "relaxed": "green", "serious": "red",
                    }
    colorcat = {"R": "orange", "D": "lightblue", "I": "grey",
                "M": "lightblue", "F": "orange",
                "Midwest": "green", "Northeast": "blue", "Southeast": "red", "South": "teal", "West": "orange",
                "Silent": "red", "Boomers": "orange", "Gen X": "green",
                "(0, 1]": "green", "(1, 5]": "orange", "(1, 10]": "orange", "(5, 100]": "red", "(10, 100]": "red",
                "Catholic": "red", "Presbyterian": "orange", "Baptist": "yellow", "Jewish": "cyan",
                "Unspecified/Other (Protestant)": "black", "Methodist": "purple", "Lutheran": "green", "Mormon": "blue",
                "Anglican / Episcopal": "grey", "Don’t Know/Refused": "grey", "Congregationalist": "grey",
                "Nondenominational Christian": "grey", "Buddhist": "grey",
                "2005": "red", "2006": "orange", "2007": "green",
                "chairman": "blue", "vice chairman": "purple", "governor": "black", "ms_or_mr": "green",
                "short": "green", "medium": "orange", "long": "red",
                "relaxed": "green", "serious": "red",
                }
    for i in range(model.num_authors):
        plt.scatter(x=locs[i], y=0.0, color=colorcat[author_category[i]], marker="x")
        idealdist = tfp.distributions.Normal(loc=locs[i], scale=scls[i])
        xmin = idealdist.quantile(0.02)
        xmax = idealdist.quantile(0.98)
        xdif = (xmax - xmin) / 20
        xs = tf.range(xmin, xmax+xdif, delta=xdif)
        ys = idealdist.prob(xs)
        plt.plot(xs, ys, color=colorcat[author_category[i]], linewidth=0.5, linestyle='--')

    xmin = tf.reduce_min(tfp.distributions.Normal(loc=locs, scale=scls).quantile(0.02))
    xmax = tf.reduce_max(tfp.distributions.Normal(loc=locs, scale=scls).quantile(0.98))
    xdif = (xmax - xmin) / 200
    xs = tf.range(xmin, xmax + xdif, delta=xdif)
    for key in colorcat.keys():
        if tf.reduce_sum(tf.cast(author_category == key, tf.int32)) > 0:
            lockey = tf.gather(locs, tf.where(author_category == key))[:, 0]
            sclkey = tf.gather(scls, tf.where(author_category == key))[:, 0]
            mixture = tfp.distributions.MixtureSameFamily(
                mixture_distribution=tfp.distributions.Categorical(probs=tf.fill(lockey.shape[0], 1/lockey.shape[0])),
                components_distribution=tfp.distributions.Normal(loc=lockey, scale=sclkey)
                )
            ys = mixture.prob(xs)
            plt.plot(xs, 5*ys, color=darkcolorcat[key], linewidth=1.5, linestyle='-')
    plt.xlim(xlim)
    plt.tight_layout()
    plt.gcf().set_size_inches(size)
    # plt.show()
    plt.savefig(os.path.join(fig_dir, 'ideal_points_as_distribution_vs_' + author_category.name + '_' + str(k) + '.png'))
    plt.close()

def plot_heatmap_coefficients(table, title, fig_path, xtick=[], ytick=[]):
    # Define the plot
    fig, ax = plt.subplots(figsize=(13, 7))
    # Set the font size and the distance of the title from the plot
    plt.title(title, fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5, 1.05])
    # new pandas dataframe to name the rows and columns
    named_table = pd.DataFrame(table, index=ytick, columns=xtick)
    # Use the heatmap function from the seaborn package
    sns.heatmap(named_table, fmt="", cmap='bwr', linewidths=0.30, ax=ax)
    # Display the eatmap
    # plt.show()
    # Save the heatmap
    plt.gcf().set_size_inches((10, 13))
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def plot_ccps(table, title, fig_path, size = (7, 13)):
    xlabels = ["Topic " + str(i) for i in range(0, table.shape[0]-1)] + ["Averaged"]
    ylabels = ["Gender", "Region", "Generation", "Experience", "Religion"]
    if table.shape[1] == 6:
        ylabels = ["Party"] + ylabels
    # Define the plot
    fig, ax = plt.subplots(figsize=(13, 7))
    # Set the font size and the distance of the title from the plot
    plt.title(title, fontsize=18)
    ax.title.set_position([0.5, 1.05])
    # new pandas dataframe to name the rows and columns
    named_table = pd.DataFrame(table.to_numpy(), index=xlabels, columns=ylabels)
    # Use the heatmap function from the seaborn package
    sns.heatmap(named_table, vmin=0, vmax=1.0, fmt="", cmap='magma_r', linewidths=0.30, ax=ax)
    # Display the heatmap
    plt.show()
    # Save the heatmap
    plt.gcf().set_size_inches(size)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def linear_combination_CCP(C, mu, Sigma, mu0 = 0.0, separately = True):
    # C[L,Z] - matrix declaring linear combinations
    # mu[K,L] - vector of estimated means
    # Sigma[K,L,L] - variance matrix
    # mu0[K,L] - vector for testing hypothesis
    CSigmaC = tf.transpose(C)[tf.newaxis, :, :] @ Sigma @ C[tf.newaxis, :, :] # [K, Z, Z]
    Cdif = tf.transpose(C)[tf.newaxis, :, :] @ mu[:, :, tf.newaxis] - mu0 # [K, Z, 1]
    if separately:
        invdiag = tf.math.reciprocal(tf.linalg.diag_part(CSigmaC)) # [K, Z]
        chi = invdiag * tf.square(Cdif[:, :, 0])
        dist = tfp.distributions.Chi2(df=1.0)
        ccp = 1.0 - dist.cdf(chi)
    else:
        x = tf.linalg.solve(CSigmaC, Cdif)
        chi = tf.transpose(Cdif, perm=[0, 2, 1]) @ x
        dist = tfp.distributions.Chi2(df=tf.cast(tf.shape(C)[1], tf.float64))
        ccp = 1.0 - dist.cdf(tf.cast(chi[:, 0, 0], tf.double))
    return ccp

def covariates_hein_daily():
    covariates = {
        'party': {'dem_ind': [1, 2],
                   'names': ['Democrats', 'Republicans', "Independent"],
                   'labels': ['D', 'R', 'I']},
        'gender': {'dem_ind': [3],
                   'names': ['Male', 'Female'],
                   'labels': ['M', 'F']},
        'region': {'dem_ind': [4, 5, 6, 7],
                   'names': ["Northeast", "Midwest", "Southeast", "South", "West"],
                   'labels': ["Northeast", "Midwest", "Southeast", "South", "West"]},
        'generation': {'dem_ind': [8, 9],
                       'names': ["Silent", "Boomer", "Gen X"],
                       'labels': ["Silent", "Boomers", "Gen X"]},
        'exper_cong': {'dem_ind': [10, 11],
                       'names': ["(10,100]", "(1,10]", "(0,1]"],
                       'labels': ["(10, 100]", "(1, 10]", "(0, 1]"]},
        'religion': {'dem_ind': [12, 13, 14, 15, 16, 17, 18],
                     'names': ["Other", "Catholic", "Presbyterian", "Baptist",
                     "Jewish", "Methodist", "Lutheran", "Mormon"],
                     'labels': [["Congregationalist", "Anglican/Episcopal", "Unspecified/Other (Protestant)",
                       "Nondenominational Christian", "Don’t Know/Refused", "Buddhist"],
                      "Catholic", "Presbyterian", "Baptist",
                      "Jewish", "Methodist", "Lutheran", "Mormon"]},
    }
    return covariates

def covariates_fomc():
    covariates = {
        'gender': {'ind': [1],
                   'names': ['Male', 'Female'],
                   'labels': ['M', 'F']},
        'title': {'ind': [2, 3],
                  'names': ["Ms. or Mr.", "Chairman", "Vice Chairman"],
                  'labels': ["ms_or_mr", "chairman", "vice chairman"]},
        'year': {'ind': [4, 5],
                 'names': ["2005", "2006", "2007"],
                 'labels': ["2005", "2006", "2007"]},
        'flength': {'ind': [6, 7],
                    'names': ["Short", "Medium", "Long"],
                    'labels': ["short", "medium", "long"]},
        'flaughter': {'ind': [8, 9],
                      'names': ["Serious", "Medium", "Relaxed"],
                      'labels': ["serious", "medium", "relaxed"]},
    }
    return covariates

def create_lin_komb_interactions(party="D", category="gender", include_baseline=False, include_party=False, L=51):
    dem_ind = covariates_hein_daily()[category]['dem_ind']
    c0 = tf.zeros(L)
    if include_party:
        if party == 'R':
            c1 = tf.tensor_scatter_nd_update(c0, [[0], [1]], [1.0, 1.0])
        elif party == 'I':
            c1 = tf.tensor_scatter_nd_update(c0, [[0], [2]], [1.0, 1.0])
        else:
            c1 = tf.tensor_scatter_nd_update(c0, [[0]], [1.0])
    else:
        c1 = c0
    if include_baseline:
        c2 = tf.repeat(c1[:, tf.newaxis], len(dem_ind)+1, axis=1)
    else:
        c2 = tf.repeat(c1[:, tf.newaxis], len(dem_ind), axis=1)
    indices = []
    for i in range(len(dem_ind)):
        if include_baseline:
            j = i+1
        else:
            j = i

        if party == 'R':
            indices += [[dem_ind[i], j], [dem_ind[i] + 16, j]]
        elif party == 'I':
            indices += [[dem_ind[i], j], [dem_ind[i] + 32, j]]
        else:
            indices += [[dem_ind[i], j]]

    c = tf.tensor_scatter_nd_update(c2, indices, tf.ones(len(indices)))
    return c

def var_matrix_of_iota(model):
    if model.iota_varfam.family == 'normal':
        # spread all variances on the diagonals
        # scale of [K,L] to variance matrix of [K, L, L]
        Sigma = tf.linalg.diag(tf.math.square(model.iota_varfam.scale))
    elif model.iota_varfam.family == 'MVnormal':
        # duplicate Sigmas over all topics --> done automatically by .covariance()
        # (distribution is already topic-specific)
        # covariance of [K, L, L]
        # Sigma = tf.repeat(model.iota_varfam.distribution.covariance()[tf.newaxis, :, :], model.num_topics, axis=0)
        Sigma = model.iota_varfam.distribution.covariance()
    else:
        Sigma = tf.linalg.diag(tf.ones(model.iota_varfam.location.shape))
    return Sigma



def create_all_general_descriptive_figures(model, fig_dir: str, author_map, vocabulary,
                                           nwords_eta_beta_vs_ideal: int = 20,
                                           nwords: int = 10,
                                           ntopics: int = 5):
    """ Create and save all the figures that describe the model output (histograms, barplots, wordclouds, ...).
    All the plots are completely general, no specific information or assumption about the dataset are required.
    Hence, these plots can be created for any dataset of speeches.

    Args:
        model: TBIP class object
        fig_dir: directory where to save the figure
        author_map: str[num_authors] containing names of the authors
        vocabulary: str[num_words] all used words
        nwords_eta_beta_vs_ideal: int[1] number of words to be plotted in beta_eta_vs_ideal
        nwords: int[1] number of words to be included within a wordcloud
        ntopics: int[1] number of topics to be plotted together (wordclouds)

    """
    ### Histograms:
    print("Min author location: " + str(np.min(model.ideal_varfam.location)))
    print("Max author location: " + str(np.max(model.ideal_varfam.location)))
    hist_author(model, author_map, os.path.join(fig_dir, 'ideal_loc_per_author.png'),
                what="loc", grid_from=-2.5, grid_to=2.5, size=(15, 15))

    print("Min author scale: " + str(np.min(model.ideal_varfam.scale)))
    print("Max author scale: " + str(np.max(model.ideal_varfam.scale)))
    hist_author(model, author_map, os.path.join(fig_dir, 'ideal_scl_per_author.png'),
                what="scl", grid_from=1e-3, grid_to=0.2, size=(15, 15))

    print("Min log theta mean: " + str(np.min(model.get_Eqmean(model.theta_varfam, log=True))))
    print("Max log theta mean: " + str(np.max(model.get_Eqmean(model.theta_varfam, log=True))))
    hist_log_theta_means(model, os.path.join(fig_dir, 'hist_log_theta_means.png'),
                         grid_from=-9, grid_to=2.5, size=(15, 15))

    print("Min log beta mean: " + str(np.min(model.get_Eqmean(model.beta_varfam, log=True))))
    print("Max log beta mean: " + str(np.max(model.get_Eqmean(model.beta_varfam, log=True))))
    hist_log_beta_means(model, os.path.join(fig_dir, 'hist_log_beta_means.png'),
                        grid_from=-12, grid_to=2.5, size=(15, 15))

    # print("Min beta mean: " + str(np.min(model.get_Eqmean(model.beta_varfam, log=False))))
    # print("Max beta mean: " + str(np.max(model.get_Eqmean(model.beta_varfam, log=False))))
    # hist_log_beta_means(model, os.path.join(fig_dir, 'hist_beta_means.png'),
    #                     grid_from=0, grid_to=100, size=(15, 15))

    # Histogram of posterior means of beta rates + top5 & tail5 words
    beta_rates = model.get_Eqmean(model.beta_rate_varfam)
    beta_rate_head_val, beta_rate_head_ind = tf.math.top_k(beta_rates, k=5)
    beta_rate_tail_val, beta_rate_tail_ind = tf.math.top_k(-beta_rates, k=5)
    print("Top 5 words with highest beta rate:")
    for k in range(5):
        print(vocabulary[beta_rate_head_ind[k]])
    print("Top 5 words with lowest beta rate:")
    for k in range(5):
        print(vocabulary[beta_rate_tail_ind[k]])

    plt.hist(beta_rates)  # summed over documents
    plt.title("Posterior mean of beta rates")
    plt.xlabel("Beta rate")
    plt.savefig(os.path.join(fig_dir, "hist_beta_rate.png"))
    plt.close()

    print("Min eta location: " + str(np.min(model.eta_varfam.location)))
    print("Max eta location: " + str(np.max(model.eta_varfam.location)))
    hist_eta_locations(model, os.path.join(fig_dir, 'hist_eta_locations.png'),
                       grid_from=-2.5, grid_to=2.5, size=(15, 15))

    print("Min iota location: " + str(np.min(model.iota_varfam.location)))
    print("Max iota location: " + str(np.max(model.iota_varfam.location)))
    hist_iota_locations(model, os.path.join(fig_dir, 'hist_iota_locations.png'),
                        grid_from=-1.5, grid_to=1.5, size=(10, 5))

    hist_max_theta_over_k(model, os.path.join(fig_dir, 'hist_max_theta_over_k.png'), size=(12, 5))

    ### Words beta vs eta
    ### The following takes long time to render:
    # for k in range(model.num_topics):
    #     plot_words_beta_vs_eta(model, fig_dir, k, vocabulary, size=(15, 15))

    ### Beta combined with eta depending on location x: log_beta[k,v] + x*eta[k,v]
    ### Top N words
    for k in range(model.num_topics):
        plot_beta_eta_vs_ideal_top_N_words(model, fig_dir, nwords_eta_beta_vs_ideal, k, vocabulary, size=(15, 15))

    ### Wordclouds - for ideological positions -1, 0, 1
    for k in range(tf.math.maximum(model.num_topics // ntopics, 1)):
        topics = k * ntopics + np.array([i for i in range(ntopics)])
        topics = topics[topics < model.num_topics]
        plot_wordclouds(model, os.path.join(fig_dir, 'wordclouds_topics_'+str(topics[0])+"_"+str(topics[-1])+'.png'),
                        nwords, topics, vocabulary, logscale=False, justeta=False, size=(10, 10))
        plot_wordclouds(model, os.path.join(fig_dir, 'wordclouds_logscale_topics_' + str(topics[0]) + "_" + str(topics[-1]) + '.png'),
                        nwords, topics, vocabulary, logscale=True, justeta=False, size=(10, 10))
        # just etas
        plot_wordclouds(model, os.path.join(fig_dir, 'wordclouds_justeta_topics_' + str(topics[0]) + "_" + str(topics[-1]) + '.png'),
                        nwords, topics, vocabulary, logscale=False, justeta=True, size=(7, 10))
        plot_wordclouds(model,
                        os.path.join(fig_dir, 'wordclouds_justeta_logscale_topics_' + str(topics[0]) + "_" + str(topics[-1]) + '.png'),
                        nwords, topics, vocabulary, logscale=True, justeta=True, size=(7, 10))

    ### Wordclouds separately for each topic - for slides
    for k in range(model.num_topics):
        plot_wordclouds_slides(model, os.path.join(fig_dir, 'wordclouds_logscale_topic_'+str(k)+'.png'),
                               nwords, k, vocabulary, logscale=True, size=(11, 5))

    ### Author-specific theta rates
    if model.prior_choice['theta'] in ["Garte"]:
        x = model.get_Eqmean(model.theta_rate_varfam, log=True)
        barplot_ordered_labels_top(x, '', os.path.join(fig_dir, 'barplot_theta_rate_Eqlogmean.png'), size=(15, 5))

    ### Author-specific regression precisions
    if model.prior_choice['ideal_prec'] in ["Naprec"]:
        x = 1/model.get_Eqmean(model.ideal_prec_varfam, log=False)
        barplot_ordered_labels_top(x, '', os.path.join(fig_dir, 'barplot_ideal_prec_to_var_Eqmean.png'), size=(15, 5))

    ### Topic-specific eta variances on log-scale
    if model.prior_choice['eta'] in ["NkprecG", "NkprecF"]:
        x = 1.0 / model.get_Eqmean(model.eta_prec_varfam, log=False)
        barplot_ordered_labels_top(x, 'T', os.path.join(fig_dir, 'barplot_eta_prec_to_var.png'), size=(15, 5))

    ### Topic-specific kappa_rho
    if model.prior_choice['eta'] == "NkprecF":
        barplot_ordered_labels_top(model.get_Eqmean(model.eta_prec_rate_varfam, log=False),
                                   'T', os.path.join(fig_dir, 'barplot_eta_prec_rate.png'), size=(15, 5))

    ### Topic-specific omega variances
    if model.prior_choice['iota_prec'] in ["NlprecG", "NlprecF"]:
        x = 1.0 / model.get_Eqmean(model.iota_prec_varfam, log=False)
        barplot_ordered_labels_top(x, 'C', os.path.join(fig_dir, 'barplot_iota_prec_to_var.png'), size=(10, 5))

    ### Coefficient-specific kappa_omega
    if model.prior_choice['iota_prec'] == "NlprecF":
        barplot_ordered_labels_top(model.get_Eqmean(model.iota_prec_rate_varfam, log=False),
                                   'C', os.path.join(fig_dir, 'barplot_iota_prec_rate.png'), size=(10, 5))


def create_all_figures_specific_to_data(model, data_name: str, covariates: str, fig_dir: str, all_author_indices,
                                        author_map, author_info, vocabulary,
                                        nwords: int = 10,
                                        ntopics: int = 5,
                                        selected_topics=[5, 9, 11, 13, 15],
                                        ):
    """ Create and save all the figures that describe the model output which is specific to the analysed data.
    So far, only datasets 'hein-daily', 'cze_senate' and 'pharma' are recognized.
    In case a new dataset is created, add additional elif for data_name and create your own plots specific to your data.

    Args:
        model: TBIP class object
        data_name: str[1] name of the dataset which selects what plots are created
        fig_dir: directory where to save the figure
        all_author_indices: int[num_documents] indices of authors of all the documents
        author_map: str[num_authors] containing names of the authors
        author_info: str[num_authors, columns] dataframe containing strings about each author
        vocabulary: str[num_words] all used words
        nwords: int[1] number of words to be included within a wordcloud
        ntopics: int[1] number of topics to be plotted together (wordclouds)
        selected_topics: int[ntopics] topics for which word clouds should be made together

    """
    if data_name == 'hein-daily':
        ###------------------------------------###
        ###  PLOTS FOR THE hein-daily DATASET  ###
        ###------------------------------------###
        ### Wordclouds - ideological positions depending on political party
        for k in range(model.num_topics // ntopics):
            topics = k * ntopics + np.array([i for i in range(ntopics)])
            plot_wordclouds_parties(model, os.path.join(fig_dir, 'wordclouds_party_topics_' + str(topics[0]) + "_" + str(
                                        topics[-1]) + '.png'),
                                    nwords, topics, vocabulary,
                                    iota_based=model.iota_varfam.family != 'deterministic',
                                    logscale=False, size=(7, 10))
            plot_wordclouds_parties(model, os.path.join(fig_dir, 'wordclouds_party_logscale_topics_' + str(topics[0]) + "_" + str(
                                        topics[-1]) + '.png'),
                                    nwords, topics, vocabulary,
                                    iota_based=model.iota_varfam.family != 'deterministic',
                                    logscale=True, size=(7, 10))

        ### Chosen wordclouds
        plot_wordclouds(model, os.path.join(fig_dir, 'wordclouds_selected_topics.png'),
                        nwords, selected_topics, vocabulary, logscale=False, size=(10, 10))
        plot_wordclouds(model, os.path.join(fig_dir, 'wordclouds_logscale_selected_topics.png'),
                        nwords, selected_topics, vocabulary, logscale=True, size=(10, 10))

        # intercept = model.iota_varfam.location[:, 0]
        intercept = tf.zeros(model.iota_varfam.location[:, 0].shape)

        if model.iota_varfam.family != "deterministic" and model.ideal_dim[1] > 1:
            if covariates in ['party', 'all_no_int', 'all']:
                ### Best discriminating topics wrt party (Republican vs Democrats)
                topics = tf.math.top_k(tf.math.abs(model.iota_varfam.location[:, 1]), ntopics).indices
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_party_wordclouds.png'),
                                nwords, topics.numpy(), vocabulary, logscale=False, size=(10, 10))
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_party_wordclouds_logscale.png'),
                                nwords, topics.numpy(), vocabulary, logscale=True, size=(10, 10))

                ### Worst discriminating topics wrt party (Republican vs Democrats)
                topics = tf.math.top_k(-tf.math.abs(model.iota_varfam.location[:, 1]), ntopics).indices
                plot_wordclouds(model, os.path.join(fig_dir, 'tail'+str(ntopics)+'_party_wordclouds.png'),
                                nwords, topics.numpy(), vocabulary, logscale=False, size=(10, 10))
                plot_wordclouds(model, os.path.join(fig_dir, 'tail'+str(ntopics)+'_party_wordclouds_logscale.png'),
                                nwords, topics.numpy(), vocabulary, logscale=True, size=(10, 10))

            if covariates in ['all_no_int', 'all']:
                ### Best discriminating topics wrt gender (Male vs Female)
                discrepancyD = tf.math.abs(model.iota_varfam.location[:, 3])
                if covariates == 'all_no_int':
                    discrepancy = discrepancyD
                else:
                    discrepancyR = tf.math.abs(model.iota_varfam.location[:, 3]+model.iota_varfam.location[:, 3+16])
                    discrepancy = tf.math.maximum(discrepancyD, discrepancyR)
                topics = tf.math.top_k(discrepancy, ntopics).indices
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_gender_wordclouds.png'),
                                nwords, topics.numpy(), vocabulary, logscale=False, size=(10, 10))
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_gender_wordclouds_logscale.png'),
                                nwords, topics.numpy(), vocabulary, logscale=True, size=(10, 10))
                barplot_ordered_labels_top(discrepancy, 'T', os.path.join(fig_dir, 'female_vs_male_by_topic.png'), size=(15, 5))

                ### Best discriminating topics wrt region (std between regions)
                xD = tf.stack((intercept,
                              intercept + model.iota_varfam.location[:, 4],
                              intercept + model.iota_varfam.location[:, 5],
                              intercept + model.iota_varfam.location[:, 6],
                              intercept + model.iota_varfam.location[:, 7]
                              ))
                discrepancyD = tf.math.reduce_std(xD, axis=0)
                if covariates == 'all_no_int':
                    discrepancy = discrepancyD
                    plot_heatmap_coefficients(tf.transpose(xD), "Region coefficients",
                                              os.path.join(fig_dir, 'coefficients_region.png'),
                                              xtick=["Northeast", "Midwest", "Southeast", "South", "West"],
                                              ytick=range(model.num_topics))
                else:
                    xR = tf.stack((intercept,
                                   intercept + model.iota_varfam.location[:, 4] + model.iota_varfam.location[:, 4+16],
                                   intercept + model.iota_varfam.location[:, 5] + model.iota_varfam.location[:, 5+16],
                                   intercept + model.iota_varfam.location[:, 6] + model.iota_varfam.location[:, 6+16],
                                   intercept + model.iota_varfam.location[:, 7] + model.iota_varfam.location[:, 7+16]
                                   ))
                    discrepancyR = tf.math.reduce_std(xR, axis=0)
                    discrepancy = tf.math.maximum(discrepancyD, discrepancyR)
                    plot_heatmap_coefficients(tf.transpose(xD), "Region coefficients Democrats",
                                              os.path.join(fig_dir, 'coefficients_region_D.png'),
                                              xtick=["Northeast", "Midwest", "Southeast", "South", "West"],
                                              ytick=range(model.num_topics))
                    plot_heatmap_coefficients(tf.transpose(xR), "Region coefficients Republicans",
                                              os.path.join(fig_dir, 'coefficients_region_R.png'),
                                              xtick=["Northeast", "Midwest", "Southeast", "South", "West"],
                                              ytick=range(model.num_topics))

                topics = tf.math.top_k(discrepancy, ntopics).indices
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_region_wordclouds.png'),
                                nwords, topics.numpy(), vocabulary, logscale=False, size=(10, 10))
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_region_wordclouds_logscale.png'),
                                nwords, topics.numpy(), vocabulary, logscale=True, size=(10, 10))
                barplot_ordered_labels_top(discrepancy, 'T', os.path.join(fig_dir, 'region_std_by_topic.png'),
                                           size=(15, 5))


                ### Best discriminating topics wrt age (Silent vs Gen X generation)
                discrepancyD = tf.math.abs(model.iota_varfam.location[:, 9])
                if covariates == 'all_no_int':
                    discrepancy = discrepancyD
                else:
                    discrepancyR = tf.math.abs(model.iota_varfam.location[:, 9] + model.iota_varfam.location[:, 9 + 16])
                    discrepancy = tf.math.maximum(discrepancyD, discrepancyR)
                topics = tf.math.top_k(discrepancy, ntopics).indices
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_generation_wordclouds.png'),
                                nwords, topics.numpy(), vocabulary, logscale=False, size=(10, 10))
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_generation_wordclouds_logscale.png'),
                                nwords, topics.numpy(), vocabulary, logscale=True, size=(10, 10))
                x = tf.stack((intercept,
                              intercept + model.iota_varfam.location[:, 8],
                              intercept + model.iota_varfam.location[:, 9]
                              ))
                plot_heatmap_coefficients(tf.transpose(x), "Generation coefficients",
                                          os.path.join(fig_dir, 'coefficients_generation.png'),
                                          xtick=["Silent", "Boomer", "Gen X"],
                                          ytick=range(model.num_topics))
                barplot_ordered_labels_top(tf.math.reduce_std(x, axis=0), 'T',
                                           os.path.join(fig_dir, 'generation_std_by_topic.png'), size=(15, 5))

                ### Best discriminating topics wrt experience  (experience in congress)
                xD = tf.stack((intercept + model.iota_varfam.location[:, 11],
                              intercept + model.iota_varfam.location[:, 10],
                              intercept
                              ))
                discrepancyD = tf.math.reduce_std(xD, axis=0)
                if covariates == 'all_no_int':
                    discrepancy = discrepancyD
                    plot_heatmap_coefficients(tf.transpose(xD), "Experience coefficients",
                                              os.path.join(fig_dir, 'coefficients_exper.png'),
                                              xtick=["(0,1]", "(1,10]", "(10,100]"],
                                              ytick=range(model.num_topics))
                else:
                    xR = tf.stack((intercept + model.iota_varfam.location[:, 11] + model.iota_varfam.location[:, 11+16],
                                   intercept + model.iota_varfam.location[:, 10] + model.iota_varfam.location[:, 10+16],
                                   intercept
                                   ))
                    discrepancyR = tf.math.reduce_std(xR, axis=0)
                    discrepancy = tf.math.maximum(discrepancyD, discrepancyR)
                    plot_heatmap_coefficients(tf.transpose(xD), "Experience coefficients Democrats",
                                              os.path.join(fig_dir, 'coefficients_exper_D.png'),
                                              xtick=["(0,1]", "(1,10]", "(10,100]"],
                                              ytick=range(model.num_topics))
                    plot_heatmap_coefficients(tf.transpose(xR), "Experience coefficients Republicans",
                                              os.path.join(fig_dir, 'coefficients_exper_R.png'),
                                              xtick=["(0,1]", "(1,10]", "(10,100]"],
                                              ytick=range(model.num_topics))
                topics = tf.math.top_k(discrepancy, ntopics).indices
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_experience_wordclouds.png'),
                                nwords, topics.numpy(), vocabulary, logscale=False, size=(10, 10))
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_experience_wordclouds_logscale.png'),
                                nwords, topics.numpy(), vocabulary, logscale=True, size=(10, 10))
                barplot_ordered_labels_top(discrepancy, 'T', os.path.join(fig_dir, 'exper_std_by_topic.png'),
                                           size=(15, 5))

                ### Best discriminating topics wrt religion (experience in congress)
                xD = tf.stack((intercept,
                              intercept + model.iota_varfam.location[:, 12],
                              intercept + model.iota_varfam.location[:, 13],
                              intercept + model.iota_varfam.location[:, 14],
                              intercept + model.iota_varfam.location[:, 15],
                              intercept + model.iota_varfam.location[:, 16],
                              intercept + model.iota_varfam.location[:, 17],
                              intercept + model.iota_varfam.location[:, 18]
                              ))
                discrepancyD = tf.math.reduce_std(xD, axis=0)
                if covariates == 'all_no_int':
                    discrepancy = discrepancyD
                    plot_heatmap_coefficients(tf.transpose(xD), "Religion coefficients",
                                              os.path.join(fig_dir, 'coefficients_religion.png'),
                                              xtick=["Other", "Catholic", "Presbyterian", "Baptist",
                                                     "Jewish", "Methodist", "Lutheran", "Mormon"],
                                              ytick=range(model.num_topics))
                else:
                    xR = tf.stack((intercept,
                              intercept + model.iota_varfam.location[:, 12] + model.iota_varfam.location[:, 12+16],
                              intercept + model.iota_varfam.location[:, 13] + model.iota_varfam.location[:, 13+16],
                              intercept + model.iota_varfam.location[:, 14] + model.iota_varfam.location[:, 14+16],
                              intercept + model.iota_varfam.location[:, 15] + model.iota_varfam.location[:, 15+16],
                              intercept + model.iota_varfam.location[:, 16] + model.iota_varfam.location[:, 16+16],
                              intercept + model.iota_varfam.location[:, 17] + model.iota_varfam.location[:, 17+16],
                              intercept + model.iota_varfam.location[:, 18] + model.iota_varfam.location[:, 18+16]
                              ))
                    discrepancyR = tf.math.reduce_std(xR, axis=0)
                    discrepancy = tf.math.maximum(discrepancyD, discrepancyR)
                    plot_heatmap_coefficients(tf.transpose(xD), "Religion coefficients Democrats",
                                              os.path.join(fig_dir, 'coefficients_religion_D.png'),
                                              xtick=["Other", "Catholic", "Presbyterian", "Baptist",
                                                     "Jewish", "Methodist", "Lutheran", "Mormon"],
                                              ytick=range(model.num_topics))
                    plot_heatmap_coefficients(tf.transpose(xR), "Religion coefficients Republicans",
                                              os.path.join(fig_dir, 'coefficients_religion_R.png'),
                                              xtick=["Other", "Catholic", "Presbyterian", "Baptist",
                                                     "Jewish", "Methodist", "Lutheran", "Mormon"],
                                              ytick=range(model.num_topics))
                topics = tf.math.top_k(discrepancy, ntopics).indices
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_religion_wordclouds.png'),
                                nwords, topics.numpy(), vocabulary, logscale=False, size=(10, 10))
                plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_religion_wordclouds_logscale.png'),
                                nwords, topics.numpy(), vocabulary, logscale=True, size=(10, 10))
                barplot_ordered_labels_top(discrepancy, 'T', os.path.join(fig_dir, 'religion_std_by_topic.png'),
                                           size=(15, 5))

            if covariates == 'all':
                for category in ['gender', 'region', 'generation', 'exper_cong', 'religion']:
                    cat_names = covariates_hein_daily()[category]['names']
                    cat_labels = covariates_hein_daily()[category]['labels']
                    cD = create_lin_komb_interactions(party='D', category=category,
                                                      include_party=False, L=model.iota_varfam.location.shape[1])
                    cR = create_lin_komb_interactions(party='R', category=category,
                                                      include_party=False, L=model.iota_varfam.location.shape[1])
                    Sigma = var_matrix_of_iota(model)
                    ccpD = linear_combination_CCP(cD, model.iota_varfam.location, Sigma, mu0=0.0, separately=False)
                    ccpR = linear_combination_CCP(cR, model.iota_varfam.location, Sigma, mu0=0.0, separately=False)
                    min_p_value = tf.math.minimum(ccpD, ccpR)
                    topics_minccp = tf.math.top_k(-min_p_value, ntopics).indices
                    plot_wordclouds(model, os.path.join(fig_dir, 'top'+str(ntopics)+'_' + category + '_wordclouds_minccp_D_R.png'),
                                    nwords, topics_minccp.numpy(), vocabulary, logscale=True, size=(10, 10))

            if covariates == 'all_no_int':
                # No interaction model --> plot significance of each covariate
                ccp = {}
                Sigma = var_matrix_of_iota(model)
                Sigmam = tf.linalg.diag(tf.math.square(model.iota_mean_varfam.scale))
                for category in ['party', 'gender', 'region', 'generation', 'exper_cong', 'religion']:
                    C = create_lin_komb_interactions(party='D', category=category,
                                                     include_baseline=False, include_party=False,
                                                     L=model.iota_varfam.location.shape[1])
                    ccptopics = linear_combination_CCP(C, model.iota_varfam.location,
                                                           Sigma, mu0=0.0, separately=False)
                    ccpaveraged = linear_combination_CCP(C, model.iota_mean_varfam.location[tf.newaxis, :],
                                                             Sigmam, mu0=0.0, separately=False)
                    ccp[category] = np.append(ccptopics.numpy(), ccpaveraged.numpy())
                table = pd.DataFrame(ccp)
                plot_ccps(table, 'Significance in no-interaction model',
                          os.path.join(fig_dir, 'ccps_table.png'), size=(8, 13))

            if covariates == 'all':
                # Model with interactions
                ccpD = {}
                ccpR = {}
                ccpInt = {}
                Sigma = var_matrix_of_iota(model)
                Sigmam = tf.linalg.diag(tf.math.square(model.iota_mean_varfam.scale))
                for category in ['gender', 'region', 'generation', 'exper_cong', 'religion']:
                    CD = create_lin_komb_interactions(party='D', category=category,
                                                      include_baseline=False, include_party=False,
                                                      L=model.iota_varfam.location.shape[1])
                    CR = create_lin_komb_interactions(party='R', category=category,
                                                      include_baseline=False, include_party=False,
                                                      L=model.iota_varfam.location.shape[1])
                    CInt = CR - CD
                    ccptopicsD = linear_combination_CCP(CD, model.iota_varfam.location,
                                                            Sigma, mu0=0.0, separately=False)
                    ccpaveragedD = linear_combination_CCP(CD, model.iota_mean_varfam.location[tf.newaxis, :],
                                                              Sigmam, mu0=0.0, separately=False)
                    ccptopicsR = linear_combination_CCP(CR, model.iota_varfam.location,
                                                            Sigma, mu0=0.0, separately=False)
                    ccpaveragedR = linear_combination_CCP(CR, model.iota_mean_varfam.location[tf.newaxis, :],
                                                              Sigmam, mu0=0.0, separately=False)
                    ccptopicsInt = linear_combination_CCP(CInt, model.iota_varfam.location,
                                                              Sigma, mu0=0.0, separately=False)
                    ccpaveragedInt = linear_combination_CCP(CInt, model.iota_mean_varfam.location[tf.newaxis, :],
                                                                Sigmam, mu0=0.0, separately=False)
                    ccpD[category] = np.append(ccptopicsD.numpy(), ccpaveragedD.numpy())
                    ccpR[category] = np.append(ccptopicsR.numpy(), ccpaveragedR.numpy())
                    ccpInt[category] = np.append(ccptopicsInt.numpy(), ccpaveragedInt.numpy())
                plot_ccps(pd.DataFrame(ccpD), 'Significance for Democrats',
                             os.path.join(fig_dir, 'ccps_table_Democrats.png'), size=(7, 13))
                plot_ccps(pd.DataFrame(ccpR), 'Significance for Republicans',
                             os.path.join(fig_dir, 'ccps_table_Republicans.png'), size=(7, 13))
                plot_ccps(pd.DataFrame(ccpInt), 'Significance of interaction terms',
                             os.path.join(fig_dir, 'ccps_table_interactions.png'), size=(7, 13))

            if covariates == 'all':
                # we have model with interactions
                for category in ['gender', 'region', 'generation', 'exper_cong', 'religion']:
                    if category == 'gender':
                        dem_ind = [3] # tf.range(3, 4)
                        cat_names = ['Male', 'Female']
                        cat_labels = ['M', 'F']
                    elif category == 'region':
                        dem_ind = [4, 5, 6, 7] # tf.range(4, 8)
                        cat_names = ["Northeast", "Midwest", "Southeast", "South", "West"]
                        cat_labels = ["Northeast", "Midwest", "Southeast", "South", "West"]
                    elif category == 'generation':
                        dem_ind = [8, 9] #tf.range(8, 10)
                        cat_names = ["Silent", "Boomer", "Gen X"]
                        cat_labels = ["Silent", "Boomers", "Gen X"]
                    elif category == 'exper_cong':
                        dem_ind = [10, 11] # tf.range(10, 12)
                        cat_names = ["(10,100]", "(1,10]", "(0,1]"]
                        cat_labels = ["(10, 100]", "(1, 10]", "(0, 1]"]
                    elif category == 'religion':
                        dem_ind = [12, 13, 14, 15, 16, 17, 18] # tf.range(12, 19)
                        cat_names = ["Other", "Catholic", "Presbyterian", "Baptist",
                                     "Jewish", "Methodist", "Lutheran", "Mormon"]
                        cat_labels = [["Congregationalist", "Anglican/Episcopal", "Unspecified/Other (Protestant)",
                                       "Nondenominational Christian", "Don’t Know/Refused", "Buddhist"],
                                      "Catholic", "Presbyterian", "Baptist",
                                      "Jewish", "Methodist", "Lutheran", "Mormon"]
                    rep_ind = [dem_ind[i] + 16 for i in range(len(dem_ind))]
                    dem = tf.concat([tf.zeros(model.iota_varfam.location[:, 0].shape)[:, tf.newaxis],
                                     tf.gather(model.iota_varfam.location, dem_ind, axis=1)], axis=1)
                    rep = tf.concat([tf.zeros(model.iota_varfam.location[:, 0].shape)[:, tf.newaxis],
                                     tf.gather(model.iota_varfam.location, dem_ind, axis=1)+tf.gather(model.iota_varfam.location, rep_ind, axis=1)],
                                    axis=1)
                    mdem = np.concatenate(([0.0], tf.gather(model.iota_mean_varfam.location, dem_ind).numpy()))
                    mrep = np.concatenate(([0.0], tf.gather(model.iota_mean_varfam.location, dem_ind).numpy() + tf.gather(model.iota_mean_varfam.location, rep_ind).numpy()))
                    # mdem = tf.concat([0.0,  tf.gather(model.iota_mean_varfam.location, dem_ind)])
                    # mrep = tf.concat([0.0,  tf.gather(model.iota_mean_varfam.location, dem_ind)+tf.gather(model.iota_mean_varfam.location, rep_ind)], axis=0)
                    # wordclouds for best discrimination
                    std_dem = tf.math.reduce_std(dem, axis=1)
                    std_rep = tf.math.reduce_std(rep, axis=1)
                    std_max = tf.math.maximum(std_dem, std_rep)
                    topics_max = tf.math.top_k(std_max, ntopics).indices
                    plot_wordclouds(model, os.path.join(fig_dir, 'top5_'+category+'_wordclouds_max_D_R.png'),
                                    nwords, topics_max.numpy(), vocabulary, logscale=True, size=(10, 10))
                    for include_party in [True, False]:
                        cD0 = tf.zeros(model.iota_varfam.location.shape[1])
                        cR0 = tf.zeros(model.iota_varfam.location.shape[1])
                        if include_party:
                            include_party_label = "with_party_effect"
                            dem_fin = dem + model.iota_varfam.location[:, 0][:, tf.newaxis]
                            rep_fin = rep + model.iota_varfam.location[:, 0][:, tf.newaxis] + model.iota_varfam.location[:, 1][:, tf.newaxis]
                            mdem_fin = mdem + model.iota_mean_varfam.location[0]
                            mrep_fin = mrep + model.iota_mean_varfam.location[0] + model.iota_mean_varfam.location[1]
                            cD1 = tf.tensor_scatter_nd_update(cD0, [[0]], [1.0])
                            cR1 = tf.tensor_scatter_nd_update(cR0, [[0], [1]], [1.0, 1.0])
                            indDm0 = [0]
                            indRm0 = [0, 1]
                        else:
                            dem_fin = dem
                            rep_fin = rep
                            mdem_fin = mdem
                            mrep_fin = mdem
                            include_party_label = "no_party_effect"
                            cD1 = cD0
                            cR1 = cR0
                            indDm0 = []
                            indRm0 = []
                        cD2 = tf.repeat(cD1[:, tf.newaxis], len(dem_ind)+1, axis=1)
                        indicesD = []
                        indDm = []
                        for i in range(len(dem_ind)):
                            indicesD += [[dem_ind[i], i+1]]
                            indDm += [indDm0 + [dem_ind[i]]]
                        cD = tf.tensor_scatter_nd_update(cD2, indicesD, tf.ones(len(dem_ind)))
                        cR2 = tf.repeat(cR1[:, tf.newaxis], len(rep_ind)+1, axis=1)
                        indicesR = []
                        indRm = []
                        for i in range(len(rep_ind)):
                            indicesR += [[dem_ind[i], i+1], [rep_ind[i], i+1]]
                            indRm += [indRm0 + [dem_ind[i], rep_ind[i]]]
                        # print(indDm)
                        # print(indRm)
                        cR = tf.tensor_scatter_nd_update(cR2, indicesR, tf.ones(2*len(dem_ind)))
                        Sigma = var_matrix_of_iota(model)
                        Sigmam = tf.linalg.diag(tf.math.square(model.iota_mean_varfam.scale))
                        ## Compute the variances: c^T Sigma c    of the linear combinations
                        # c:                [L, num_groups]
                        # Sigma:            [K, L, L]
                        # Sigma @ c:        [K, L, num_groups]
                        # c^T @ Sigma @ c:  [K, num_groups, num_groups] -> extract the diagonal
                        # variancesD = tf.linalg.diag_part(tf.matmul(tf.transpose(cD)[tf.newaxis, :, :],
                        #                                            tf.matmul(Sigma, cD[tf.newaxis, :, :])))
                        # variancesR = tf.linalg.diag_part(tf.matmul(tf.transpose(cR)[tf.newaxis, :, :],
                        #                                            tf.matmul(Sigma, cR[tf.newaxis, :, :])))
                        ## Variances for iota_means
                        # variancesDm = tf.reduce_sum(tf.gather(tf.math.square(model.iota_mean_varfam.scale), indDm), axis=1)
                        # variancesRm = tf.reduce_sum(tf.gather(tf.math.square(model.iota_mean_varfam.scale), indRm), axis=1)

                        ## Bayesian p-value computation
                        # Ddistribution = tfp.distributions.Normal(loc=dem_fin, scale=tf.math.sqrt(variancesD))
                        # FD0 = Ddistribution.cdf(0)
                        # tensorD = tf.stack([1 - FD0, FD0])
                        # zerovarD = tf.where(variancesD==0)
                        # ccpD = tf.tensor_scatter_nd_update(2 * tf.reduce_min(tensorD, axis=0),
                        #                                       zerovarD,
                        #                                       tf.ones(len(zerovarD)))
                        ccpD = linear_combination_CCP(cD, model.iota_varfam.location, Sigma, mu0=0.0, separately=True)
                        ccpDround = tf.math.round(1000 * ccpD) / 1000
                        print("ccpD")
                        print(ccpDround)

                        # Rdistribution = tfp.distributions.Normal(loc=rep_fin, scale=tf.math.sqrt(variancesR))
                        # FR0 = Rdistribution.cdf(0)
                        # tensorR = tf.stack([1 - FR0, FR0])
                        # zerovarR = tf.where(variancesR == 0)
                        # ccpR = tf.tensor_scatter_nd_update(2 * tf.reduce_min(tensorR, axis=0),
                        #                                       zerovarR,
                        #                                       tf.ones(len(zerovarR)))
                        ccpR = linear_combination_CCP(cR, model.iota_varfam.location, Sigma, mu0=0.0, separately=True)
                        ccpRround = tf.math.round(1000 * ccpR) / 1000
                        print("ccpR")
                        print(ccpRround)

                        ## Bayesian p-value computation for iota_means
                        # Dmdistribution = tfp.distributions.Normal(loc=np.float32(mdem_fin),
                        #                                           scale=np.float32(np.concatenate(
                        #                                               ([0.0], tf.math.sqrt(variancesDm).numpy()))))
                        # FDm0 = Dmdistribution.cdf(0)
                        # tensorDm = tf.stack([1 - FDm0, FDm0])
                        # zerovarDm = tf.where(variancesDm == 0)
                        # ccpDm = tf.tensor_scatter_nd_update(2 * tf.reduce_min(tensorDm, axis=0),
                        #                                        zerovarDm,
                        #                                        tf.ones(len(zerovarDm)))
                        ccpDm = linear_combination_CCP(cD,
                                                             model.iota_mean_varfam.location[tf.newaxis, :],
                                                             Sigmam[tf.newaxis, :, :],
                                                             mu0=0.0, separately=True)
                        ccpDmround = tf.math.round(1000 * ccpDm) / 1000
                        print("ccpDm")
                        print(ccpDmround)

                        # Rmdistribution = tfp.distributions.Normal(loc=np.float32(mrep_fin),
                        #                                           scale=np.float32(np.concatenate(
                        #                                               ([0.0], tf.math.sqrt(variancesRm).numpy()))))
                        # FRm0 = Rmdistribution.cdf(0)
                        # tensorRm = tf.stack([1 - FRm0, FRm0])
                        # zerovarRm = tf.where(variancesRm == 0)
                        # ccpRm = tf.tensor_scatter_nd_update(2 * tf.reduce_min(tensorRm, axis=0),
                        #                                        zerovarRm,
                        #                                        tf.ones(len(zerovarRm)))
                        ccpRm = linear_combination_CCP(cR,
                                                             model.iota_mean_varfam.location[tf.newaxis, :],
                                                             Sigmam[tf.newaxis, :, :],
                                                             mu0=0.0, separately=True)
                        ccpRmround = tf.math.round(1000 * ccpRm) / 1000
                        print("ccpRm")
                        print(ccpRmround)

                        ## P-value as categorical - ***, **, *, .
                        ccpDcut = np.digitize(ccpD.numpy(), bins=[0, 0.001, 0.01, 0.05, 0.1, 1])
                        ccpRcut = np.digitize(ccpR.numpy(), bins=[0, 0.001, 0.01, 0.05, 0.1, 1])
                        ccpDmcut = np.digitize(ccpDm.numpy(), bins=[0, 0.001, 0.01, 0.05, 0.1, 1])
                        ccpRmcut = np.digitize(ccpRm.numpy(), bins=[0, 0.001, 0.01, 0.05, 0.1, 1])
                        signif_codes = {1: "***", 2: "**", 3: "*", 4: ".", 5: "", 6: ""}

                        ## Combine values for Democrats and Republicans into one, separate by zeros
                        dem_mdem_fin = tf.concat([dem_fin,
                                                  tf.zeros(dem_fin[0, :].shape)[tf.newaxis, :],
                                                  mdem_fin[tf.newaxis, :]], axis=0)

                        rep_mrep_fin = tf.concat([rep_fin,
                                                  tf.zeros(rep_fin[0, :].shape)[tf.newaxis, :],
                                                  mrep_fin[tf.newaxis, :]], axis=0)
                        if include_party:
                            combined_mean = tf.concat([dem_mdem_fin,
                                                       tf.zeros(dem_mdem_fin[:, 0].shape)[:, tf.newaxis],
                                                       rep_mrep_fin], axis=1)
                        else:
                            combined_mean = tf.concat([dem_mdem_fin, rep_mrep_fin], axis=1)
                        if category == 'religion':
                            cat = 'RELIGION'
                        else:
                            cat = category
                        DRfreq = pd.crosstab(index=author_info[cat],
                                             columns=author_info["party"])
                        labels = []
                        for l in range(len(cat_names)):
                            labels += [cat_names[l] + " (" + str(DRfreq['D'][cat_labels[l]].sum()) + ")"]
                        if include_party:
                            labels += [""]
                        for l in range(len(cat_names)):
                            labels += [cat_names[l] + " (" + str(DRfreq['R'][cat_labels[l]].sum()) + ")"]
                        # labels = Dcat_names + [""] + Rcat_names
                        # Define the plot
                        fig, ax = plt.subplots(figsize=(13, 7))
                        # Set the font size and the distance of the title from the plot
                        plt.title("Coefficients for "+category, fontsize=18)
                        ax.title.set_position([0.5, 1.05])
                        # new pandas dataframe to name the rows and columns
                        named_table = pd.DataFrame(combined_mean,
                                                   index=[str(i) for i in range(0, model.num_topics)] + ["", "Averaged"],
                                                   columns=labels)
                        # Use the heatmap function from the seaborn package
                        sns.heatmap(named_table, vmin=-1.0, vmax=1.0, fmt="", cmap='bwr', linewidths=0.30, ax=ax)
                        ax.text(0.0, -0.05, "Democrats", ha="left")
                        ax.text(2*len(cat_names) + int(include_party), -0.05, "Republicans", ha="right")
                        for coef in range(len(cat_names)):
                            for k in range(model.num_topics):
                                ax.text(coef+0.5, k+0.5,
                                        signif_codes[ccpDcut[k, coef]], ha="center", va="center")
                                ax.text(coef+len(cat_names)+int(include_party)+0.5, k+0.5,
                                        signif_codes[ccpRcut[k, coef]], ha="center", va="center")

                                # ax.text(coef+0.5, k+0.5, str(ccpDround[k, coef].numpy()), ha="center", va="center")
                                # ax.text(coef+len(cat_names)+1.5, k+0.5, str(ccpRround[k, coef].numpy()), ha="center", va="center")
                                # ax.scatter(coef+0.5, k+0.5,
                                #            s=100*max(-tf.math.log(ccpDround[k, coef].numpy())+tf.math.log(0.05), 0),
                                #            marker="*", c="gold")
                                # ax.scatter(coef+len(cat_names)+1.5, k+0.5,
                                #            s=100*max(-tf.math.log(ccpRround[k, coef].numpy())+tf.math.log(0.05), 0),
                                #            marker="*", c="gold")
                            ax.text(coef + 0.5, model.num_topics + 1.0 + 0.5,
                                    signif_codes[ccpDmcut[0, coef]], ha="center", va="center")
                            ax.text(coef + len(cat_names) + int(include_party) + 0.5, model.num_topics + 1.0 + 0.5,
                                    signif_codes[ccpRmcut[0, coef]], ha="center", va="center")
                        # Display the heatmap
                        # plt.show()
                        # Save the heatmap
                        plt.gcf().set_size_inches((8+len(cat_names), 13))
                        plt.savefig(os.path.join(fig_dir, 'coefficients_by_party_'+include_party_label+'_'+category+'.png'),
                                    bbox_inches='tight')
                        plt.close()


        ### Authors: verbosity against topic-specific location
        for k in range(model.ideal_dim[1]):
            for category in ['party', 'gender', 'region', 'generation', 'exper_cong', 'religion']:
                if category in author_info.columns:
                    plot_verbosity_vs_location(model, fig_dir, k, all_author_indices, author_map, author_info[category], size=(15, 15))
                    plot_ideal_points_as_distribution(model, fig_dir, k, author_info[category], xlim=(-2.5, 2.5), size=(10, 5))
            plot_ideal_points_thin(model, fig_dir, k, author_info['party'], xlim=(-1.6, 1.6), size=(5, 0.25))

    elif data_name == 'cze_senate':
        ###------------------------------------###
        ###  PLOTS FOR THE cze_senate DATASET  ###
        ###------------------------------------###
        ### Ideological positions
        if model.prior_choice["ideal_dim"] == "ak" and covariates == 'party':
            author_party = author_info['party']
            author_party[~author_party.isin(["ODS", "CSSD", "ANO", "TOP09", "STAN", "KDU-CSL", "KSCM", "NK"])] = "Other"
            party_by_mean = False
            weighted_mean = True
            parties = ["ODS", "CSSD", "ANO", "TOP09", "STAN", "KDU-CSL", "KSCM", "NK", "Other"]
            Eqtheta = model.theta_varfam.shape / model.theta_varfam.rate
            weights = tf.math.unsorted_segment_mean(Eqtheta, all_author_indices, model.num_authors)
            weights_row = weights / tf.reduce_sum(weights, axis=0)[tf.newaxis, :]
            weights_column = weights / tf.reduce_sum(weights, axis=1)[:, tf.newaxis]
            weights_total = weights / tf.reduce_sum(weights)

            for transposed in [True, False]:
                if transposed:
                    label_t = "_transposed"
                else:
                    label_t = ""
                # weights.shape: [num_authors, num_topics]
                x = {}
                if party_by_mean:
                    if weighted_mean:
                        # weighing ideal locations:
                        wloc = model.ideal_varfam.location * weights_row
                        # Party means
                        for p in parties:
                            x[p] = tf.reduce_sum(tf.gather(wloc, tf.where(author_party == p), axis=0), axis=0)[0, :]
                        how_party = 'party_by_weighted_average'
                    else:
                        # Party means
                        for p in parties:
                            x[p] = tf.reduce_mean(
                                tf.gather(model.ideal_varfam.location, tf.where(author_party == p), axis=0), axis=0)[0,
                                   :]
                        how_party = 'party_by_average'
                else:
                    for ip in range(len(parties)):
                        p = parties[ip]
                        if p == "Other":
                            x[p] = model.iota_varfam.location[:, 0]
                        else:
                            x[p] = model.iota_varfam.location[:, 0] + model.iota_varfam.location[:, ip + 1]
                    how_party = 'party_by_iota'

                intercept = tf.zeros(model.iota_varfam.location[:, 0].shape)
                xiota = tf.stack((intercept,
                                  intercept + model.iota_varfam.location[:, 1],
                                  intercept + model.iota_varfam.location[:, 2],
                                  intercept + model.iota_varfam.location[:, 3],
                                  intercept + model.iota_varfam.location[:, 4],
                                  intercept + model.iota_varfam.location[:, 5],
                                  intercept + model.iota_varfam.location[:, 6],
                                  intercept + model.iota_varfam.location[:, 7],
                                  intercept + model.iota_varfam.location[:, 8]
                                  ))
                permutation = tf.argsort(tf.math.reduce_std(xiota, axis=0), direction='DESCENDING')
                for p in parties:
                    x[p] = tf.gather(x[p], permutation)

                if weighted_mean:
                    meanlocs = tf.reduce_sum(model.ideal_varfam.location * weights_column, axis=1)
                    if transposed:
                        meanlab = 'WA'
                    else:
                        meanlab = 'WghtAvrg'
                else:
                    meanlocs = tf.reduce_mean(model.ideal_varfam.location, axis=1)
                    if transposed:
                        meanlab = 'Avg'
                    else:
                        meanlab = 'Averaged'
                x0 = {}
                for p in parties:
                    x0[p] = tf.reduce_mean(tf.gather(meanlocs, tf.where(author_party == p), axis=0), axis=0)
                locs = tf.gather(model.ideal_varfam.location, permutation, axis=1)

                markercat = {"ODS": "o", "CSSD": "o", "ANO": "o", "TOP09": "o",
                             "STAN": "o", "KDU-CSL": "o", "KSCM": "o", "NK": "o", "Other": "+"}
                colorcat = {"ODS": "blue", "CSSD": "orange", "ANO": "lightblue", "TOP09": "purple",
                            "STAN": "green", "KDU-CSL": "yellow", "KSCM": "red", "NK": "grey", "Other": "grey"}
                topics = -tf.range(model.num_topics) - 1
                topicslab = permutation.numpy().astype(str)

                if transposed:
                    plt.hlines(0.0, -model.num_topics - 1, 1.5, colors='grey', linestyles='--')
                    for i in range(model.num_authors):
                        plt.scatter(y=meanlocs[i], x=0.0, color=colorcat[author_party[i]],
                                    marker=markercat[author_party[i]])
                    plt.vlines(-0.5, -1.4, 1.9, colors='grey', linestyles='-')
                    for i in range(model.num_authors):
                        plt.scatter(y=locs[i, :], x=topics,
                                    color=colorcat[author_party[i]], marker=markercat[author_party[i]])
                    plt.xticks(np.append(topics.numpy(), [0]), np.append(topicslab, [meanlab]))
                    plt.ylim((-1.5, 2.0))
                    plt.ylabel('Ideological position')
                    # Party means
                    for p in parties:
                        plt.scatter(y=x[p], x=topics, color=colorcat[p], marker='x')
                        plt.scatter(y=x0[p], x=0, color=colorcat[p], marker='x')
                    plt.gcf().set_size_inches((11, 5))
                else:
                    plt.vlines(0.0, -model.num_topics - 1, 1.5, colors='grey', linestyles='--')
                    for i in range(model.num_authors):
                        plt.scatter(x=meanlocs[i], y=0.0, color=colorcat[author_party[i]],
                                    marker=markercat[author_party[i]])
                    plt.hlines(-0.5, -1.4, 1.9, colors='grey', linestyles='-')
                    for i in range(model.num_authors):
                        plt.scatter(x=locs[i, :], y=topics,
                                    color=colorcat[author_party[i]], marker=markercat[author_party[i]])
                    plt.yticks(np.append(topics.numpy(), [0]), np.append(topicslab, [meanlab]))
                    plt.xlim((-1.5, 2.0))
                    plt.xlabel('Ideological position')
                    # Party means
                    for p in parties:
                        plt.scatter(x=x[p], y=topics, color=colorcat[p], marker='x')
                        plt.scatter(x=x0[p], y=0, color=colorcat[p], marker='x')
                    plt.gcf().set_size_inches((7.5, 8))
                plt.box(False)
                plt.margins(x=0, y=0)
                plt.tight_layout()
                # plt.show()
                plt.savefig(os.path.join(fig_dir, 'ideal_points_' + how_party + label_t + '.png'),
                            bbox_inches='tight')
                plt.close()

        if covariates == 'party':
            intercept = tf.zeros(model.iota_varfam.location[:, 0].shape)
            x = tf.stack((intercept,
                          intercept + model.iota_varfam.location[:, 1],
                          intercept + model.iota_varfam.location[:, 2],
                          intercept + model.iota_varfam.location[:, 3],
                          intercept + model.iota_varfam.location[:, 4],
                          intercept + model.iota_varfam.location[:, 5],
                          intercept + model.iota_varfam.location[:, 6],
                          intercept + model.iota_varfam.location[:, 7],
                          intercept + model.iota_varfam.location[:, 8]
                          ))
            topics = tf.math.top_k(tf.math.reduce_std(x, axis=0), ntopics).indices
            plot_wordclouds(model, os.path.join(fig_dir, 'top5_party_wordclouds.png'),
                            nwords, topics.numpy(), vocabulary, logscale=False, size=(10, 10))
            plot_wordclouds(model, os.path.join(fig_dir, 'top5_party_wordclouds_logscale.png'),
                            nwords, topics.numpy(), vocabulary, logscale=True, size=(10, 10))
            plot_heatmap_coefficients(tf.transpose(x), "Party coefficients",
                                      os.path.join(fig_dir, 'coefficients_region.png'),
                                      xtick=["Other", "ODS", "ČSSD", "ANO", "TOP-09",
                                             "STAN", "Piráti", "KDU-ČSL", "KSČM", "Nezávislý"],
                                      ytick=range(model.num_topics))
    elif data_name == 'pharma':
        ###--------------------------------###
        ###  PLOTS FOR THE pharma DATASET  ###
        ###--------------------------------###
        for k in range(model.ideal_dim[1]):
            for category in ['Nutzen', 'Company', 'Anwendungsgebiet.general']:
                if category in author_info.columns:
                    x = model.ideal_varfam.location[:, k].numpy()
                    plot_location_jittered(model, x, fig_dir, k,
                                           author_info['Wirkstoff'].to_numpy(),
                                           author_info[category],
                                           size=(15, 8))

        # Company coefficients
        if covariates == 'additive':
            for k in range(model.ideal_dim[1]):
                x = model.iota_varfam.location[k, 2:(2 + 159)].numpy()
                companies = author_info['Company'].drop_duplicates().to_numpy()
                novartis = np.where(companies == 'Novartis Pharma GmbH')[0]
                companies = np.delete(companies, novartis)
                plot_coef_jittered(model, x, fig_dir, k, companies, 'Company', size=(15, 8))

        # Anwendungsgebiet.general coefficients
        if covariates == 'additive':
            for k in range(model.ideal_dim[1]):
                x = model.iota_varfam.location[k, (2 + 159):(2 + 159 + 34)].numpy()
                ans = author_info['Anwendungsgebiet.general'].drop_duplicates().to_numpy()
                baseline = np.where(ans == 'onkologische Erkrankungen')[0]
                ans = np.delete(ans, baseline)
                plot_coef_jittered(model, x, fig_dir, k, ans, 'Anwendungsgebiet.general', size=(15, 8))

    elif data_name == 'fomc':
        ###--------------------------------###
        ###   PLOTS FOR THE fomc DATASET   ###
        ###--------------------------------###
        # intercept = model.iota_varfam.location[:, 0]
        intercept = tf.zeros(model.iota_varfam.location[:, 0].shape)
        if model.iota_varfam.family != "deterministic":
            print("Iota coefficients:")
            print(model.iota_varfam.location)

        ### Authors: verbosity against topic-specific location
        for k in range(model.ideal_dim[1]):
            for category in ['fgender', 'title', 'year', 'flength', 'flaughter']:
                if category in author_info.columns:
                    plot_verbosity_vs_location(model, fig_dir, k, all_author_indices, author_map, author_info[category],
                                               size=(15, 15))
                    plot_ideal_points_as_distribution(model, fig_dir, k, author_info[category], xlim=(-2.5, 2.5),
                                                      size=(10, 5))

        ### Development of the ideological position of speakers in time:
        surnames = np.unique(author_info['surname'])
        colors = {}
        for s in surnames:
            # random color
            colors[s] = mcolors.CSS4_COLORS[np.random.choice(list(mcolors.CSS4_COLORS.keys()), 1)[0]]
        splitted_map = pd.DataFrame(author_map)[0].str.split("_", expand=True)
        splitted_map["date"] = pd.to_datetime(splitted_map[1].apply(change_format))
        for k in range(model.ideal_dim[1]):
            for s in surnames:
                sindices = (splitted_map[0] == s)
                dates = splitted_map["date"][sindices]
                #ideals = tf.boolean_mask(tf.gather(ideal, k, axis=1), sindices)
                ideals = tf.boolean_mask(tf.gather(model.ideal_varfam.location, k, axis=1), sindices)
                plt.plot(dates, ideals, linestyle='-', color=colors[s])
                rs = np.random.choice(range(len(dates)), 2)
                for r in rs:
                    plt.text(dates.to_numpy()[r], ideals.numpy()[r], s, color=colors[s], ha='center', va='bottom')
            plt.xlabel("Date")
            plt.ylabel("Ideological position")
            plt.title("Topic " + str(k))
            plt.tight_layout()
            plt.gcf().set_size_inches((15, 10))
            # plt.show()
            plt.savefig(os.path.join(fig_dir, 'ideal_points_in_time_' + str(k) + '.png'))
            plt.close()


    else:
        raise ValueError('Unrecognized data_name.')


def hist_word_counts(countsSparse, countsNonzero, fig_dir, name_start="orig_"):
    """Create histograms that summarize the document-term matrix.

    Args:
        countsSparse: Document-term matrix containing word counts in a sparse format.
        countsNonzero: 0/1 indicators whether the word count is nonzero or not.
        fig_dir: Directory to save the plots.
        name_start: How should the plots be named at the beggining.
    """
    # Histogram of counts
    count_freq = tf.sparse.bincount(countsSparse)
    f0 = tf.cast(tf.reduce_prod(countsSparse.shape) - countsSparse.indices.shape[0],
                 dtype=tf.int64)  # frequency of 0
    f = tf.concat(([f0], count_freq.values), axis=0)
    x = tf.concat(([0], count_freq.indices[:, 0]), axis=0)
    plt.bar(x.numpy(), np.log(f.numpy()))
    plt.title("Log-frequencies of word counts")
    plt.ylabel("Log-frequency")
    plt.xlabel("Word count in a document")
    plt.savefig(os.path.join(fig_dir, name_start + "log_hist_counts.png"))
    plt.close()

    # Word counts summed over documents
    plt.hist(tf.sparse.reduce_sum(countsSparse, axis=0), histtype='step', bins=100)  # summed over documents
    plt.title("Word counts across all documents")
    plt.xlabel("Word frequencies")
    plt.savefig(os.path.join(fig_dir, name_start + "hist_counts_words.png"))
    plt.close()

    # Word counts summed over words.
    plt.hist(tf.sparse.reduce_sum(countsSparse, axis=1), histtype='step', bins=100)  # summed over words
    plt.title("Word counts across all words")
    plt.xlabel("Document length")
    plt.savefig(os.path.join(fig_dir, name_start + "hist_counts_docs.png"))
    plt.close()

    # Number of different words used in the document
    x = tf.sparse.reduce_sum(countsNonzero, axis=1)
    plt.hist(x, histtype='step', bins=100)
    plt.title("Words in a document")
    plt.xlabel("Word count per document")
    plt.savefig(os.path.join(fig_dir, name_start + "hist_word_in_doc.png"))
    plt.close()

    # Number of documents in which the word appears
    x = tf.sparse.reduce_sum(countsNonzero, axis=0)
    plt.hist(x, histtype='step', bins=100)
    plt.title("Documents with given word")
    plt.xlabel("Document count per word")
    plt.savefig(os.path.join(fig_dir, name_start + "hist_doc_having_word.png"))
    plt.close()

