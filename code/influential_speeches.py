import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse as sparse


def find_most_influential_speeches_hein_daily(model, data_name, data_dir, source_dir, txt_dir, addendum,
                                              how_influential,
                                              batch_size, nspeeches):
    """Finds the most influential speeches for the hein-daily dataset.
    First, find 'batch_size' speeches that maximize the variational means of document intensities (theta).
    This (and all following) is done for each topic separately.
    Then, for these speeches compute the log-likelihood ratio test statistic for comparing:
        null: ideological positions for these documents are 0,
        true: ideological positions are the ones estimated by STBS.

    Args:
        model: A STBS.
        data_name: A string containing the name of the data set.
        data_dir: Directory with clean data. (source_dir + clean/).
        source_dir: Directory where the original data (orig/) lies.
        txt_dir: Directory where to save the most influential speeches.
        addendum: A string that specifies which version of the data set.
        how_influential: A string that specifies how should we select the most influential speeches
        batch_size: The size of the batch. How many documents should be considered wrt maximizing Eq theta.
        nspeeches: Number of the most influential speeches to be saved.

    """
    ### Load all the speeches
    speeches = pd.read_csv(os.path.join(source_dir, 'orig', 'speeches_' + addendum + '.txt'),
                           encoding="ISO-8859-1",
                           sep="|", quoting=3,
                           on_bad_lines='warn')
    description = pd.read_csv(os.path.join(source_dir, 'orig', 'descr_' + addendum + '.txt'),
                              encoding="ISO-8859-1",
                              sep="|")
    speaker_map = pd.read_csv(os.path.join(source_dir, 'orig', addendum + '_SpeakerMap.txt'),
                              encoding="ISO-8859-1",
                              sep="|")
    merged_df = speeches.merge(description,
                               left_on='speech_id',
                               right_on='speech_id')
    df = merged_df.merge(speaker_map, left_on='speech_id', right_on='speech_id')

    speech_id_indices = np.load(os.path.join(data_dir, 'speech_id_indices' + addendum + '.npy'))
    # shuffled_speech_id_indices = speech_id_indices[permutation]

    if how_influential == 'theta':
        influence = model.get_Eqmean(model.theta_varfam)
        influence_ind = tf.repeat(speech_id_indices[:, tf.newaxis], model.num_topics, axis=1)
    elif how_influential == 'theta_then_loglik_ratio_test':
        ### First step - for each topic find speeches with highest variational means of thetas
        theta = model.get_Eqmean(model.theta_varfam)
        # transposition needed because tf.math.top_k finds maxima for the last dimension
        val, ind = tf.math.top_k(tf.transpose(theta), batch_size)
        sub_theta = tf.transpose(val)  # [B, V] transposition back

        ### Second step - computation of log-likelihood ratio test statistic
        beta = model.get_Eqmean(model.beta_varfam, log=False)  # [K, V]
        eta = model.get_Eqmean(model.eta_varfam, log=False)  # [K, V]
        ideal = model.get_Eqmean(model.ideal_varfam, log=False)  # [A, 1 or K]
        ideal_doc = tf.gather(ideal, model.all_author_indices, axis=0)  # [D, 1 or K]
        sub_ideal = tf.transpose(tf.gather(tf.transpose(ideal_doc), ind, axis=1, batch_dims=1))  # [B, 1 or K]

        rates_all_null = sub_theta[:, :, tf.newaxis] * beta[tf.newaxis, :, :]  # [B, K, V]
        ideological_term = tf.math.exp(eta[tf.newaxis, :, :] * sub_ideal[:, :, tf.newaxis])  # [B, K, V]
        rates = rates_all_null * ideological_term  # [B, K, V]
        rate_true = tf.reduce_sum(rates, axis=1)  # [B, V]
        rate_dif = rates_all_null - rates  # [B, K, V]

        # Subset counts for each topic separately: sparse[D, V] --> dense[B, K, V]
        # sub_counts = tf.transpose(tf.gather(tf.transpose(counts.todense()), ind, axis=1), [2, 1, 0])
        # sub_counts = tf.gather(tf.sparse.to_dense(counts), tf.transpose(ind), axis=0)
        counts = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + ".npz"))
        sub_counts = tf.gather(counts.todense(), tf.transpose(ind), axis=0)

        # Difference in log_probabilities of Poisson counts (true - null)
        loglik_dif = sub_counts * (tf.math.log(rate_true[:, tf.newaxis, :]) - tf.math.log(
            rate_true[:, tf.newaxis, :] + rate_dif)) + rate_dif  # [D, K, V]
        influence = tf.reduce_sum(loglik_dif, axis=2)  # [D, K]
        influence_ind = tf.transpose(tf.gather(speech_id_indices, ind))
    else:
        raise ValueError('Unrecognized choice for the selection of most influential speeches: ' + how_influential)

    ### Find the original speeches and save them into separate file
    for k in range(model.num_topics):
        docs = tf.math.top_k(influence[:, k], nspeeches).indices
        print("Topic: " + str(k))
        # auts = tf.gather(model.all_author_indices, docs)
        # ids = tf.gather(shuffled_speech_id_indices, docs)
        # ids = tf.gather(speech_id_indices, docs)
        ids = tf.gather(influence_ind[:, k], docs)
        top_unsorted = df.loc[df['speech_id'].isin(ids.numpy())]
        # print(top_unsorted)
        top = top_unsorted.set_index('speech_id').reindex(ids.numpy())
        # print(top)
        # top.speech_id = top.speech_id.astype("category")
        # top.speech_id = top.speech_id.cat.set_categories(ids.numpy())
        # top = top.sort_values('speech_id')
        with open(os.path.join(txt_dir, 'top' + str(nspeeches) + 'speeches_' + addendum + '_topic_' + str(k) + '.txt'), 'w') as file:
            file.write('----------------------------------------\n')
            file.write('The most influential speeches for topic ' + str(k) + '\n')
            file.write('----------------------------------------\n\n\n')
            for j in range(nspeeches):
                file.write('Speech ' + str(j) + '\n')
                file.write('-------------------\n')
                file.write('Author: ' + top['speaker'][top.index[j]] + '\n')
                file.write('Party: ' + top['party'][top.index[j]] + '\n')
                file.write('State: ' + top['state_y'][top.index[j]] + '\n')
                file.write('Date: ' + str(top['date'][top.index[j]]) + '\n')
                file.write('Speech: ' + top['speech'][top.index[j]] + '\n')
                file.write('-------------------\n\n\n')


def find_most_influential_speeches_cze_senate(model, data_name, data_dir, source_dir, txt_dir, addendum,
                                              how_influential, batch_size, nspeeches):
    """Finds the most influential speeches for sze_senate dataset.
    For now, just speeches with the highest variational means of theta parameter.
    """
    if how_influential == 'theta':
        influence = model.get_Eqmean(model.theta_varfam)
    else:
        raise ValueError('Unrecognized choice for the selection of most influential speeches: ' + how_influential)
    for k in range(model.num_topics):
        docs = tf.math.top_k(influence[:, k], nspeeches).indices
        with open(os.path.join(txt_dir, 'top' + str(nspeeches) + 'speeches_' + addendum + '_topic_' + str(k) + '.txt'), 'w') as file:
            file.write('----------------------------------------\n')
            file.write('The most influential speeches for topic ' + str(k) + '\n')
            file.write('----------------------------------------\n\n\n')
            for j in range(nspeeches):
                file.write('Speech ' + str(j) + '\n')
                file.write('-------------------\n')
                file.write('Speech index: ' + str(docs[j]) + '\n')
                file.write('-------------------\n\n\n')


def find_most_influential_speeches_pharma(model, data_name, data_dir, source_dir, txt_dir, addendum,
                                          how_influential, batch_size, nspeeches):
    """Finds the most influential speeches for pharma dataset.
    For now, just speeches with the highest variational means of theta parameter.
    """
    if how_influential == 'theta':
        influence = model.get_Eqmean(model.theta_varfam)
    else:
        raise ValueError('Unrecognized choice for the selection of most influential speeches: ' + how_influential)
    for k in range(model.num_topics):
        docs = tf.math.top_k(influence[:, k], nspeeches).indices
        with open(txt_dir + 'top' + str(nspeeches) + 'speeches_' + addendum + '_topic_' + str(k) + '.txt', 'w') as file:
            file.write('----------------------------------------\n')
            file.write('The most influential speeches for topic ' + str(k) + '\n')
            file.write('----------------------------------------\n\n\n')
            for j in range(nspeeches):
                file.write('Speech ' + str(j) + '\n')
                file.write('-------------------\n')
                file.write('Speech index: ' + str(docs[j]) + '\n')
                file.write('-------------------\n\n\n')

def find_most_influential_speeches_fomc(model, data_name, data_dir, source_dir, txt_dir, addendum,
                                        how_influential, batch_size, nspeeches):
    """Finds the most influential speeches for fomc dataset.
    For now, just speeches with the highest variational means of theta parameter.
    """
    speech_data = pd.read_csv(os.path.join(data_dir, "speech_data" + addendum + ".csv"))
    speech_id_indices = tf.constant(speech_data.index)
    if how_influential == 'theta':
        influence = model.get_Eqmean(model.theta_varfam)
        influence_ind = tf.repeat(speech_id_indices[:, tf.newaxis], model.num_topics, axis=1)
    elif how_influential == 'theta_then_loglik_ratio_test':
        theta = model.get_Eqmean(model.theta_varfam)
        # transposition needed because tf.math.top_k finds maxima for the last dimension
        val, ind = tf.math.top_k(tf.transpose(theta), batch_size)
        sub_theta = tf.transpose(val)  # [B, V] transposition back

        ### Second step - computation of log-likelihood ratio test statistic
        beta = model.get_Eqmean(model.beta_varfam, log=False)  # [K, V]
        eta = model.get_Eqmean(model.eta_varfam, log=False)  # [K, V]
        ideal = model.get_Eqmean(model.ideal_varfam, log=False)  # [A, 1 or K]
        ideal_doc = tf.gather(ideal, model.all_author_indices, axis=0)  # [D, 1 or K]
        sub_ideal = tf.transpose(tf.gather(tf.transpose(ideal_doc), ind, axis=1, batch_dims=1))  # [B, 1 or K]

        rates_all_null = sub_theta[:, :, tf.newaxis] * beta[tf.newaxis, :, :]  # [B, K, V]
        ideological_term = tf.math.exp(eta[tf.newaxis, :, :] * sub_ideal[:, :, tf.newaxis])  # [B, K, V]
        rates = rates_all_null * ideological_term  # [B, K, V]
        rate_true = tf.reduce_sum(rates, axis=1)  # [B, V]
        rate_dif = rates_all_null - rates  # [B, K, V]

        # Subset counts for each topic separately: sparse[D, V] --> dense[B, K, V]
        # sub_counts = tf.transpose(tf.gather(tf.transpose(counts.todense()), ind, axis=1), [2, 1, 0])
        # sub_counts = tf.gather(tf.sparse.to_dense(counts), tf.transpose(ind), axis=0)
        counts = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + ".npz"))
        sub_counts = tf.gather(counts.todense(), tf.transpose(ind), axis=0)

        # Difference in log_probabilities of Poisson counts (true - null)
        loglik_dif = sub_counts * (tf.math.log(rate_true[:, tf.newaxis, :]) - tf.math.log(
            rate_true[:, tf.newaxis, :] + rate_dif)) + rate_dif  # [D, K, V]
        influence = tf.reduce_sum(loglik_dif, axis=2)  # [D, K]
        influence_ind = tf.transpose(tf.gather(speech_id_indices, ind))
    else:
        raise ValueError('Unrecognized choice for the selection of most influential speeches: ' + how_influential)

    for k in range(model.num_topics):
        docs = tf.math.top_k(influence[:, k], nspeeches).indices
        print("Topic: " + str(k))
        ids = tf.gather(influence_ind[:, k], docs)
        top = speech_data.iloc[ids.numpy()]
        with open(os.path.join(txt_dir, 'top' + str(nspeeches) + 'speeches_' + addendum + '_topic_' + str(k) + '.txt'), 'w') as file:
            file.write('----------------------------------------\n')
            file.write('The most influential speeches for topic ' + str(k) + '\n')
            file.write('----------------------------------------\n\n\n')
            for j in range(nspeeches):
                file.write('Speech ' + str(j) + '\n')
                file.write('-------------------\n')
                file.write('Speech index: ' + str(docs[j]) + '\n')
                file.write('Speaker: ' + top['Speaker'][top.index[j]] + '\n')
                file.write('Meeting: ' + str(top['meeting'][top.index[j]]) + '\n')
                file.write('Speech: ' + top['Speech'][top.index[j]] + '\n')
                file.write('-------------------\n\n\n')

def find_most_influential_speeches(model, data_name, data_dir, source_dir, txt_dir, addendum,
                                   how_influential, batch_size, nspeeches):
    """Triggers function for finding the most influential speeches depending on the dataset."""
    if data_name == 'hein-daily':
        find_most_influential_speeches_hein_daily(model, data_name, data_dir, source_dir, txt_dir, addendum,
                                                  how_influential, batch_size, nspeeches)

    elif data_name == 'cze_senate':
        find_most_influential_speeches_cze_senate(model, data_name, data_dir, source_dir, txt_dir, addendum,
                                                  how_influential, batch_size, nspeeches)

    elif data_name == 'pharma':
        find_most_influential_speeches_pharma(model, data_name, data_dir, source_dir, txt_dir, addendum,
                                              how_influential, batch_size, nspeeches)
    elif data_name == 'fomc':
        find_most_influential_speeches_fomc(model, data_name, data_dir, source_dir, txt_dir, addendum,
                                            how_influential, batch_size, nspeeches)
    else:
        raise ValueError("Unrecognized data_name for finding the most influential speeches. "
                         "You have to create your own implementation within 'influential_speeches.py' before.")


