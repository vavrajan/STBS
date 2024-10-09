# Import global packages
import os
import time

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import re
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp
from STBS.code.plotting_functions import hist_word_counts



def build_input_pipeline_hein_daily(data_dir,
                                    batch_size,
                                    random_state,
                                    fig_dir=None,
                                    counts_transformation="nothing",
                                    addendum=''):
    """Load data and build iterator for minibatches.
    Specific to hein-daily data.

    Args:
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts.npz`, `author_indices.npy`, `author_map.txt`, and `vocabulary.txt`.
            All also contain 'addendum' at the name of the file.
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.

    """
    counts = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + ".npz"))
    num_documents, num_words = counts.shape
    author_indices = np.load(
    os.path.join(data_dir, "author_indices" + addendum + ".npy")).astype(np.int32)
    author_data = np.loadtxt(os.path.join(data_dir, "author_map" + addendum + ".txt"),
                             dtype=str, delimiter=" ", usecols=[0, 1, -1])
    # author_party = np.char.replace(author_data[:, 2], '(', '')
    # author_party = np.char.replace(author_party, ')', '')
    author_map = np.char.add(author_data[:, 0], author_data[:, 1])
    documents = random_state.permutation(num_documents)
    shuffled_author_indices = author_indices[documents]
    shuffled_counts = counts[documents]
    if counts_transformation == "nothing":
        count_values = shuffled_counts.data
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + shuffled_counts.data))
    else:
        raise ValueError("Unrecognized counts transformation.")
    shuffled_counts = tf.SparseTensor(
        indices=np.array(shuffled_counts.nonzero()).T,
        values=count_values,
        dense_shape=shuffled_counts.shape)
    if fig_dir is not None:
        shuffled_countsSparse = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=count_values.astype(np.int64),
            dense_shape=shuffled_counts.dense_shape)
        shuffled_countsNonzero = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=tf.fill(count_values.shape, 1),
            dense_shape=shuffled_counts.dense_shape)  # replace nonzero counts with 1 (indicator of non-zeroness)
        # counts = tf.sparse.to_dense(countsSparse)
        hist_word_counts(shuffled_countsSparse, shuffled_countsNonzero, fig_dir, name_start="orig_")

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"document_indices": documents,
          "author_indices": shuffled_author_indices}, shuffled_counts))
    # dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)
    # Prefetching to speed up computations
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    file = open(os.path.join(data_dir, "vocabulary" + addendum + ".txt"))
    lines = file.readlines()
    voc = np.array(lines)
    vocabulary = np.char.replace(voc, '\n', '')
    ## Loading with delimiter as new line does not work
    # vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"),
    #                         dtype=str,
    #                         delimiter="\n",
    #                         comments="<!-")
    shuffled_author_indices = tf.constant(shuffled_author_indices)
    # author_info = pd.read_csv(os.path.join(data_dir, "author_detailed_info_with_religion" + addendum + ".csv"))
    author_info = pd.read_csv(os.path.join(data_dir, "author_detailed_info" + addendum + ".csv"))

    return dataset, documents, shuffled_author_indices, vocabulary, author_map, author_info


def build_input_pipeline_cze_senate(data_dir,
                                    batch_size,
                                    random_state,
                                    fig_dir=None,
                                    counts_transformation="nothing",
                                    addendum=''):
    """Load data and build iterator for minibatches.
    Specific to cze_senate data.

    Args:
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts.npz`, `speech_data.csv`, both also contain 'addendum' at the name of the file.
            And then 'senator_data.csv' which is common to all addenda.
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.

    """
    ### Import clean datasets
    print("Importing the data from clean directory.")
    counts = sparse.load_npz(os.path.join(data_dir, 'counts' + addendum + '.npz'))
    num_documents, num_words = counts.shape
    print("Number of documents: " + str(num_documents))
    print("Number of words: " + str(num_words))
    speech_data = pd.read_csv(os.path.join(data_dir, 'speech_data' + addendum + '.csv'), encoding='utf-8')
    author_info = pd.read_csv(os.path.join(data_dir, "senator_data.csv"))
    author_counts = speech_data['author_id'].value_counts()
    print(author_counts)
    num_authors = len(author_counts)

    ### Processing information about authors
    # Current format of the data:
    #  id | name | obvod | strana | od | do | party
    #   1 |  AAA |     1 |    ODS | 98 | 04 |   ODS
    #   1 |  AAA |     1 |    ODS | 96 | 98 |   ODS
    #   2 |  BBB |     2 |   ČSSD | 96 | 00 |  CSSD
    #   3 |  CCC |     3 |   ČSSD | 96 | 02 |  CSSD
    #   i |  XXX |     x |     P1 | yy | YY |     P
    #   i |  XXX |     x |     P2 | yy | YY |     P
    # How to determine strana --> party if P1 != P2 ?
    # ---> paste behind each other (there are some coalitions anyway)
    #  id | name | obvod | strana |
    #   1 |  AAA |     1 |    ODS |
    #   2 |  BBB |     2 |   ČSSD |
    #   3 |  CCC |     3 |   ČSSD |
    #   i |  XXX |     x |   P1P2 |
    print("Number of authors: " + str(num_authors))
    print(author_info.head())
    print(author_info.tail())
    # author_info = author_info.groupby('number')['strana'].apply('*'.join).reset_index()
    author_info = author_info.groupby('number')['party'].first().reset_index()
    print(author_info.head())
    print(author_info.tail())
    print(author_info.shape)
    print(author_counts.index)
    print(author_info['number'])
    author_info = author_info[author_info['number'].isin(author_counts.index)].reset_index()
    print(author_info)

    ### Shuffling the dataset
    author_ids = speech_data['author_id'].to_numpy()
    # print("author_ids")
    # print(author_ids)
    # print(len(author_ids))
    id_to_index = pd.DataFrame(author_info.index, index=author_info['number'])
    # print("id_to_index")
    # print(id_to_index)
    author_indices = id_to_index.loc[author_ids]
    # print(author_indices)
    author_indices = author_indices.to_numpy()[:, 0]
    # print("author_indices")
    # print(author_indices)
    # print(len(author_indices))

    permutation = random_state.permutation(num_documents)
    shuffled_author_indices = author_indices[permutation]
    shuffled_counts = counts[permutation]
    if counts_transformation == "nothing":
        count_values = shuffled_counts.data
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + shuffled_counts.data))
    else:
        raise ValueError("Unrecognized counts transformation.")
    shuffled_counts = tf.SparseTensor(
        indices=np.array(shuffled_counts.nonzero()).T,
        values=count_values,
        dense_shape=shuffled_counts.shape)
    if fig_dir is not None:
        shuffled_countsSparse = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=count_values.astype(np.int64),
            dense_shape=shuffled_counts.dense_shape)
        shuffled_countsNonzero = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=tf.fill(count_values.shape, 1),
            dense_shape=shuffled_counts.dense_shape)  # replace nonzero counts with 1 (indicator of non-zeroness)
        hist_word_counts(shuffled_countsSparse, shuffled_countsNonzero, fig_dir, name_start="orig_")

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"document_indices": permutation,
          "author_indices": shuffled_author_indices}, shuffled_counts))
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    file = open(os.path.join(data_dir, 'vocabulary' + addendum + '.txt'))
    lines = file.readlines()
    voc = np.array(lines)
    vocabulary = np.char.replace(voc, '\n', '')
    all_author_indices = tf.constant(shuffled_author_indices)
    author_map = author_info['name'].to_numpy()

    return dataset, permutation, all_author_indices, vocabulary, author_map, author_info


def build_input_pipeline_pharma(data_dir,
                                batch_size,
                                random_state,
                                fig_dir=None,
                                counts_transformation="nothing",
                                addendum=''):
    """Load data and build iterator for minibatches.
    Specific to pharma data.

    Args:
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts.npz`, `author_indices.npy`, and `vocabulary.txt`.
            All also contain 'addendum' at the name of the file.
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.

    """
    counts = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + ".npz"))
    num_documents, num_words = counts.shape
    author_indices = np.load(os.path.join(data_dir, "author_indices" + addendum + ".npy")).astype(np.int32)
    documents = random_state.permutation(num_documents)
    shuffled_author_indices = author_indices[documents]
    shuffled_counts = counts[documents]
    if counts_transformation == "nothing":
        count_values = shuffled_counts.data
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + shuffled_counts.data))
    else:
        raise ValueError("Unrecognized counts transformation.")
    shuffled_counts = tf.SparseTensor(
        indices=np.array(shuffled_counts.nonzero()).T,
        values=count_values,
        dense_shape=shuffled_counts.shape)
    if fig_dir is not None:
        shuffled_countsSparse = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=count_values.astype(np.int64),
            dense_shape=shuffled_counts.dense_shape)
        shuffled_countsNonzero = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=tf.fill(count_values.shape, 1),
            dense_shape=shuffled_counts.dense_shape)  # replace nonzero counts with 1 (indicator of non-zeroness)
        # counts = tf.sparse.to_dense(countsSparse)
        hist_word_counts(shuffled_countsSparse, shuffled_countsNonzero, fig_dir, name_start="orig_")

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"document_indices": documents,
          "author_indices": shuffled_author_indices}, shuffled_counts))
    # dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)
    # Prefetching to speed up commputations
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    file = open(os.path.join(data_dir, "vocabulary" + addendum + ".txt"))
    lines = file.readlines()
    voc = np.array(lines)
    vocabulary = np.char.replace(voc, '\n', '')
    ## Loading with delimiter as new line does not work
    # vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"),
    #                         dtype=str,
    #                         delimiter="\n",
    #                         comments="<!-")
    all_author_indices = tf.constant(shuffled_author_indices)

    # Import infodata about senators.
    author_info = pd.read_csv(os.path.join(data_dir, "author_info" + addendum + ".csv"))
    author_map = author_info['id']

    return dataset, documents, all_author_indices, vocabulary, author_map, author_info

def build_input_pipeline_fomc(data_dir,
                              batch_size,
                              random_state,
                              fig_dir=None,
                              counts_transformation="nothing",
                              addendum='',):
    """Load data and build iterator for minibatches.
    Specific to pharma data.

    Args:
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts.npz`, `author_indices.npy`, and `vocabulary.txt`.
            All also contain 'addendum' at the name of the file.
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.

    """
    counts = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + ".npz"))
    num_documents, num_words = counts.shape
    speech_data = pd.read_csv(os.path.join(data_dir, "speech_data" + addendum + ".csv"))
    # Combine author with meeting id to create author_meeting indices
    author_list = np.unique(speech_data['author_meeting']).tolist()
    meetings = np.unique(speech_data['meeting']).tolist()
    author_indices = np.array([author_list.index(am) for am in speech_data['author_meeting']])
    meeting_indices = np.array([meetings.index(m) for m in speech_data['meeting']])
    # number of appearances of '[Laughter]' and '[laughter]' within a speech
    speech_data['laughter'] = speech_data['Speech'].str.count("\\[Laughter\\]", flags=re.IGNORECASE)

    documents = random_state.permutation(num_documents)
    shuffled_author_indices = author_indices[documents]
    shuffled_counts = counts[documents]
    if counts_transformation == "nothing":
        count_values = shuffled_counts.data
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + shuffled_counts.data))
    else:
        raise ValueError("Unrecognized counts transformation.")
    shuffled_counts = tf.SparseTensor(
        indices=np.array(shuffled_counts.nonzero()).T,
        values=count_values,
        dense_shape=shuffled_counts.shape)
    if fig_dir is not None:
        shuffled_countsSparse = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=count_values.astype(np.int64),
            dense_shape=shuffled_counts.dense_shape)
        shuffled_countsNonzero = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=tf.fill(count_values.shape, 1),
            dense_shape=shuffled_counts.dense_shape)  # replace nonzero counts with 1 (indicator of non-zeroness)
        # counts = tf.sparse.to_dense(countsSparse)
        hist_word_counts(shuffled_countsSparse, shuffled_countsNonzero, fig_dir, name_start="orig_")

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"document_indices": documents,
          "author_indices": shuffled_author_indices}, shuffled_counts))
    # dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)
    # Prefetching to speed up commputations
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    file = open(os.path.join(data_dir, "vocabulary" + addendum + ".txt"))
    lines = file.readlines()
    voc = np.array(lines)
    vocabulary = np.char.replace(voc, '\n', '')
    all_author_indices = tf.constant(shuffled_author_indices)

    # Reduce speech_data into author_meeting_data
    author_info = speech_data.groupby('author_meeting')['gender', 'title', 'surname', 'meeting'].first().reset_index()
    meeting_length = tf.math.unsorted_segment_sum(tf.reduce_sum(counts.todense(), axis=1),
                                                  segment_ids=meeting_indices,
                                                  num_segments=len(meetings))
    breaks_length = np.quantile(meeting_length, q=[0, 0.25, 0.75, 1.0])
    breaks_length[0] = 0.0
    fmeeting_length = pd.cut(meeting_length, breaks_length, labels=['short', 'medium', 'long'])
    meeting_within_author_indices = np.array([meetings.index(m) for m in author_info['meeting']])
    meeting_laughter = tf.math.unsorted_segment_sum(speech_data['laughter'].to_numpy(),
                                                    segment_ids=meeting_indices,
                                                    num_segments=len(meetings))
    breaks_laughter = np.quantile(meeting_laughter, q=[0, 0.25, 0.75, 1.0])
    breaks_laughter[0] = 0.0
    fmeeting_laughter = pd.cut(meeting_laughter, breaks_laughter, labels=['serious', 'medium', 'relaxed'])

    author_info['year'] = author_info["meeting"].astype(str).str[:4]
    author_info['fgender'] = pd.cut(author_info['gender'], [-0.5, 0.5, 1.5], labels=['F', 'M'])
    author_info['meeting_length'] = tf.gather(meeting_length, meeting_within_author_indices)
    author_info['flength'] = tf.gather(fmeeting_length, meeting_within_author_indices)
    author_info['flength'] = author_info['flength'].str.decode('utf-8')
    author_info['flaughter'] = tf.gather(fmeeting_laughter, meeting_within_author_indices)
    author_info['flaughter'] = author_info['flaughter'].str.decode('utf-8')
    author_map = np.array(author_list)

    return dataset, documents, all_author_indices, vocabulary, author_map, author_info


def build_input_pipeline(data_name, data_dir, batch_size, random_state, fig_dir, counts_transformation, addendum):
    """Triggers the right build_input_pipeline depending on the current dataset.

    Args:
        data_name: String containing the name of the dataset. Important to choose build_input_pipeline function.
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts.npz`, `author_indices.npy`, `author_map.txt`, and `vocabulary.txt`.
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.

    Returns:
        dataset: Batched document_indices and author_indices.
        shuffled_counts: Counts shuffled according to permutation of documents.
        permutation: Permutation of the documents.
        shuffled_author_indices: Indices of the author of the documents after permutation.
        vocabulary: A vector of strings with the actual word-terms for word dimension.
        author_map: A vector of string  with author names (or labels).
        author_info: A pandas DataFrame containig author-level covariates.

    """
    if data_name == "hein-daily":
        dataset, permutation, shuffled_author_indices, vocabulary, author_map, author_info = build_input_pipeline_hein_daily(
            data_dir, batch_size, random_state, fig_dir, counts_transformation, addendum)
    elif data_name == "cze_senate":
        dataset, permutation, shuffled_author_indices, vocabulary, author_map, author_info = build_input_pipeline_cze_senate(
            data_dir, batch_size, random_state, fig_dir, counts_transformation, addendum)
    elif data_name == "pharma":
        dataset, permutation, shuffled_author_indices, vocabulary, author_map, author_info = build_input_pipeline_pharma(
            data_dir, batch_size, random_state, fig_dir, counts_transformation, addendum)
    elif data_name == "fomc":
        dataset, permutation, shuffled_author_indices, vocabulary, author_map, author_info = build_input_pipeline_fomc(
            data_dir, batch_size, random_state, fig_dir, counts_transformation, addendum)
    else:
        raise ValueError("Unrecognized dataset name in order to load the data. "
                         "Implement your own 'build_input_pipeline_...' for your dataset.")
    return dataset, permutation, shuffled_author_indices, vocabulary, author_map, author_info
