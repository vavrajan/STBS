import os
#import setup_utils as utils
import numpy as np
# import tensorflow as tf
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from collections import defaultdict

#Unabridged source code originally available at: https://github.com/keyonvafa/tbip
# difference to data quoting=3!!!

#data source : https://data.stanford.edu/congress_text#download-data
#Please download and unzip hein-daily.zip
#data diractory where Hein-Daily database is saved
data_name = 'hein-daily'

### Directories setup
project_dir = os.getcwd()
source_dir = os.path.join(project_dir, "data", data_name)

# As described in the docstring, the data directory must have the following
# files: counts.npz, author_indices.npy, vocabulary.txt, author_map.txt.
### My laptop
data_dir = os.path.join(source_dir, "orig")
save_dir = os.path.join(source_dir, "clean")



#predefined set of stopwords
# stopwords = set(np.loadtxt(os.path.join(data_dir, "stopwords.txt"),
#                            dtype=str, delimiter="\n")) #to be changed approrpriately wherever stopwords are stored

file = open(os.path.join(data_dir, "stopwords.txt"))
lines = file.readlines()
sw = np.array(lines)
stopwords = np.char.replace(sw, '\n', '')

#stopwords available at: https://github.com/keyonvafa/tbip/blob/master/setup/stopwords/senate_speeches.txt
#to be downloaded and saved to data_dir as defined above

#Parameters

#minimum number of speeches given by a senator
#default value 24
min_speeches = 24
#minimum number of senators using a bigram
#default value 10
min_authors_per_word = 10

#parameters for count vectorizer
min_df = 0.001 #minimum document frequency
max_df = 0.3 #maximum document frequency
stop_words = stopwords.tolist()
ngram_range = (2, 2) #bigrams only
token_pattern = "[a-zA-Z]+"

#Helper function
#source code originally available at: https://github.com/keyonvafa/tbip
#Count number of occurrences of each value in array of non-negative integers
#documentation: https://numpy.org/doc/stable/reference/generated/numpy.bincount.html

def bincount_2d(x, weights):
    _, num_topics = weights.shape
    num_cases = np.max(x) + 1
    counts = np.array(
      [np.bincount(x, weights=weights[:, topic], minlength=num_cases)
       for topic in range(num_topics)])
    return counts.T


# creating a complete vocabulary covering all the sessions
# for i in range(97, 115):
# Only session 114
for i in range(114, 115):
    if (i < 100):
        stri = '0' + str(i)
    else:
        stri = str(i)
    speeches = pd.read_csv(os.path.join(data_dir, 'speeches_' + stri + '.txt'),
                           encoding="ISO-8859-1",
                           sep="|", quoting=3,
                           on_bad_lines='warn')
    description = pd.read_csv(os.path.join(data_dir, 'descr_' + stri + '.txt'),
                              encoding="ISO-8859-1",
                              sep="|")
    speaker_map = pd.read_csv(os.path.join(data_dir, stri + '_SpeakerMap.txt'),
                              encoding="ISO-8859-1",
                              sep="|")

    merged_df = speeches.merge(description,
                               left_on='speech_id',
                               right_on='speech_id')
    df = merged_df.merge(speaker_map, left_on='speech_id', right_on='speech_id')

    # Only look at senate speeches.
    # to select speakers with speeches in the senate (includes Senators and House Reps)
    senate_df = df[df['chamber_x'] == 'S']
    # to select ONLY Senators uncomment the next line
    senate_df = senate_df[senate_df['chamber_y'] == 'S'] ##  here 7.2
    speaker = np.array(
        [' '.join([first, last]) for first, last in
         list(zip(np.array(senate_df['firstname']),
                  np.array(senate_df['lastname'])))])
    speeches = np.array(senate_df['speech'])
    speech_ids = np.array(senate_df['speech_id'])
    party = np.array(senate_df['party'])
    gender = np.array(senate_df['gender_y'])
    state = np.array(senate_df['state_y'])

    # Remove senators who make less than 24 speeches
    unique_speaker, speaker_counts = np.unique(speaker, return_counts=True)
    absent_speakers = unique_speaker[np.where(speaker_counts < min_speeches)]
    present_speakers = unique_speaker[np.where(speaker_counts >= min_speeches)]
    absent_speaker_inds = [ind for ind, x in enumerate(speaker)
                           if x in absent_speakers]
    present_speaker_inds = [ind for ind, x in enumerate(speaker)
                            if x in present_speakers]
    speaker = np.delete(speaker, absent_speaker_inds)
    speeches = np.delete(speeches, absent_speaker_inds)
    speech_ids = np.delete(speech_ids, absent_speaker_inds)
    party = np.delete(party, absent_speaker_inds)
    gender = np.delete(gender, absent_speaker_inds)
    state = np.delete(state, absent_speaker_inds)
    speaker_party = np.array([speaker[j] + " (" + party[i] + ")" for j in range(len(speaker))])

    # Create mapping between names and IDs.
    speaker_to_speaker_id = dict([(y.title(), x) for x, y in enumerate(sorted(set(speaker_party)))])
    author_indices = np.array([speaker_to_speaker_id[s.title()] for s in speaker_party])
    author_map = np.array(list(speaker_to_speaker_id.keys()))

    # author_D = tf.math.unsorted_segment_sum(tf.cast(party == 'D', tf.int32), author_indices, len(author_map))
    # author_R = tf.math.unsorted_segment_sum(tf.cast(party == 'R', tf.int32), author_indices, len(author_map))
    # author_I = tf.math.unsorted_segment_sum(tf.cast(party == 'I', tf.int32), author_indices, len(author_map))
    # author_party_counts = tf.convert_to_tensor([author_D, author_R, author_I])
    # tf.reduce_sum(tf.cast(author_party_counts > 0, tf.int32), axis=0) # in how many parties the author is
    # author_party_indices = tf.where(author_party_counts > 0)
    # party_label = ['D', 'R', 'I']
    # author_party = [party_label[author_party_indices[a, 0]] for a in tf.argsort(author_party_indices[:, 1])]
    #
    # author_M = tf.math.unsorted_segment_sum(tf.cast(gender == 'M', tf.int32), author_indices, len(author_map))
    # author_F = tf.math.unsorted_segment_sum(tf.cast(gender == 'F', tf.int32), author_indices, len(author_map))
    # author_gender_counts = tf.convert_to_tensor([author_M, author_F])
    # tf.reduce_sum(tf.cast(author_gender_counts > 0, tf.int32), axis=0)  # in how many parties the author is
    # author_gender_indices = tf.where(author_gender_counts > 0)
    # gender_label = ['M', 'F']
    # author_gender = [gender_label[author_gender_indices[a, 0]] for a in tf.argsort(author_gender_indices[:, 1])]

    author_party = [np.unique(party[author_indices == a])[0] for a in range(len(author_map))]
    author_gender = [np.unique(gender[author_indices == a])[0] for a in range(len(author_map))]
    author_state = [np.unique(state[author_indices == a])[0] for a in range(len(author_map))]
    author_infomatrix = [[np.unique(party[author_indices == a])[0],
                          np.unique(gender[author_indices == a])[0],
                          np.unique(state[author_indices == a])[0],
                          np.unique(speaker[author_indices == a])[0]] for a in range(len(author_map))]
    author_info = pd.DataFrame(author_infomatrix, columns=['party', 'gender', 'state', 'name'])

    count_vectorizer = CountVectorizer(min_df=min_df,
                                       max_df=max_df,
                                       stop_words=stop_words,
                                       ngram_range=ngram_range,
                                       token_pattern=token_pattern)

    # Learn initial document term matrix. This is only initial because we use it to
    # identify words to exclude based on author counts.
    counts = count_vectorizer.fit_transform(speeches.astype(str))
    vocabulary = np.array(
        [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(),
                                key=lambda kv: kv[1])])

    # Remove bigrams spoken by less than 10 Senators.
    counts_per_author = bincount_2d(author_indices, counts.toarray())
    author_counts_per_word = np.sum(counts_per_author > 0, axis=0)
    acceptable_words = np.where(author_counts_per_word >= min_authors_per_word)[0]

    # Fit final document-term matrix with modified vocabulary.
    count_vectorizer2 = CountVectorizer(min_df=min_df,
                                        max_df=max_df,
                                        stop_words=stop_words,
                                        ngram_range=ngram_range,
                                        token_pattern=token_pattern,
                                        vocabulary=vocabulary[acceptable_words])
    counts2 = count_vectorizer2.fit_transform(speeches.astype(str))
    # counts2.shape
    # np.sum(counts2, axis=0)
    vocabulary = np.array(
        [k for (k, v) in sorted(count_vectorizer2.vocabulary_.items(),
                                key=lambda kv: kv[1])])

    # counts_dense = remove_cooccurring_ngrams(counts, vocabulary) #not required since only bigrams are being considered
    # Remove speeches with no words.
    existing_speeches = np.where(np.sum(counts2, axis=1) > 0)[0]
    counts = counts2[existing_speeches]
    author_indices = author_indices[existing_speeches]
    speech_id_indices = speech_ids[existing_speeches]

    # np.sum(np.sum(counts, axis=0) == 0) # no word does not appear
    # np.sum(np.sum(counts, axis=0) <= 10) # no word does appear less than 10-times
    # np.sum(np.sum(counts, axis=1) == 0) # no empty documents
    # np.sum(np.sum(counts, axis=1) <= 1)  # there are documents of just one bigram!!

    # saving input matrices for STBS
    sparse.save_npz(os.path.join(save_dir, 'counts' + str(i) + '.npz'),
                    sparse.csr_matrix(counts).astype(np.float32))
    np.save(os.path.join(save_dir, 'author_indices' + str(i) + '.npy'), author_indices)
    np.savetxt(os.path.join(save_dir, 'author_map' + str(i) + '.txt'), author_map, fmt="%s")
    np.savetxt(os.path.join(save_dir, 'vocabulary' + str(i) + '.txt'), vocabulary, fmt="%s")
    np.save(os.path.join(save_dir, 'speech_id_indices' + str(i) + '.npy'), speech_id_indices)
    author_info.to_csv(os.path.join(save_dir, 'author_info' + str(i) + '.csv'))
    #np.save(os.path.join(save_dir, 'author_info' + str(i) + '.npy'), author_info)
    print('done for session ' + str(i))

