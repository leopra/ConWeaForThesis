import pickle
import argparse
import json
import gc
import math
from util import *
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from keras.callbacks import ModelCheckpoint
from collections import defaultdict
from gensim.models import word2vec
from keras_han.model import HAN
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import string
import os
import numpy as np
from keras.models import load_model
from gensim.models import word2vec
from keras_han.layers import AttentionLayer
from keras_han.model import HAN
from nltk import tokenize
import pandas as pd
from keras.metrics import TopKCategoricalAccuracy
import matplotlib.pyplot as plt

#path where all the pretrained models are saved
basepath = './PipelinePredictions/models/'
#tokenizer
tokenizer = pickle.load(open(basepath+ "tokenizer.pkl", "rb"))

#new data needs to be processed with bert to disambiguate word meanings
#encoded with bert then checks for the nearest cluster
#also is undercased and classic preprocesses
def preprocess(strings):
    #this data structure is a dictionary that saves thw words and the coentroid vector for each meaning found by kmeans
    # bank$0 corresponds to the first element
    word_cluster = pickle.load(open(basepath + "word_cluster_map.pkl", "rb"))
    def get_vec(word, word_cluster, stop_words):
        if word in stop_words:
            return []
        t = word.split("$")
        if len(t) == 1:
            prefix = t[0]
            cluster = 0
        elif len(t) == 2:
            prefix = t[0]
            cluster = t[1]
            try:
                cluster = int(cluster)
            except:
                prefix = word
                cluster = 0
        else:
            prefix = "".join(t[:-1])
            cluster = t[-1]
            try:
                cluster = int(cluster)
            except:
                cluster = 0

        word_clean = prefix.translate(str.maketrans('', '', string.punctuation))
        if len(word_clean) == 0 or word_clean in stop_words:
            return []
        try:
            vec = word_cluster[word_clean][cluster]
        except:
            try:
                vec = word_cluster[prefix][cluster]
            except:
                try:
                    vec = word_cluster[word][0]
                except:
                    vec = []
        return vec

    print("Preprocessing data..")
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    word_vec = {}
    for index,line in enumerate(strings):
        if index % 100 == 0:
            print("Finished rows: " + str(index) + " out of " + str(len(strings)))

        words = line.strip().split()
        new_words = []
        for word in words:
            try:
                vec = word_vec[word]
            except:
                vec = get_vec(word, word_cluster, stop_words)
                if len(vec) == 0:
                    continue
                word_vec[word] = vec
            new_words.append(word)
        strings[index] = " ".join(new_words)

    #i guess it's to free memory
    del word_cluster
    gc.collect()
    return strings, word_vec


#function that uses a tokenizer and a dictionary of seedowrds to predict labels
def generate_pseudo_labels(strings, tokenizer):

    #labels and dictionary of seedwords
    with open(basepath + "seedwordsencoded.json") as fp:
        label_term_dict = json.load(fp)
    labels = sorted(label_term_dict.keys())

    #this an implementation for multilabel, returns a one-hot-encoded array
    def argmax_perfectmatch(count_dict, percentage=0.2):
        total = 0
        labcounts = []
        for l in labels:
            count = 0
            try:
                for t in count_dict[l]:
                    count += count_dict[l][t]
            except:
                pass
            labcounts.append((l, count))
            total += count

        current = np.zeros(len(labels))

        # add 1 to labels over the threshold
        for i in range(len(current)):
            # if i have only match of less than 3 classes assign all of them
            if len(labcounts) < 3:
                if labcounts[i][1] != 0:
                    current[i] = 1.0

            # if they are more check for threshold
            else:
                if (labcounts[i][1] / total) >= percentage:
                    current[i] = 1.0

        # if there was no label over the threshold give the best one
        if np.sum(current) == 0:
            labcounts = [x[1] for x in labcounts]
            index_max = max(range(len(labcounts)), key=labcounts.__getitem__)
            current[index_max] = 1.0

        return current

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w

    for line in strings:
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
        if flag:
            lbl = argmax_perfectmatch(count_dict)
            #currently is impossible that there is no label, in the future maybe this should be possible
            if np.sum(lbl) == 0:
                continue
            y.append(lbl)
            X.append(line)
    return X, y

#this function encodes strings to integer and pads/trim using the tokenizer pretrained in /models
def prep_data_for_HAN(max_sentence_length, max_sentences, texts, tokenizer):
    data = np.zeros((len(texts), max_sentences, max_sentence_length), dtype='int32')
    documents = []
    for text in texts:
        # added lowercase
        sents = tokenize.sent_tokenize(text.lower())
        documents.append(sents)
    for i, sentences in enumerate(documents):
        tokenized_sentences = tokenizer.texts_to_sequences(
            sentences
        )
        tokenized_sentences = pad_sequences(
            tokenized_sentences, maxlen=max_sentence_length
        )

        pad_size = max_sentences - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:max_sentences]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )

        data[i] = tokenized_sentences[None, ...]
    return data

#PREDICT WITH HAN NETWORK
#returns one-hot-encoding
def predictWithHAN(strings, word_vec):
    max_sentence_length = 100
    #TODO this value could be less, study necessary
    max_sentences = 15
    tokenizer = pickle.load(open(basepath + "tokenizer.pkl", "rb"))

    strings = df.sentence.values
    strings = prep_data_for_HAN(texts=strings, max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)

    # modelvec = word2vec.Word2Vec(
    #     strings, size=300, window=5, min_count=5, workers=5,
    #     sg=1, hs=1, negative=0
    # )
    # modelvec.save_word2vec_format(basepath + 'word2vec.bin', binary=True)

    # load model
    model = load_model(basepath + 'model_conwea.h5',
                       custom_objects={'AttentionLayer': AttentionLayer, 'HAN': HAN})

    model.load_weights(basepath + 'model_weights_conwea.h5')
    #TODO load weigths
    pred = model.predict(strings)
    pred = (pred > 0.5).astype(int)
    return pred

#DEBUG
df = pickle.load(open(basepath + "df_contextualized.pkl", "rb"))
strings = df.sentence.values
x, y =preprocess(strings)
for t in df.sentence.values:
#t = df.iloc[].sentence
    print(generate_pseudo_labels(t))

#
# label_term_dict = add_all_interpretations(label_term_dict, word_cluster)
# print_label_term_dict(label_term_dict, None, print_components=False)
# labels = list(set(label_term_dict.keys()))