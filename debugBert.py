import pickle
from flair.embeddings import BertEmbeddings
from nltk import sent_tokenize
# from flair.data import Sentence
# from pytorch_pretrained_bert import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# embedding = BertEmbeddings('bert-base-uncased')
import pickle
import json
import numpy as np
import pandas as pd

# with open('./data/eutopiavertzz/df.pkl', 'rb') as handle:
#     df = pickle.load(handle)
#
# len(df.iloc[14].sentence)
#
# for index, row in df[14:15].iterrows():
#     if index % 100 == 0:
#         print("Finished sentences: " + str(index) + " out of " + str(len(df)))
#     # all sentences are undercase now
#     line = row["sentence"].lower()
#     sentences = sent_tokenize(line)
#     for sentence_ind, sent in enumerate(sentences):
#         tokenized_text = tokenizer.tokenize(sent)
#         if len(tokenized_text) > 512:
#             print('sentence too long for Bert: truncating')
#             sentence = Sentence(' '.join(sent[:512]), use_tokenizer=True)
#         else:
#             sentence = Sentence(sent, use_tokenizer=True)
#         try:
#             embedding.embed(sentence)
#         except Exception as e:
#             print("Exception Counter while getting BERT: ", sentence_ind, index, e)
#             print(sentence)
#             print(index)
#             continue

#CODE USED TO SAVE FILES TO AZURE BLOB STORAGE
#
# import os
# import sys
# PATH = os.path.dirname(os.__file__)
# sys.path.append(os.path.join(PATH, 'Libraries-GP'))
#
# from AzureStorage import blob_upload
# blob_upload('verticals-ml', 'cluster-map1,8gb', './data/eutopiavert/word_cluster_map.pkl')

dataset_path = './data/eutopiavert/'
df = pickle.load(open(dataset_path + "df_contextualized.pkl", "rb"))
tokenizer = pickle.load(open(dataset_path + "tokenizer.pkl", "rb"))
with open(dataset_path + "seedwordsencoded.json") as fp:
    label_term_dict = json.load(fp)

labels = list(set(label_term_dict.keys()))

def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    #TODO this is bad code, i must be sure that labels follows the one-hot-index order
    labels = sorted(labels)
    #this an implementation for multilabel, returns a one-hot-encoded array
    def argmax_perfectmatch(count_dict, percentage=0.05):
        print(count_dict)
        total = 0
        labcounts = {}
        for l in labels:
            count = 0
            try:
                for t in count_dict[l]:
                    count += count_dict[l][t]
            except:
                pass
            if count != 0:
                labcounts[l] = count
            total += count

        current = np.zeros(len(labels), dtype = int)

        # add 1 to labels over the threshold
        for i,lbl in enumerate(labels):
            # if i have only match of less than 3 classes assign all of them
            try:
                if labcounts[lbl] > percentage:
                    current[i] = 1
            except:
                pass
        return current


    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        line = row["sentence"]
        label = row["label"]
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
            #TODO currently is impossible that there is no label, in the future maybe this should be possible
            if np.sum(lbl) == 0:
                continue
            y.append(lbl)
            X.append(line)
            y_true.append(label)
            lbl = argmax_perfectmatch(count_dict)
    return X, y, y_true


dfs = df
X, y, y_true = generate_pseudo_labels(dfs, labels, label_term_dict, tokenizer)