import pickle
import json
import numpy as np
import pandas as pd




def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    labels = sorted(labels)
    #this an implementation for multilabel, returns a one-hot-encoded array
    # V1
    def argmax_perfectmatch(count_dict, percentage=0.5):
        total = 0
        labcounts = []
        labfound=0
        for l in labels:
            count = 0
            try:
                for t in count_dict[l]:
                    count += count_dict[l][t]
                labfound += 1

            except:
                pass
            labcounts.append((l, count))
            total += count

        MAX_COUNT = max(labcounts,key=lambda item:item[1])[1]

        current = np.zeros(len(labels), dtype=int)

        # if i have only match of less than 2 classes assign all of them
        if labfound < 2:
            for i in range(len(current)):
                if labcounts[i][1] != 0:
                    current[i] = 1

        #TODO what if (man, 1)(health,1)(logistic,1)(agriculture,1)
        else:
            labcountsvalue = [x[1] for x in labcounts]
            orderbycount = sorted(range(len(labcounts)), key=labcountsvalue.__getitem__, reverse=True)
            #get only 3 values not more
            for i in orderbycount[:3]:
                # check for threshold from the max
                if (labcounts[i][1] / MAX_COUNT) > percentage:
                    current[i] = 1

        return current

    # V2
    def argmax_perfectV2(count_dict, percentage=0.1):
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

        current = np.zeros(len(labels), dtype=int)

        labcounts = [x[1] for x in labcounts]
        index_max = max(range(len(labcounts)), key=labcounts.__getitem__)
        current[index_max] = 1
        MAX_VALUE = labcounts[index_max][1]

        #TODO finish


        # add 1 to labels over the threshold
        for i in range(len(current)):
            # if i have only match of less than 3 classes assign all of them
            if len(labcounts) < 3:
                if labcounts[i][1] != 0:
                    current[i] = 1.0

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
            if np.sum(lbl) == 0:
                continue
            y.append(lbl)
            X.append(line)
            y_true.append(label)
    return X, y, y_true

dataset_path = './data/eutopiavert/'

df = pickle.load(open(dataset_path + "df_contextualized.pkl", "rb"))

tokenizer = pickle.load(open(dataset_path + "tokenizer.pkl", "rb"))

with open("debugdata/seedwords.json") as fp:
    label_term_dict = json.load(fp)

labels = list(set(label_term_dict.keys()))

generate_pseudo_labels(df[500:], labels, label_term_dict, tokenizer)