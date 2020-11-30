import pandas as pd
import os
import sys
import re
import json
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz

#ignore warnings
warnings.filterwarnings("ignore")


# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt
from AzureStorage import blob_download

basepath = './PipelinePredictions/models/'

tagtoidmap = pd.read_sql_query("""SELECT [id],[tag],[edited_date] FROM [dbo].[tb_tags]""", sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
idtotag = dict(zip(tagtoidmap.id, tagtoidmap.tag.apply(lambda x : x.lower())))
tagtoid = dict((x,i) for i,x in idtotag.items())

with open('PipelinePredictions/tag_data/tags_mapping_dict.json') as fp:
    nested_tags = json.load(fp)

#function that uses a tokenizer and a dictionary of seedowrds to predict labels
def generate_pseudo_labelsTag(strings, labelsdict):

    #labels and dictionary of seedwords, not needed the sorting anymore
    labels = sorted(labelsdict.keys())

    #this an implementation for multilabel, returns a one-hot-encoded array
    def argmax_perfectmatch(count_dict, thresh=1):
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

        predlabels = []
        for lbl,count in labcounts:
            if count >= thresh:
                predlabels.append(tagtoid[lbl])

        return  predlabels

    y = []
    X = []

    for line in strings:
        count_dict= {}
        flag = 0
        try:
            countvec = CountVectorizer(ngram_range=(1, 4))
            countvec.fit_transform([line])
            words = countvec.get_feature_names()
        except:
            # if there are some errors in fit transform just skip the line
            continue

        for l in labels:
            seed_words = set()
            for w in labelsdict[l]:
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
            for lb in lbl:
                linked_tags = nested_tags[str(lb)]
                lbl = lbl + linked_tags
            #remove duplicates
            lbl = list(set(lbl))

            y.append(lbl)
            X.append(line)
        else:
            #fix to not allow empty result from pseudo label
            X.append(line)
            y.append([-1])

    y = [item for sublist in y for item in sublist]
    y = [tg for tg in y if tg in idtotag.keys()]
    return X, y
