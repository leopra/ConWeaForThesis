import pandas as pd
import os
import sys
from PipelinePredictions import Pipeline_predict as pipred
from PipelinePredictions import SubBigrams as subbi
import fasttext
import numpy as np
import re
import json
import warnings

#ignore warnings
warnings.filterwarnings("ignore")


# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt
from AzureStorage import blob_download

basepath = './PipelinePredictions/models/'

data = pd.read_sql_query("select top(10000) client_id from co_aggregations where client_id not in "
                         "(select client_id from co_verticals) ", #TODO add " and vertical_id is not null"
                         sql_cnnt("cnnt", DATABASE_CONFIG_NEW))


qry_desc = """select client_id, [description], source, acq_date from
(select client_id, [description], source, acq_date,row_number()
over(partition by client_id, source order by acq_date desc) as rn
from co_descriptions) as T
where rn = 1"""

qry_pitch = """select client_id, pitch_line, source, acq_date from 
(select client_id, pitch_line, source, acq_date,row_number() 
over(partition by client_id, source order by acq_date desc) as rn
from co_pitch_lines) as T
where rn = 1"""

priority_pitch = ['cbinsights', 'owler','i3', 'dealroom','crunchbase','old_db','linkedin']

priority_desc = ['cbinsights', 'crunchbase', 'dealroom', 'linkedin', 'old_db', 'wb_text_nlp', 'wb_html_position']


desc = pd.read_sql_query(qry_desc, sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
pitch = pd.read_sql_query(qry_pitch, sql_cnnt("cnnt", DATABASE_CONFIG_NEW))

def encodeTagBigrams(bigrams):
    subbi.substituteBigwithMono()



def clean_text(text):
        return re.sub(r'[;,\.!\?\(\)]', ' ', text.lower()).replace('\n', '').replace('[\s+]', ' ')



tagtoidmap = pd.read_sql_query("""SELECT [id],[tag],[edited_date] FROM [dbo].[tb_tags]""", sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
idtotag = dict(zip(tagtoidmap.id, tagtoidmap.tag))
tagtoid = dict((x,i) for i,x in idtotag.items())

with open('PipelinePredictions/tag_data/tags_mapping_dict.json') as fp:
    nested_tags = json.load(fp)

#function that uses a tokenizer and a dictionary of seedowrds to predict labels
def generate_pseudo_labelsTag(strings, labelsdict):

    strings = ['hydroceramics self-healing materials carbon fiber nano fiber bioconcrete innovative materials biobased materials']
    with open('PipelinePredictions/tag_data/tags_synonyms_dict.json') as fp:
        labelsdict = json.load(fp)
    #labels and dictionary of seedwords
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
                predlabels.append(lbl)

        return  predlabels

    y = []
    X = []

    for line in strings:
        tokens = line.split(' ')
        words = tokens
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            linked_tags = [idtotag[l] for l in nested_tags[str(tagtoid[l])]]
            for link in linked_tags:
                for w in labelsdict[link]:
                    seed_words.add(w)
                #add also the tag name to search, if double words split on token /
                seed_words.append(link)
                k = [x.strip() for x in link.split('/')]
                for jj in k:
                    seed_words.append(jj)
            #TODO add tfidfmatcher
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
        else:
            #fix to not allow empty result from pseudo label
            X.append(line)
            y.append([-1])
    return X, y

#company ids to be classified
ids =data.client_id.values
predictionssql = {}
countforoutput = 0
for index in ids[:50]:
    #group all descritption of same company together
    desc_client = desc[desc.client_id == index]
    desc_client["source"] = pd.Categorical(desc_client["source"], categories=priority_desc, ordered=True)
    desc_client = desc_client.sort_values('source')

    pitch_client = pitch[pitch.client_id == index]
    pitch_client["source"] = pd.Categorical(pitch_client["source"], categories=priority_pitch, ordered=True)
    pitch_client = pitch_client.sort_values('source')

    descrlist = desc_client.description.apply(lambda x: clean_text(x))
    pitchlist = pitch_client.pitch_line.apply(lambda x: clean_text(x))


    with open('PipelinePredictions/tag_data/tags_synonyms_dict.json') as fp:
        tag_labels = json.load(fp)

    bigrams = []
    # mapping from bigrams to fake monograms
    bigtomon = {}

    newdictseed = {}
    for vert in tag_labels.keys():
        newterms = []
        for i, seed in enumerate(tag_labels[vert]):
            x = seed.split(' ')
            if len(x) == 2:
                bigrams.append(seed)
                bigtomon[seed] = ''.join(x)
                newterms.append(''.join(x))
            else:
                newterms.append(seed)
        newdictseed[vert] = newterms

        bigramset = set(bigrams)

        #TODO handle trigrams
        #TODO handle bat match |i need [oil fuel]ing| when matching oil fuel
    descrlist = descrlist.apply(lambda x: subbi.substituteBigwithMono(x, bigramset))
    pitchlist = pitchlist.apply(lambda x: subbi.substituteBigwithMono(x, bigramset))

    print(generate_pseudo_labelsTag(descrlist, newdictseed))