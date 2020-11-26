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
#
# data = pd.read_sql_query("select top(10000) client_id from co_aggregations where client_id not in "
#                          "(select client_id from co_verticals) ", #TODO add " and vertical_id is not null"
#                          sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
#
#
# qry_desc = """select client_id, [description], source, acq_date from
# (select client_id, [description], source, acq_date,row_number()
# over(partition by client_id, source order by acq_date desc) as rn
# from co_descriptions) as T
# where rn = 1"""
#
# qry_pitch = """select client_id, pitch_line, source, acq_date from
# (select client_id, pitch_line, source, acq_date,row_number()
# over(partition by client_id, source order by acq_date desc) as rn
# from co_pitch_lines) as T
# where rn = 1"""
#
# priority_pitch = ['cbinsights', 'owler','i3', 'dealroom','crunchbase','old_db','linkedin']
#
# priority_desc = ['cbinsights', 'crunchbase', 'dealroom', 'linkedin', 'old_db', 'wb_text_nlp', 'wb_html_position']
#
#
# desc = pd.read_sql_query(qry_desc, sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
# pitch = pd.read_sql_query(qry_pitch, sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
#
#
# def clean_text(text):
#         return re.sub(r'[;,\.!\?\(\)]', ' ', text.lower()).replace('\n', '').replace('[\s+]', ' ')



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
        count_dict = {}
        flag = 0
        countvec = CountVectorizer(ngram_range=(1, 4))
        countvec.fit_transform([line])
        words = countvec.get_feature_names()

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
    # for k in y:
    #     try:
    #         print(idtotag[k])
    #     except:
    #         pass
    return X, y

#company ids to be classified
# ids =data.client_id.values
# predictionssql = {}
# countforoutput = 0
# for index in ids:
#     #group all descritption of same company together
#     desc_client = desc[desc.client_id == index]
#     desc_client["source"] = pd.Categorical(desc_client["source"], categories=priority_desc, ordered=True)
#     desc_client = desc_client.sort_values('source')
#
#     pitch_client = pitch[pitch.client_id == index]
#     pitch_client["source"] = pd.Categorical(pitch_client["source"], categories=priority_pitch, ordered=True)
#     pitch_client = pitch_client.sort_values('source')
#
#     descrlist = desc_client.description.apply(lambda x: clean_text(x))
#     pitchlist = pitch_client.pitch_line.apply(lambda x: clean_text(x))

# whole_text = ' '.join(descrlist.values.tolist() + pitchlist.values.tolist())
#
# with open('PipelinePredictions/tag_data/tags_synonyms_dict.json') as fp:
#     tag_labels = json.load(fp)
#
# print(generate_pseudo_labelsTag([whole_text], tag_labels))
