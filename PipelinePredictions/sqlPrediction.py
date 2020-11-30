import pandas as pd
import os
import sys
from PipelinePredictions import Pipeline_predict as pipred
from PipelinePredictions import ExternalTagsIntegration as External
import fasttext
import numpy as np
import re
import json
import warnings
from PipelinePredictions import bertSupervised
from fuzzywuzzy import fuzz
from itertools import combinations
from PipelinePredictions import TagPredictions
from PipelinePredictions import VertPicker
import datetime
from SQLServer import insert_into_sql_auto_incr

with open('PipelinePredictions/tag_data/tags_synonyms_dict.json') as fp:
    tag_labels = json.load(fp)

basepath = './PipelinePredictions/models/'

#ignore warnings
warnings.filterwarnings("ignore")


# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt
from AzureStorage import blob_download, blob_upload

data = pd.read_sql_query("select client_id from co_aggregations where client_id not in "
                         "(select client_id from co_verticals) and is_platform = 1", #TODO add " and vertical_id is not null"
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


#model to classify description and pitch , download if needed
#blob_download('verticals-ml', 'fasttext-descr', basepath + 'fasttext-descr.bin') #86% top1 accuracy
#blob_download('verticals-ml', 'fasttext-pitch', basepath +'fasttext-pitch.bin') #62% top1 accuracy
#blob_download('verticals-ml', 'bert-superv-descr', 'PipelinePredictions/models/bert_weigths.pt')


modeldescr = fasttext.load_model(basepath +'fasttex_pitch_model_ova_18_11_2020.bin')

#model to classify pitchlines
#modelpitch = fasttext.load_model(basepath +'fasttext-pitch.bin')

label_args = {'__label__Agriculture': 0,
 '__label__Buildings': 1, '__label__Constructions': 2, '__label__Energy': 3,
 '__label__Financial_services': 4, '__label__Food_&_Beverage': 5,
 '__label__Healthcare': 6, '__label__Logistics': 7, '__label__Manufacturing': 8,
 '__label__Mining': 9, '__label__Public_Administration': 10,
 '__label__Transportation': 11, '__label__Utilities_(electricity,_water,_waste)': 12}

inverse_dict = dict([(i,f) for f,i in label_args.items()])

real_vertical_index = pd.read_sql_query("SELECT * FROM [tb_verticals]",sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
mapfastindextoeutopiaindex = {0:1,1:2,2:3,3:4,4:5,5:6,6:8,7:9,8:10,9:11,10:13,11:15,12:16}
invertmapfasteutopia = dict([(i,x) for x,i in mapfastindextoeutopiaindex.items()])
#[(1, 'agriculture'), (2, 'building'), (3, 'constructions'), (4, 'energy'), (5, 'financial services'),
# (6, 'food & beverage'), (7, 'forestry'), (8, 'healthcare'), (9, 'logistics'), (10, 'manufacturing'),
# (11, 'mining'), (12, 'other activities'), (13, 'public administration'), (14, 'telecommunications & ICT'),
# (15, 'transportation'), (16, 'utilities (electricity, water, waste)')]

#get external verticals and tags
externaldb = External.getExternals()
externaldb.set_index('client_id').to_dict('series')
diziovert = dict(zip(externaldb.client_id, externaldb.vert))
diziotag = dict(zip(externaldb.client_id, externaldb.tag))

#this function convert the information from the database external to ertical one hot encoding in "fasttext format"
#importanceofexternal is an integer saying the weight of the classification
def predFromExternal(diziovert, client_id, inversemap, importanceofexternal):
    out = np.zeros(len(invertmapfasteutopia), dtype=int)
    fastlabel = []
    if client_id in diziovert.keys():
        labels = diziovert[client_id]
        for lbl in labels:
            try:
                x = inversemap[lbl]
                fastlabel.append(x)
            except:
                pass
    for i in fastlabel:
        out[i] = importanceofexternal
    return out



#this function taker the fasttext prediction and returns a one hot encoding checking if the values is over 0.5
#the object returned by fasttext prediction is a tuple of 2 arrays, the first one containing the arrays with predictions for each label, the second
#containing the confidence value for each label
def convertLabeltoOneHot(fastpred):
    out = []
    for k,i in enumerate(fastpred[0]):
        values = fastpred[1][k]
        x = [label_args[z] for k,z in enumerate(i) if values[k]>0.5]
        onehot = np.zeros(len(label_args), dtype=int)
        for j in x:
            onehot[j] = 1
        out.append(onehot)
    return out

def clean_text(text):
        return re.sub(r'[;,\.!\?\(\)]', ' ', text.lower()).replace('\n', '').replace('[\s+]', ' ')


with open(basepath + 'seedwords.json') as fp:
    seedwords = json.load(fp)

#company ids to be classified
ids =data.client_id.values
predictionssql = {}
countforoutput = 0
for index in ids:
    modelcount = 0
    #group all descritption of same company together
    desc_client = desc[desc.client_id == index]
    desc_client["source"] = pd.Categorical(desc_client["source"], categories=priority_desc, ordered=True)
    desc_client = desc_client.sort_values('source')

    pitch_client = pitch[pitch.client_id == index]
    pitch_client["source"] = pd.Categorical(pitch_client["source"], categories=priority_pitch, ordered=True)
    pitch_client = pitch_client.sort_values('source')
    descrlist = desc_client.description.apply(lambda x: clean_text(x))
    pitchlist = pitch_client.pitch_line.apply(lambda x: clean_text(x))



    if len(descrlist) > 0:
        #classify descr if not empty
        descrlistx = descrlist.values.tolist()

        #this code is to delete similar descriptions
        comb = combinations(range(len(descrlistx)),2)
        todrop = []
        tosave = []
        for c0,c1 in comb:
            if c0 in todrop and c1 in todrop:
                continue
            if fuzz.ratio(descrlistx[c0], descrlistx[c1]) > 80:
                if c0 in tosave and c1 in tosave:
                    todrop.append(c1)
                    tosave.remove(c1)
                elif c0 in tosave:
                    todrop.append(c1)
                elif c1 in tosave:
                    todrop.append(c0)
                else:
                    tosave.append(c0)
                    todrop.append(c1)

        descrlistx = [i for j, i in enumerate(descrlistx) if j not in todrop]

        #pseudo
        text, preds_desc_pseudo = pipred.generate_pseudo_labels(descrlistx, seedwords)
        #fasttext
        predfastdescr = modeldescr.predict(descrlistx, k=4)
        onehotfastdescr = convertLabeltoOneHot(predfastdescr)
        #bert supervised
        bertpred = bertSupervised.predictBert(descrlistx)



    else:
        onehotfastdescr = [np.zeros(len(label_args), dtype=int)]
        preds_desc_pseudo = [np.zeros(len(label_args), dtype=int)]
        bertpred = [np.zeros(len(label_args), dtype=int)]

    if len(pitchlist) > 0:
        #classify pitch if not empty
        pitchlistx = pitchlist.values.tolist()
        text, preds_pitch_pseudo = pipred.generate_pseudo_labels(pitchlistx, seedwords)
        #TODO removed pitch to lower noise
        #predfastpitch = modelpitch.predict(pitchlistx, k=4)
        #onehotfastpitch = convertLabeltoOneHot(predfastpitch)
        onehotfastpitch = [np.zeros(len(label_args), dtype=int)]

    else:
        preds_pitch_pseudo = [np.zeros(len(label_args), dtype=int)]
        onehotfastpitch = [np.zeros(len(label_args), dtype=int)]

    # externals, array of zeros if no prediction
    predFromext = [predFromExternal(diziovert, index, invertmapfasteutopia, 5)]

    #code for nicer output
    res = np.concatenate((preds_desc_pseudo, onehotfastdescr, bertpred, preds_pitch_pseudo, onehotfastpitch, predFromext), axis=0)
    df = pd.DataFrame(res, dtype=int, columns = [x[9:14] for x in label_args])
    #codice fogna per avere indicato da dove viene la predizione nella tabella
    x = ['pseudo_descr'] * len(preds_desc_pseudo) + ['fast_descr'] * len(onehotfastdescr)\
        + ["bert"] * len(bertpred) + ['pseudo_pitch']* len(preds_pitch_pseudo) \
        + ['fast_pitch'] * len(onehotfastpitch) + ['ext']
    df['predtype'] = x

    #print(df)
    columnstosum = list(df)
    #remove label from sum
    columnstosum.pop()
    finalcounts = df[columnstosum].sum(axis=0).values
    print(finalcounts)

    #get verticals over threshold
    modelout = VertPicker.vertPicker(finalcounts)

    kk = [mapfastindextoeutopiaindex[ix] for ix in modelout]

    #get tags
    whole_text = ' '.join(descrlist.values.tolist() + pitchlist.values.tolist())
    pseudotag = TagPredictions.generate_pseudo_labelsTag([whole_text], tag_labels)[1]
    try:
        extertag = diziotag[index]
    except:
        extertag = []
    tags = list(set(pseudotag + extertag))



    timestamp = str(datetime.datetime.now())[:23]
    clientid = index
    verticals = kk if len(kk)>0 else [None]
    tags = tags if len(tags)>0 else [None]

    DATABASE_CONFIG_NEW = {'server': 'eutop.database.windows.net','database': 'eutop_companies','username': 'eutop-user','password': 'Topaltop93#'}

    tagsql = pd.DataFrame(data=[[clientid, x, timestamp] for x in tags], columns=["client_id", "tag_id", "acq_date"])
    print(tagsql)

   # insert_into_sql_auto_incr(conn_str=DATABASE_CONFIG_NEW, table_name="co_tags", db=tagsql, ex_man=True)

    vertsql = pd.DataFrame(data=[[clientid, x, timestamp] for x in verticals], columns=["client_id", "vertical_id", "acq_date"])
    print(vertsql)

    #insert_into_sql_auto_incr(conn_str=DATABASE_CONFIG_NEW, table_name="co_verticals", db=vertsql, ex_man=True)

    countforoutput += 1
    if countforoutput %100 ==0:
        print('classificate {} aziende'.format(countforoutput))



