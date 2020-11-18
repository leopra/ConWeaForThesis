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


#model to classify description and pitch , download if needed
#blob_download('verticals-ml', 'fasttext-descr', basepath + 'fasttext-descr.bin') #86% top1 accuracy
#blob_download('verticals-ml', 'fasttext-pitch', basepath +'fasttext-pitch.bin') #62% top1 accuracy

modeldescr = fasttext.load_model(basepath +'fasttext-descr.bin')

#model to classify pitchlines
modelpitch = fasttext.load_model(basepath +'fasttext-pitch.bin')

label_args = {'__label__Agriculture': 0,
 '__label__Buildings': 1, '__label__Constructions': 2, '__label__Energy': 3,
 '__label__Financial_services': 4, '__label__Food_&_Beverage': 5,
 '__label__Healthcare': 6, '__label__Logistics': 7, '__label__Manufacturing': 8,
 '__label__Mining': 9, '__label__Public_Administration': 10,
 '__label__Transportation': 11, '__label__Utilities_(electricity,_water,_waste)': 12}

real_vertical_index = pd.read_sql_query("SELECT * FROM [tb_verticals]",sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
mapfastindextoeutopiaindex = {0:1,1:2,2:3,3:4,4:5,5:6,6:8,7:9,8:10,9:11,10:13,11:15,12:16}
#[(1, 'agriculture'), (2, 'building'), (3, 'constructions'), (4, 'energy'), (5, 'financial services'),
# (6, 'food & beverage'), (7, 'forestry'), (8, 'healthcare'), (9, 'logistics'), (10, 'manufacturing'),
# (11, 'mining'), (12, 'other activities'), (13, 'public administration'), (14, 'telecommunications & ICT'),
# (15, 'transportation'), (16, 'utilities (electricity, water, waste)')]

#TODO check if this is ok, should be
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


#company ids to be classified
ids =data.client_id.values
predictionssql = {}
countforoutput = 0
for index in ids[:220]:
    #group all descritption of same company together
    desc_client = desc[desc.client_id == index]
    desc_client["source"] = pd.Categorical(desc_client["source"], categories=priority_desc, ordered=True)
    desc_client = desc_client.sort_values('source')

    pitch_client = pitch[pitch.client_id == index]
    pitch_client["source"] = pd.Categorical(pitch_client["source"], categories=priority_pitch, ordered=True)
    pitch_client = pitch_client.sort_values('source')
    #TODO handle bigrams
    #TODO handle empty data in the clearest way
    #TODO remove duplicates description
    descrlist = desc_client.description.apply(lambda x: clean_text(x))
    pitchlist = pitch_client.pitch_line.apply(lambda x: clean_text(x))

    #pseudo predictions, encode bigrams
    with open(basepath + 'seedwordsencoded.json') as fp:
        seedenc = json.load(fp)

    if len(descrlist) > 0:
        #classify descriptions if not empty
        try:
            descrlistbig = descrlist.apply(lambda x: subbi.substituteBigwithMono(x, seedenc)).values
        except:
            print(descrlist.values)
        text, preds_desc_pseudo = pipred.generate_pseudo_labels(descrlistbig)
        predfastdescr = modeldescr.predict(descrlist.values.tolist(), k=4)
        onehotfastdescr = convertLabeltoOneHot(predfastdescr)

    else:
        onehotfastdescr = [np.zeros(len(label_args), dtype=int)]
        preds_desc_pseudo = [np.zeros(len(label_args), dtype=int)]

    if len(pitchlist) > 0:
        #classify pitch if not empty
        try:
            pitchlistbig = pitchlist.apply(lambda x: subbi.substituteBigwithMono(x, seedenc)).values
        except:
            print(pitchlist.values)

        text, preds_pitch_pseudo = pipred.generate_pseudo_labels(pitchlistbig)
        predfastpitch = modelpitch.predict(pitchlist.values.tolist(), k=4)
        onehotfastpitch = convertLabeltoOneHot(predfastpitch)

    else:
        preds_pitch_pseudo = [np.zeros(len(label_args), dtype=int)]
        onehotfastpitch = [np.zeros(len(label_args), dtype=int)]


    #code for nicer output
    res = np.concatenate((preds_desc_pseudo, onehotfastdescr, preds_pitch_pseudo, onehotfastpitch), axis=0)
    df = pd.DataFrame(res, dtype=int, columns = [x[9:14] for x in label_args])
    #codice fogna per avere indicato da dove viene la predizione nella tabella
    x = ['pseudo_descr'] * len(preds_desc_pseudo) + ['fast_descr'] * len(onehotfastdescr) + ['pseudo_pitch']* len(preds_pitch_pseudo) + ['fast_pitch'] * len(onehotfastpitch)
    df['predtype'] = x

    #print(df)
    columnstosum = list(df)
    #remove label from sum
    columnstosum.pop()
    finalcounts = df[columnstosum].sum(axis=0).values
    print(finalcounts)
    #norm1 = finalcounts / np.linalg.norm(finalcounts)
    mean = np.mean(finalcounts)
    std = np.std(finalcounts)
    labelsovermean = (finalcounts > (mean +std)).astype(int)
    kk = []
    for ix,value in enumerate(labelsovermean):
        if value == 1:
            kk.append(mapfastindextoeutopiaindex[ix])

    predictionssql[index] = kk
    countforoutput += 1
    if countforoutput %20 ==0:
        print('classificate {} aziende'.format(countforoutput))


