import pandas as pd
import os
import sys
from PipelinePredictions import Pipeline_predict as pipred
import fasttext
import numpy as np
import re
# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt

data = pd.read_sql_query("select client_id from co_aggregations where client_id not in "
                         "(select client_id from co_verticals)",
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


#model to classify description
modeldescr = fasttext.load_model('../PipelinePredictions/models/fasttex_model_23_10_2020.bin')

#model to classify pitchlines
modelpitch = fasttext.load_model('../PipelinePredictions/models/fasttex_model_23_10_2020.bin')

label_args = {'__label__Agriculture': 0,
 '__label__Buildings': 1, '__label__Constructions': 2, '__label__Energy': 3,
 '__label__Financial_services': 4, '__label__Food_&_Beverage': 5,
 '__label__Healthcare': 6, '__label__Logistics': 7, '__label__Manufacturing': 8,
 '__label__Mining': 9, '__label__Public_Administration': 10, '__label__Telecommunications_&_ICT': 10, #TODO remove this
 '__label__Transportation': 11, '__label__Utilities_(electricity,_water,_waste)': 12}

def convertLabeltoOneHot(fastpred):
    out = []
    for i in fastpred[0]:
        x = [label_args[z] for z in i]
        onehot = np.zeros(len(label_args)-1, dtype=int)
        for j in x:
            onehot[j] = 1
        out.append(onehot)
    return out

def clean_text(text):
        return re.sub(r'[;,\.!\?\(\)]', ' ', text.lower()).replace('\n', '').replace('[\s+]', ' ')
#company ids to be classified
ids =data.client_id.values

for index in ids[:1]:
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
    descrlist = desc_client.description.apply(lambda x: clean_text(x)).values
    pitchlist = pitch_client.pitch_line.apply(lambda x: clean_text(x)).values

    #pseudo predictions
    text, preds_desc = pipred.generate_pseudo_labels(descrlist)
    text, preds_pitch = pipred.generate_pseudo_labels(pitchlist)

    #fasttext predictions
    predfastdescr = modeldescr.predict(descrlist.tolist(), k=3)
    predfastpitch = modelpitch.predict(pitchlist.tolist(), k=3)
    onehotpreddescr = convertLabeltoOneHot(predfastdescr)
    onehotpredpitch = convertLabeltoOneHot(predfastpitch)

    res = np.concatenate((preds_desc, onehotpreddescr, preds_pitch, onehotpredpitch), axis=0)
    results = np.stack(res, axis=0)
    print(np.sum(results, axis=0))
    print(results)