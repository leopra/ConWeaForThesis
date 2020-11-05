import os
import re
import sys
import pickle

import nltk
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH,'Libraries-GP'))

# eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt
import CrunchbaseDatasetProcessing as cp

data = pd.read_sql_query("""select * from ml_crunchbase  where id in (select min(id) from ml_crunchbase group by cb_url)""",
    sql_cnnt("cnnt", DATABASE_CONFIG_NEW))


verticals = pd.read_sql_query("""select * from tb_verticals""",
    sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
# remove url duplicates
data = data.drop_duplicates(subset=["cb_url"])
# remove companies with empy description
data = data[data['description'] != '—']

# create column with eutopia categories as list for 1hot encoding
# CREATES COLUMN 'LISTLABEL'
data = cp.assign_eutop_labelsV2(data)


#create single label dataframe compatible with conwea
dfcon = data[['description', 'listlabel']]
dfcon = dfcon[dfcon['listlabel'].map(len) > 0]

dfcon['todrop'] = ['Telecommunications & ICT' in r for r in dfcon['listlabel']]
dfcon = dfcon[dfcon['todrop'] == False]

mlb = MultiLabelBinarizer()
categories_1hot = mlb.fit_transform(dfcon.listlabel)
categories_cols = mlb.classes_
dfcon['label'] = list(categories_1hot)
#this is the final dataset onehotencoded
dfcon = dfcon[['description','label']]
dfcon.columns= ['sentence','label']

out = dfcon.sample(frac=1).reset_index().drop('index', axis=1)
out.to_pickle('./data/eutopiavert/df.pkl', protocol=3)

#
# #see descriptions length
# def clean_and_tokenize(text):
#     text = re.sub('[;,:\!\?\.\(\)\n]', ' ', text).replace('[\s+]', ' ')
#     return nltk.word_tokenize(text)
#
#
# with open('./data/eutopiavert/df.pkl', 'rb') as handle:
#     b = pickle.load(handle)
#
# smaller = b.groupby('label', as_index=False).apply(lambda x: x.sample(20))
# smaller = smaller.reset_index().drop('level_0', axis=1).drop('level_1', axis=1).sample(frac=1).reset_index()
# smaller.to_pickle('./data/eutopiaverttest/df.pkl', protocol=3)