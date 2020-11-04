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


#CREATE 1HOT encode for categories
mlb = MultiLabelBinarizer()
categories_1hot = mlb.fit_transform(data.listlabel)
categories_cols = mlb.classes_

#this is the final dataset onehotencoded
dataready = pd.concat([data[['id', 'description','listlabel']],pd.DataFrame(categories_1hot, columns=categories_cols, index=data.index)],axis=1)

#see descriptions length
def clean_and_tokenize(text):
    text = re.sub('[;,:\!\?\.\(\)\n]', ' ', text).replace('[\s+]', ' ')
    return nltk.word_tokenize(text)

#create single label dataframe compatible with conwea
datasinglelabel = dataready[dataready['listlabel'].map(len) == 1]
dataforconwea = datasinglelabel[['description', 'listlabel']]
dataforconwea['listlabel'] = dataforconwea['listlabel'].apply(lambda x: x[0])
dataforconwea.columns= ['sentence','label']
k = dataforconwea[dataforconwea['label'] == 'Telecommunications & ICT'].sample(6000)
z = dataforconwea[dataforconwea['label'] != 'Telecommunications & ICT']

out = pd.concat([k,z])
out = out.sample(frac=1).reset_index().drop('index', axis=1)
out.to_pickle('./data/eutopiavert/df.pkl', protocol=3)


with open('./data/eutopiavert/df.pkl', 'rb') as handle:
    b = pickle.load(handle)

smaller = b.groupby('label', as_index=False).apply(lambda x: x.sample(20))
smaller = smaller.reset_index().drop('level_0', axis=1).drop('level_1', axis=1).sample(frac=1).reset_index()
smaller.to_pickle('./data/eutopiaverttest/df.pkl', protocol=3)