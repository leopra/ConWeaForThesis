import sys
import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from util import *
import numpy

# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt
from AzureStorage import blob_upload, blob_download

#basepath = os.path.join(os.getcwd(), "models")

#blob_download('verticals-ml', 'dataset240k.pkl', 'dataforbert.pkl')

real_vertical_index = pd.read_sql_query("SELECT * FROM [tb_verticals]",sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
real_vertical_index = dict(list(zip(real_vertical_index.id, real_vertical_index.vertical)))

columns = [x for x in list(real_vertical_index.keys()) if x not in [12,14,7]]

# dataa = pickle.load(open('data/eutopiavert/df_contextualized.pkl', 'rb'))
#
# h = np.stack(dataa['label'].values)
# sum = np.sum(h, axis=0)

data = pickle.load(open('dataforbert.pkl', 'rb'))
#tokenizer = Tokenizer(num_words=150000, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
#tokenizer.fit_on_texts(data.about.values)

data['single'] = data.listlabel.apply(lambda x: 1 if np.sum(x) == 1 else 0)

data = data[data.single == 1]

#40 k elements have multiple classes

news = pickle.load(open('data/20news/coarse/df.pkl','rb'))

NUMGRAMS=2
words = data.iloc[0].about.strip().split()
countvec = CountVectorizer(ngram_range=(2, NUMGRAMS+1))
countvec.fit_transform([' '.join(words)])
words = countvec.get_feature_names()