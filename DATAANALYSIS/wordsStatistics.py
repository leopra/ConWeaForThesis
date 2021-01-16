import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from SingleLabelNoContextualization.util import *
import pickle
# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt
from AzureStorage import blob_upload, blob_download

# #blob_download('verticals-ml', 'dataset240k.pkl', 'dataforbert.pkl')
#
# real_vertical_index = pd.read_sql_query("SELECT * FROM [tb_verticals]",sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
# real_vertical_index = dict(list(zip(real_vertical_index.id, real_vertical_index.vertical)))
#
# data = pickle.load(open('dataforbert.pkl', 'rb'))
#
# data['products'] = data.about.apply(lambda x: 1 if 'products' in x else 0)
#
# notmanifac = data[(data['products'] == 1) & (data[10] == 0)]
#
# #distribution of the word products in all other classes
# columns = [x for x in list(real_vertical_index.keys()) if x not in [12,14,7]]
# notmanifac[columns].sum(axis=0)
#
# mandata = data[data[10]==1]


#see combination of manifacture with other classes
#mandata['single'] = data.listlabel.apply(lambda x: 1 if np.sum(x) == 1 else 0)
#notmanifac = data[(data[10] == 1) & (np.sum(data[columns].values) > 2)] #TODO does not work anymore

def calculateWordStatisticsAsConWea(ngramms, whole_dataset_as_text, dataofsubclass):
    ngramsss = ngramms
    df = whole_dataset_as_text
    docs = dataofsubclass

    doc_freq_thresh = 0
    docfreq = calculate_df_doc_freq(df,ngramsss)
    inv_docfreq = calculate_inv_doc_freq(df, docfreq)

    components = {}
    docfreq_local = calculate_doc_freq(docs,ngramsss)
    vect = CountVectorizer(tokenizer=lambda x: x.split(), ngram_range=(1,ngramsss))
    X = vect.fit_transform(docs)
    rel_freq = X.sum(axis=0) / len(docs)
    rel_freq = np.asarray(rel_freq).reshape(-1)
    names = vect.get_feature_names()
    E_LT = np.zeros(len(vect.vocabulary_))

    word_to_index = vect.vocabulary_

    for i, name in enumerate(names):
        try:
            if docfreq_local[name] < doc_freq_thresh:
                continue
        except:
            continue

        # docfreq local = count of document of specific label containing that word
        # docfreq = count of all document of that specific class
        # inv_doc_freq = log(number of documents, word frequency in all documents) to get unusual words
        # tanh(relative frequency of word in document of specific class)
        E_LT[word_to_index[name]] = (docfreq_local[name] / docfreq[name]) * inv_docfreq[
            name] * np.tanh(rel_freq[i])
        components[name] = {"reldocfreq": docfreq_local[name] / docfreq[name],
                               "idf": inv_docfreq[name],
                                   "rel_freq": np.tanh(rel_freq[i]),
                                   "rank": E_LT[word_to_index[name]]}

    gettoptwenty = sorted(components.items(), key = lambda x: x[1]['rank'], reverse=True)[:80]
    return gettoptwenty


def savebestDicts(df, labels):
    dictio = {}
    grouped = df.groupby('label')
    for i in labels:
        subset = grouped.get_group(i)
        topn = calculateWordStatisticsAsConWea(1, df.sentence.values, subset.sentence.values)
        dictio[i] = topn
    return dictio


