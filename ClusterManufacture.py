#in this code i try to analyze the Manufacture class, try to split it in subclasses as professor San Martino said
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
import gensim
from SingleLabelNoContextualization.util import *
from DATAANALYSIS.wordsStatistics import calculateWordStatisticsAsConWea as calcWea
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import matplotlib.pyplot as plt

# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt

#basepath = os.path.join(os.getcwd(), "models")

#blob_download('verticals-ml', 'dataset240k.pkl', 'dataforbert.pkl')

real_vertical_index = pd.read_sql_query("SELECT * FROM [tb_verticals]",sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
real_vertical_index = dict(list(zip(real_vertical_index.id, real_vertical_index.vertical)))

data = pickle.load(open('dataforbert.pkl', 'rb')).reset_index()
mandata = data[data[10]==1]

# LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
# all_content_train = []
# for j,em in enumerate(data.about.values):
#     all_content_train.append(LabeledSentence1(gensim.utils.simple_preprocess(em),[j]))
# print("Number of texts processed: ", j)

#UNCOMMENT IF NEED TO TRAIN MODEL
# d2v_model = Doc2Vec(vector_size = 100, window = 10, min_count = 500, workers=7, dm = 1,alpha=0.025, min_alpha=0.001)
# d2v_model.build_vocab(all_content_train)
# d2v_model.train(all_content_train, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)
#d2v_model2.save('SingleLabelNoContextualization/data/doc2vec240kmodel')


def plot2d(docvectors, n_cluster):
    kmeans_model = KMeans(n_clusters= n_cluster, init='k-means++', max_iter=100)
    X = kmeans_model.fit(docvectors)
    labels=kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(docvectors)
    pca = PCA(n_components=2).fit(docvectors)
    datapoint = pca.transform(docvectors)
    import matplotlib.pyplot as plt
    plt.figure()
    label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#AAAAAA", '#529397', '#BBBBBB']
    color = [label1[i] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='o', s=10, c='#000000')
    plt.show()

#plot the data after kmeans and return kmeans model
def plot3d(docvectors, n_cluster, encoding):
    kmeans_model = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100)
    X = kmeans_model.fit(docvectors)
    labels = kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(docvectors)

    if encoding == 'pca':
        pca = PCA(n_components=3).fit(docvectors)
        datapoint = pca.transform(docvectors)

    if encoding == 'svd':
        clf = TruncatedSVD(3)
        datapoint = clf.fit_transform(docvectors)

    fig = plt.figure()
    label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#AAAAAA", '#529397', '#BBBBBB']
    color = [label1[i] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], datapoint[:,2], c=color)
    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(centroidpoint[:, 0], centroidpoint[:, 1], centroidpoint[:, 2], marker='o', s=10, c="#FFFFFF")
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    plt.show()
    return kmeans_model

#represent the docs with doc2vec
def plotusingDoc2Vec():
    d2v_model2 = Doc2Vec.load('SingleLabelNoContextualization/data/doc2vec240kmodel')
    testdoc_words = [gensim.utils.simple_preprocess(x) for x in mandata.about.values]
    testdoc_vector = np.array([d2v_model2.infer_vector(c) for c in testdoc_words])
    kmeans_model = plot3d(testdoc_vector,5,'pca')
    return testdoc_vector, kmeans_model

#cluster using tfidf
def plotusingTfIdf():
    def preprocessing(line):
        line = line.lower()
        line = re.sub(r"[{}]".format(string.punctuation), " ", line)
        return line
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessing)
    tfidf = tfidf_vectorizer.fit_transform(mandata.about.values)
    kmeans_model = plot3d(tfidf,3,'svd')
    return tfidf, kmeans_model

testdoc_vector, kmeans_model = plotusingDoc2Vec()
#testdoc_vector, kmeans_model = plotusingTfIdf()

results = kmeans_model.predict(testdoc_vector)

#statistics
mandata['cluster'] = results
from collections import Counter
print(Counter(results))
stasforcluster = mandata.groupby('cluster').apply(lambda x: calcWea(3, mandata.about.values, x.about.values))





