import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import time
import datetime
from collections import Counter
import math

# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt

# query not used as i have data in Verticals_Clustering_inputs
#data = pd.read_sql_query("SELECT top(1000) DISTINCT  * FROM ml_crunchbase", sql_cnnt("cnnt", DATABASE_CONFIG_NEW))

#clean dataset to have eutopia verticals
# def cleanDataset(data):
#     cleandataset = pd.DataFrame().reindex_like(data)[0:0]
#     #add empty label column
#     cleandataset['label'] = ''
#
#     notinany=0
#     for i,row in data.iterrows():
#         checker = 0
#         indlist = [x.strip() for x in row['industry'].split(',')]
#         for ind in indlist:
#             if ind in counts:
#                 ui = pd.DataFrame(columns = cleandataset.columns, data = [row.tolist() + [[ind]]])
#                 cleandataset = cleandataset.append(ui, ignore_index=True)
#                 checker = 1
#         if checker == 0:
#             notinany += 1
#
#     return cleandataset

#this method loads the mapping of the cruchbase tag to the eutopia one
#and builds the dictionary
def loadDictionaryXCrunchMapping():
    df = pd.read_excel('./data/CRUNCHMAPPING.xlsx')
    map = dict()
    for i,row in df.iterrows():
        if row['Andrew'] == 1:
            map[row['Crunchbase industries']] = row['Eutop industries']
    return map


def assign_eutop_labels(data):

    # Eutop mapping
    mapp = loadDictionaryXCrunchMapping()

    # Assign to every id a label
    s = time.time()
    vect_tgr = []
    for row in data[["id", "industry"]].values.tolist():
        industrylist = [x.strip() for x in row[1].split(',')]
        vect_tgr = vect_tgr + [[row[0], mapp[c]] for c in industrylist if c in mapp.keys()]
    e = time.time()
    print(str(datetime.timedelta(seconds=e-s))) #  on my machine '0:03:18.140845' ~ 3 minutes

    # Check 1
    if [len(g) for g in vect_tgr if len(g) != 2] != []:
        raise Exception("Problem with labels - 1")

    # Check 2
    if len(list(set([g[0] for g in vect_tgr]))) > len(data):
        raise Exception("Problem with labels - 2")

    # Transfor data into dataframe
    df_4_merge = pd.DataFrame(data = vect_tgr, columns = ["id", "eutopia_label"])
    df_4_merge.shape
    # Left join
    df_4_merge_1 = df_4_merge.merge(data, how = "left", right_on="id", left_on="id")
    df_4_merge_1.shape

    # Check 3
    if len(df_4_merge) != len(df_4_merge_1):
        raise Exception("Problem with left join")

    return df_4_merge_1

# this function takes a dataframe creates a column 'listlabel' from the column industry, is then
# used for one-hot-encoding
def assign_eutop_labelsV2(data):

    # Eutop mapping
    mapp = loadDictionaryXCrunchMapping()
    def listFromStr(stringa):
        industrylist = [x.strip() for x in stringa.split(',')]
        return list(set([mapp[c] for c in industrylist if c in mapp.keys()]))

    data['listlabel'] = data['industry'].apply(lambda x: listFromStr(x))
    # Assign to every id a label


    return data

#create a mappin from tags to most important verticals
#keeps track of how many time each tag appeared
def createMapping():
    # query that keeps distinct urls and hopefully also distinct companies
    data = pd.read_sql_query(
        "select * from ml_crunchbase  where id in (select min(id) from ml_crunchbase group by cb_url)",
        sql_cnnt("cnnt", DATABASE_CONFIG_NEW))

    # get all industries labels, strip removes trailing spaces
    industries = data['industry'].values
    catlist = list(map(str.strip, ','.join(industries).split(',')))

    verticals = ["Agriculture", "Building",
                 "Constructions", "Energy", "Financial services", "Food & beverage", "Forestry", "Healthcare",
                 "Logistics", "Manufacturing",
                 "Mining", "Other activities", "Public administration", "Telecommunications & ICT", "Transportation",
                 "Utilities (electricity, water, waste)"]

    len(Counter(catlist).keys())  # equals to list(set(words))
    Counter(catlist).values()  # counts the elements' frequency

    # get the category from source
    sourcecategories = data['source'].apply(lambda x: x.split(',')[1].strip())
    counts = list(Counter(sourcecategories).keys())
    # remove the All key because it is useless for mapping
    counts.remove('All')

    tagcrunchcount = dict()
    mapping = dict()

    for indust in industries:
        indlist = list(map(str.strip, indust.split(',')))
        for k in indlist:
            if k in tagcrunchcount.keys():
                tagcrunchcount[k] += 1
            else:
                tagcrunchcount[k] = 1
        for item in counts:
            if item in indlist:
                    if k in mapping.keys():
                       if item in mapping[k].keys():
                           mapping[k][item] +=1
                       else:
                           mapping[k][item] = 1
                    else:
                        mapping[k] = dict()
                        mapping[k][item] = 1

    return mapping, tagcrunchcount

#this function is applied to the mapping between the crunchbase tags,
#show the distribution of tags in relation to the verticals (of crunchbase)
#used too see how to pair crunchbase mappings (CRUNCHBASEMAPPINGS.xlml)
def plotHistDistrib(mapping, tagcrunchcount, tag):
    distr = mapping[tag]
    notpaired = tagcrunchcount[tag] - sum(mapping[tag].values())
    toplot = list(mapping[tag].values())
    toplot.append(notpaired)
    xlabel = [x[0:6] for x in list(mapping[tag].keys())]

    xlabel.append('NONE')
    plt.bar(xlabel,toplot)
    return

#this method creates as much balanced dataset as classes and returns a list of dataframes
# classes are labeled binarily 'class-name','Other
def create1vsAllDatasets(data):
    dataframes = []
    data = assign_eutop_labels(data)
    otherlabel = len(data['eutopia_label'].unique()) - 1
    alllabels = data['eutopia_label'].unique()
    classes = data.groupby('eutopia_label', as_index=False)
    s = time.time()
    for label in alllabels:
        pred1 = data[data['eutopia_label'] == label]
        nsamples = math.ceil(len(pred1) / otherlabel)
        pred0 = classes.apply(lambda x: x.sample(nsamples) if len(x) >= nsamples else x.sample(len(x)))
        pred0 = pred0[pred0['eutopia_label'] != label]
        pred0['eutopia_label'] = ['Other'] * len(pred0)
        dataframes.append(pd.concat([pred1, pred0]))
    e = time.time()
    print('tempo creazione frames: ' + str(datetime.timedelta(seconds=e-s)))
    return dataframes




