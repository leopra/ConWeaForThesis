import debugBert
import pickle
import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

dataset_path = './data/eutopiavert/'
df = pickle.load(open(dataset_path + "df_contextualized.pkl", "rb"))
tokenizer = pickle.load(open(dataset_path + "tokenizer.pkl", "rb"))

dictionaries = {}
for file in sorted(os.listdir('debugdata')):
    if file.endswith(".json"):
        with open('debugdata/' + file) as fp:
            dictionaries[file] = json.load(fp)

out = []
#dict analysis
for i in dictionaries.keys():
    nseed = []
    total = 0
    for k in dictionaries[i]:
        nseed.append((k, len(dictionaries[i][k])))
        total += len(dictionaries[i][k])
    out.append({'dict': i, 'total': total, 'values':nseed})


dataset_path = './data/eutopiavert/'

df = pickle.load(open(dataset_path + "df_contextualized.pkl", "rb"))

tokenizer = pickle.load(open(dataset_path + "tokenizer.pkl", "rb"))

labels = list(set(dictionaries[list(dictionaries.keys())[0]]))

fig, axs = plt.subplots(1,len(dictionaries.keys()) + 1, figsize=(15, 30))
for n,i in enumerate(dictionaries.keys()):
    X, y, y_true = debugBert.generate_pseudo_labels(df, labels, dictionaries[i], tokenizer)
    twodmatrix = np.stack(y, axis=0)
    labelcounts = np.sum(twodmatrix, axis=0)
    axs[n].bar(range(0, 13), labelcounts)
    axs[n].set_ylim(0, 12000)
    axs[n].title.set_text(i)

twodmatrix = np.stack(y_true, axis=0)
labelcounts = np.sum(twodmatrix, axis=0)
axs[-1].bar(range(0, 13), labelcounts)
axs[-1].set_ylim(0, 12000)
axs[-1].title.set_text('real distribution')
