#this code shows all the statistics for the training results of COnWea Single Label Contextualized 30k elements
import pickle
import dill
import matplotlib.pyplot as plt
from DATAANALYSIS import wordsStatistics
#this code should be more automatized

#folder = 'perfectditcexpansion'
folder = 'keepinginitial'
#folder = 'standardrun'


stats = dill.load(open('DATAANALYSIS/datalys/'+ folder + '/accuracies.pkl', 'rb'))
dicts = dill.load(open('DATAANALYSIS/datalys/' + folder + '/dicts.pkl', 'rb'))

#NOT USED for now
#comps = dill.load(open('DATAANALYSIS/datalys/standardrun/components.pkl', 'rb'))

#1 plot accuracy changes through iterations
def getAccs(stats, metric):
  keys = list(stats[0].keys())
  keys.remove('accuracy')
  keys.remove('macro avg')
  keys.remove('weighted avg')
  alllabels = dict()
  for j in keys:
    alllabels[j] = []

  for ky in keys:
    for i in range(len(stats)):
      alllabels[ky].append(stats[i][ky][metric])

  return alllabels

yprec = getAccs(stats, 'precision')
yrec = getAccs(stats, 'recall')
yf1 = getAccs(stats, 'f1-score')

data = pickle.load(open('DATAANALYSIS/data47k.pkl', 'rb'))
data['single'] = data.label.apply(lambda x: 1 if sum(x)==1 else 0)
singledata = data[data.single ==1]
vertical_index = {1: 'agriculture',
 2: 'building',
 3: 'constructions',
 4: 'energy',
 5: 'financial services',
 6: 'food & beverage',
 8: 'healthcare',
 9: 'logistics',
 10: 'manufacturing',
 11: 'mining',
 13: 'public administration',
 15: 'transportation',
 16: 'utilities (electricity, water, waste)'}
maponehottolabel = dict(list(zip(list(range(0,13)),sorted(vertical_index.values()))))
singledata['label'] = singledata.label.apply(lambda x: maponehottolabel[list(x).index(1)])

bestdicts = wordsStatistics.savebestDicts(singledata, list(vertical_index.values()))

#2 show dictionary changes through iterations, considering best dictionary possible

def getCountDatasets(itacc):
    keys = list(itacc.keys())
    keys.remove('accuracy')
    keys.remove('macro avg')
    keys.remove('weighted avg')
    for kk in keys:
        print(kk, itacc[kk]['support'])


getCountDatasets(stats[0])

def plotIterations(label, yprec, yrec, yf1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max=1
    div = 20

    N = len(yprec['public administration'])

    ax.axis([0, len(dicts)+1, 0, max])
    plt.title(label)
    plt.plot(list(range(N)), yprec[label], label='prec')
    plt.plot(list(range(N)), yrec[label], label='rec')
    plt.plot(list(range(N)), yf1[label], label='f1')

    for i in range(len(dicts)):
      for n,word in enumerate(sorted(dicts[i][label])):
        if word in dicts[i-1][label] and i != 0:
          ax.text(i,n/div, word, color = 'black')

        if word not in dicts[i - 1][label] and i != 0:
          ax.text(i,n/div, word, color = 'green')

        if i == 0:
          ax.text(i,n/div, word, color = 'black')

      if i != N-1:
        deleted = set(dicts[i][label]) - set(dicts[i+1][label])
        for k, w in enumerate(deleted):
          ax.text((i+1), max - k/div, w, color='red')


    bestw = [ww[0] for ww in bestdicts[label]]
    for h,ww in enumerate(bestw):
      if ww in dicts[len(dicts)-1][label]:
        ax.text(len(dicts)+1, h / div, ww, color='blue')
      else:
        ax.text(len(dicts) + 1, h / div, ww, color='red')

    plt.legend()
    plt.show()
    return

for i in list(vertical_index.values()):
    plotIterations(i, yprec, yrec, yf1)



def plotAverages(stats):
    acc = []
    macro = []
    weig = []
    for keys in stats:
        acc.append(keys['accuracy'])
        macro.append(keys['macro avg']['f1-score'])
        weig.append(keys['weighted avg']['f1-score'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([0, len(dicts) + 1, 0, 1])
    plt.plot(list(range(len(acc))), acc, label='acc')
    plt.plot(list(range(len(acc))), macro, label='macro')
    plt.plot(list(range(len(acc))), weig, label='weig')
    plt.legend()
    return

plotAverages(stats)

#plt.close('all') #close all windows

