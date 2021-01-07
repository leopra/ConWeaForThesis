#this code shows all the statistics for the training results of COnWea Single Label Contextualized 30k elements
import pickle
import dill
import matplotlib.pyplot as plt
import wordsStatistics
#this code should be more automatized
stats = dill.load(open('datalys/accuracies.pkl','rb'))
dicts = dill.load(open('datalys/dicts.pkl','rb'))
comps = dill.load(open('datalys/components.pkl','rb'))

metrics = 'precision'

#1 plot accuracy changes through iterations
def getAccs(stats):
  keys = list(stats[0].keys())
  keys.remove('accuracy')
  keys.remove('macro avg')
  keys.remove('weighted avg')
  alllabels = dict()
  for j in keys:
    alllabels[j] = []

  for ky in keys:
    for i in range(len(stats)):
      alllabels[ky].append(stats[i][ky][metrics])

  return alllabels

y = getAccs(stats)

data = pickle.load(open('data47k.pkl','rb'))
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

#fig, axs = plt.subplots(13)
for i,val in enumerate(y.keys()):
  plt.plot(list(range(6)),y[val], label = val)

plt.legend()
plt.show()

#2 show dictionary changes through iterations, considering best dictionary possible
label = 'manufacturing'
fig = plt.figure()
ax = fig.add_subplot(111)
max=1
div = 20



ax.axis([0, len(dicts)+1, 0, max])
plt.plot(list(range(6)), y[label], label=label)
for i in range(len(dicts)):
  for n,word in enumerate(sorted(dicts[i][label])):
    if word in dicts[i-1][label] and i != 0:
      ax.text(i,n/div, word, color = 'black')

    if word not in dicts[i - 1][label] and i != 0:
      ax.text(i,n/div, word, color = 'green')

    if i == 0:
      ax.text(i,n/div, word, color = 'black')

  if i != 5:
    deleted = set(dicts[i][label]) - set(dicts[i+1][label])
    for k, w in enumerate(deleted):
      ax.text((i+1), max - k/div, w, color='red')


bestw = [ww[0] for ww in bestdicts[label]]
for h,ww in enumerate(bestw):
  if ww in dicts[len(dicts)-1][label]:
    ax.text(len(dicts)+1, h / div, ww, color='blue')
  else:
    ax.text(len(dicts) + 1, h / div, ww, color='red')

plt.show()




