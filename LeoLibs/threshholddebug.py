import numpy as np
import pickle
import keras

def generate_pseudo_labels(df, labels, label_term_dict, tokenizer, argmaxparam):


    # this an implementation for multilabel, returns a one-hot-encoded array
    def argmax_multilabel(count_dict, percentage=0.4):
        total = 0
        labcounts = []
        for l in labels:
            count = 0
            try:
                for t in count_dict[l]:
                    count += count_dict[l][t]
            except:
                pass
            labcounts.append((l, count))
            total += count

        current = np.zeros(len(labels))

        # add 1 to labels over the threshold
        for i in range(len(current)):
            if (labcounts[i][1] / total) >= percentage:
                current[i] = 1.0

        # if there was no label over the threshold give the best one
        if np.sum(current) == 0:
            labcounts = [x[1] for x in labcounts]
            index_max = max(range(len(labcounts)), key=labcounts.__getitem__)
            current[index_max] = 1.0

        return current

        # TODO DEBUG
        # x = {'a': {'pane':3, 'riso':2}, 'b': {'pesce':10, 'carne':22}, 'c': {'papate': 99, 'gamb': 101}}
        # argmax_multilabel(x, 0.2)

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        line = row["sentence"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1

        #only saves data for nn training if at least one seedword was found (obviously)
        if flag:
            lbl = argmax_multilabel(count_dict, argmaxparam)
            # TODO currently is impossible that there is no label, in the future maybe this should be possible
            if np.sum(lbl) == 0:
                continue
            y.append(lbl)
            X.append(line)
            y_true.append(label)
    return X, y, y_true


labels = sorted(['Mining', 'Food & Beverage', 'Financial services', 'Constructions', 'Public Administration',
                 'Utilities (electricity, water, waste)', 'Buildings', 'Transportation', 'Energy', 'Logistics',
                 'Agriculture', 'Healthcare', 'Manufacturing'])
label_term_dict = {
    'Public Administration': ['education', 'government$0', 'government$1', 'administration$0', 'administration$1',
                              'regulation', 'jj6', 'jj7', 'jj8', 'department'],
    'Healthcare': ['health', 'diseases', 'blood', 'healthcare', 'dental', 'pharmaceutical', 'cancer', 'therapeutic',
                   'medical'],
    'Food & Beverage': ['food', 'beverages$0', 'beverages$1', 'eat', 'drink', 'ingredients$0', 'ingredients$1', 'jj6',
                        'jj7', 'jj8', 'jj9', 'protein', 'vegan'],
    'Buildings': ['jj0', 'properties$0', 'properties$1', 'properties$2', 'rent', 'residential', 'jj5', 'jj6', 'jj7',
                  'housing'],
    'Transportation': ['car$0', 'car$1', 'bus', 'vehicle', 'transit', 'jj7', 'traffic$0', 'traffic$1'],
    'Constructions': ['construction', 'builder', 'architecture', 'infrastructure$0', 'infrastructure$1', 'jj9'],
    'Manufacturing': ['production', 'manufacturer$0', 'manufacturer$1', 'manufacturer$2', 'manufacture', 'production',
                      'industry$0', 'industry$1', 'jj6', 'manufacturing', 'jj8', 'jj9'],
    'Utilities (electricity, water, waste)': ['water', 'electrical', 'waste', 'recycling', 'jj5', 'jj6', 'jj7', 'jj8',
                                              'jj9', 'electricity', 'wastewater'],
    'Energy': ['solar', 'energy', 'thermal', 'power$0', 'power$1', 'power$2', 'power$3', 'power$4', 'jj6', 'jj7',
               'photovoltaic', 'hybrid'], 'Mining': ['mining', 'mineral', 'drilling', 'jj9'],
    'Financial services': ['asset', 'portfolio', 'finance', 'invest$0', 'invest$1', 'invest$2', 'bank', 'credit$0',
                           'credit$1', 'jj6', 'banking', 'insurance$0', 'insurance$1', 'financial'],
    'Agriculture': ['farming', 'plants$0', 'plants$1', 'cultivation', 'soil', 'nutrient'],
    'Logistics': ['delivery', 'shipping', 'freight', 'supply$0', 'supply$1', 'jj7', 'jj8', 'logistics',
                  'distribution$0', 'distribution$1', 'distribution$2']}

dataset_path = '../data/eutopiaverttest/'
tokenizer = pickle.load(open(dataset_path + "tokenizer.pkl", "rb"))

pkl_dump_dir = dataset_path
df = pickle.load(open(pkl_dump_dir + "df_contextualized.pkl", "rb"))
# word_cluster = pickle.load(open(pkl_dump_dir + "word_cluster_map.pkl", "rb"))

#smalldf = df[9:10]

#for a in range(10):
a=3/10
X,y,y_true = generate_pseudo_labels(df, labels, label_term_dict, tokenizer, a)

catacc = keras.metrics.CategoricalAccuracy()
catacc.update_state(y_true, y)
print(a, "result: ", catacc.result().numpy())