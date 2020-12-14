from TextRank import Textrank
import pickle
from keras.preprocessing.text import Tokenizer
from gensim.models import word2vec
import numpy as np
from scipy import spatial
import re
import nltk
nltk.download('stopwords')

def cleanTex(descrlist):
    REGEX_URL = r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    okdesc = list(filter(lambda x: [] if 'home' in x.lower() and len(x) < 50 else x, descrlist))
    out = []
    for a in okdesc:
        a = re.sub(REGEX_URL, '', a) #remove urls
        a = re.sub('[?!]', '' , a) #remove question marks
        a = re.sub(r"(\w)\1{2,}", '', a) #remove wrong words #aaaa fffff
        a = ' '.join(a.split()) #remove trailing spaces
        out.append(a)

    return out

def train_word2vec(strings):
    basepath = 'TextRank/'
    def fit_get_tokenizer(data, max_words):
        tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(data)
        return tokenizer

    def get_embeddings(inp_data, vocabulary_inv, size_features=100,
                       mode='skipgram',
                       min_word_count=2,
                       context=5):
        num_workers = 15  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in inp_data]
        if mode == 'skipgram':
            sg = 1
            print('Model: skip-gram')
        elif mode == 'cbow':
            sg = 0
            print('Model: CBOW')
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            sg=sg,
                                            size=size_features,
                                            min_count=min_word_count,
                                            window=context,
                                            sample=downsampling)
        embedding_model.init_sims(replace=True)
        embedding_weights = np.zeros((len(vocabulary_inv) + 1, size_features))
        embedding_weights[0] = 0
        for i, word in vocabulary_inv.items():
            if word in embedding_model:
                embedding_weights[i] = embedding_model[word]
            else:
                embedding_weights[i] = np.random.uniform(-0.25, 0.25, embedding_model.vector_size)

        return embedding_weights

    tokenizer = fit_get_tokenizer(strings, max_words=150000)
    print("Total number of words: ", len(tokenizer.word_index))
    tagged_data = tokenizer.texts_to_sequences(strings)
    vocabulary_inv = {}
    for word in tokenizer.word_index:
        vocabulary_inv[tokenizer.word_index[word]] = word
    embedding_mat = get_embeddings(tagged_data, vocabulary_inv)
    pickle.dump(tokenizer, open(basepath + "tokenizer.pkl", "wb"))
    pickle.dump(embedding_mat, open(basepath + "embedding_matrix.pkl", "wb"))
    return embedding_mat, tokenizer

with open('TextRank/verticals-dataset.pkl', 'rb') as f:
    df = pickle.load(f)

embedding, tokenizer = train_word2vec(df.about)


def cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)

wordsstr = list(tokenizer.word_index.keys())[30:80]

man = embedding[tokenizer.word_index['manufacture']]
prod = embedding[tokenizer.word_index['production']]


cosine_similarity(man, prod)
for i in wordsstr:
    print(i,cosine_similarity(man, embedding[tokenizer.word_index[i]]))

tr4w = Textrank.TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
tr4w.get_keywords(10)