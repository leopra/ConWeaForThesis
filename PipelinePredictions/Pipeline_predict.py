import pickle
import argparse
import json
import gc
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import string as str_ing
import numpy as np
from keras.models import load_model
from gensim.models import word2vec
from keras_han.layers import AttentionLayer
from keras_han.model import HAN
from nltk import tokenize
from flair.embeddings import BertEmbeddings
from pytorch_pretrained_bert import BertTokenizer
from nltk import sent_tokenize
from flair.data import Sentence
from scipy import spatial

#path where all the pretrained models are saved
basepath = './PipelinePredictions/models/'
#tokenizer
tokenizer = pickle.load(open(basepath+ "tokenizer.pkl", "rb"))

#this function learns the word embeddings ans saves them in the file embedding_matrix.pkl
#TODO why word2vec and not fasttext?
def train_word2vec(strings):
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
    tagged_data = tokenizer.texts_to_sequences(df.sentence)
    vocabulary_inv = {}
    for word in tokenizer.word_index:
        vocabulary_inv[tokenizer.word_index[word]] = word
    embedding_mat = get_embeddings(tagged_data, vocabulary_inv)
    return embedding_mat
    pickle.dump(tokenizer, open(basepath + "tokenizer.pkl", "wb"))
    pickle.dump(embedding_mat, open(basepath + "embedding_matrix.pkl", "wb"))

#new data needs to be processed with bert to disambiguate word meanings
#encoded with bert then checks for the nearest cluster
#also is undercased and classic preprocesses
def preprocess(strings):
    #this data structure is a dictionary that saves the words and the centroid vector for each meaning found by kmeans
    # bank$0 corresponds to the first element
    word_cluster = pickle.load(open(basepath + "word_cluster_map.pkl", "rb"))
    def get_vec(word, word_cluster, stop_words):
        if word in stop_words:
            return []
        t = word.split("$")
        if len(t) == 1:
            prefix = t[0]
            cluster = 0
        elif len(t) == 2:
            prefix = t[0]
            cluster = t[1]
            try:
                cluster = int(cluster)
            except:
                prefix = word
                cluster = 0
        else:
            prefix = "".join(t[:-1])
            cluster = t[-1]
            try:
                cluster = int(cluster)
            except:
                cluster = 0

        word_clean = prefix.translate(str.maketrans('', '', str_ing.punctuation))
        if len(word_clean) == 0 or word_clean in stop_words:
            return []
        try:
            vec = word_cluster[word_clean][cluster]
        except:
            try:
                vec = word_cluster[prefix][cluster]
            except:
                try:
                    vec = word_cluster[word][0]
                except:
                    vec = []
        return vec

    print("Preprocessing data..")
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    word_vec = {}
    for index,line in enumerate(strings):
        if index % 100 == 0:
            print("Finished rows: " + str(index) + " out of " + str(len(strings)))

        words = line.strip().split()
        new_words = []
        for word in words:
            try:
                vec = word_vec[word]
            except:
                vec = get_vec(word, word_cluster, stop_words)
                if len(vec) == 0:
                    continue
                word_vec[word] = vec
            new_words.append(word)
        strings[index] = " ".join(new_words)

    #i guess it's to free memory
    del word_cluster
    gc.collect()
    return strings, word_vec


#function that uses a tokenizer and a dictionary of seedowrds to predict labels
def generate_pseudo_labels(strings):

    #labels and dictionary of seedwords
    with open(basepath + "seedwordsencoded.json") as fp:
        label_term_dict = json.load(fp)
    labels = sorted(label_term_dict.keys())

    #this an implementation for multilabel, returns a one-hot-encoded array
    def argmax_perfectmatch(count_dict, percentage=0.2):
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
            # if i have only match of less than 3 classes assign all of them
            if len(labcounts) < 3:
                if labcounts[i][1] != 0:
                    current[i] = 1.0

            # if they are more check for threshold
            else:
                if (labcounts[i][1] / total) >= percentage:
                    current[i] = 1.0

        # if there was no label over the threshold give the best one
        if np.sum(current) == 0:
            labcounts = [x[1] for x in labcounts]
            index_max = max(range(len(labcounts)), key=labcounts.__getitem__)
            current[index_max] = 1.0

        return current

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w

    for line in strings:
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
        if flag:
            lbl = argmax_perfectmatch(count_dict)
            #currently is impossible that there is no label, in the future maybe this should be possible
            if np.sum(lbl) == 0:
                continue
            y.append(lbl)
            X.append(line)
    return X, y

#this function encodes strings to integer and pads/trim using the tokenizer pretrained in /models
def prep_data_for_HAN(max_sentence_length, max_sentences, texts, tokenizer):
    data = np.zeros((len(texts), max_sentences, max_sentence_length), dtype='int32')
    documents = []
    for text in texts:
        # added lowercase
        sents = tokenize.sent_tokenize(text.lower())
        documents.append(sents)
    for i, sentences in enumerate(documents):
        tokenized_sentences = tokenizer.texts_to_sequences(
            sentences
        )
        tokenized_sentences = pad_sequences(
            tokenized_sentences, maxlen=max_sentence_length
        )

        pad_size = max_sentences - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:max_sentences]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )

        data[i] = tokenized_sentences[None, ...]
    return data

#PREDICT WITH HAN NETWORK
#returns one-hot-encoding
def predictWithHAN(strings, word_vec):
    max_sentence_length = 100
    #TODO this value could be less, study necessary
    max_sentences = 15
    tokenizer = pickle.load(open(basepath + "tokenizer.pkl", "rb"))

    strings = df.sentence.values
    strings = prep_data_for_HAN(texts=strings, max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)

    # load model (embedding matrix already saved inside the model
    model = load_model(basepath + 'model_conwea.h5',
                       custom_objects={'AttentionLayer': AttentionLayer, 'HAN': HAN})

    model.load_weights(basepath + 'model_weights_conwea.h5')

    #convert real number to binary classification
    pred = model.predict(strings)
    pred = (pred > 0.5).astype(int)
    return pred

#this function contextualizes each word with the most similar cluster assigning (nothing/$0,$1..)
def contextualizeSentences(strings, word_cluster):

    def cosine_similarity(a, b):
        return 1 - spatial.distance.cosine(a, b)

    def to_tokenized_string(sentence):
        tokenized = " ".join([t.text for t in sentence.tokens])
        return tokenized

    def get_cluster(tok_vec, cc):
        max_sim = -10
        max_sim_id = -1
        for i, cluster_center in enumerate(cc):
            sim = cosine_similarity(tok_vec, cluster_center)
            if sim > max_sim:
                max_sim = sim
                max_sim_id = i
        return max_sim_id

    out = []
    embedding = BertEmbeddings('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for index,string in enumerate(strings):
        print("Contextualizing the corpus ", index)
        stop_words = set(stopwords.words('english'))
        stop_words.add('would')

        # this tokenizer is used to check for length > 512
        sentences = sent_tokenize(string)
        for sentence_ind, sent in enumerate(sentences):
            tokenized_text = tokenizer.tokenize(sent)
            if len(tokenized_text) > 512:
                print('sentence too long for Bert: truncating')
                sentence = Sentence(' '.join(sent[:512]), use_tokenizer=True)
            else:
                sentence = Sentence(sent, use_tokenizer=True)
            try:
                embedding.embed(sentence)
            except:
                print(index)
                print(sentence)
            for token_ind, token in enumerate(sentence):
                word = token.text
                if word in stop_words:
                    continue
                word_clean = word.translate(str.maketrans('', '', str_ing.punctuation))
                if len(word_clean) == 0 or word_clean in stop_words or "/" in word_clean:
                    continue
                try:
                    cc = word_cluster[word_clean]
                except Exception as e:
                    print("Exception Counter while getting clusters: ", index, e)
                    continue
                    # try:
                    #     cc = word_cluster[word]
                    # except:
                    #     word_clean_path = cluster_dump_dir + word_clean + "/cc.pkl"
                    #     word_path = cluster_dump_dir + word + "/cc.pkl"
                    #     try:
                    #         with open(word_clean_path, "rb") as handler:
                    #             cc = pickle.load(handler)
                    #         word_cluster[word_clean] = cc
                    #     except:
                    #         try:
                    #             with open(word_path, "rb") as handler:
                    #                 cc = pickle.load(handler)
                    #             word_cluster[word] = cc
                    #         except Exception as e:

                if len(cc) > 1:
                    tok_vec = token.embedding.cpu().numpy()
                    cluster = get_cluster(tok_vec, cc)
                    sentence.tokens[token_ind].text = word + "$" + str(cluster)
            sentences[sentence_ind] = to_tokenized_string(sentence)
            out.append(" . ".join(sentences))
    return out

# #DEBUG
# df = pickle.load(open(basepath + "df.pkl", "rb"))
# strings = df.sentence.values
# aas = train_word2vec(strings)
# x, y =preprocess(strings)
#
# word_cluster = pickle.load(open(basepath + "word_cluster_map.pkl", "rb"))
# contextualizeSentences(strings, word_cluster)

