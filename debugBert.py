import pickle
from flair.embeddings import BertEmbeddings
from nltk import sent_tokenize
from flair.data import Sentence
from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
embedding = BertEmbeddings('bert-base-uncased')

with open('./data/eutopiavertzz/df.pkl', 'rb') as handle:
    df = pickle.load(handle)

len(df.iloc[14].sentence)

for index, row in df[14:15].iterrows():
    if index % 100 == 0:
        print("Finished sentences: " + str(index) + " out of " + str(len(df)))
    # all sentences are undercase now
    line = row["sentence"].lower()
    sentences = sent_tokenize(line)
    for sentence_ind, sent in enumerate(sentences):
        tokenized_text = tokenizer.tokenize(sent)
        if len(tokenized_text) > 512:
            print('sentence too long for Bert: truncating')
            sentence = Sentence(' '.join(sent[:512]), use_tokenizer=True)
        else:
            sentence = Sentence(sent, use_tokenizer=True)
        try:
            embedding.embed(sentence)
        except Exception as e:
            print("Exception Counter while getting BERT: ", sentence_ind, index, e)
            print(sentence)
            print(index)
            continue

#CODE USED TO SAVE FILES TO AZURE BLOB STORAGE
#
# import os
# import sys
# PATH = os.path.dirname(os.__file__)
# sys.path.append(os.path.join(PATH, 'Libraries-GP'))
#
# from AzureStorage import blob_upload
# blob_upload('verticals-ml', 'cluster-map1,8gb', './data/eutopiavert/word_cluster_map.pkl')
