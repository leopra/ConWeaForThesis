#here i load a supervised bert model trained on colab to predict the verticals

# model.save('bert_model.h5')
#torch.save(model, 'bert_model.h5')

import pickle
import torch
import transformers
from transformers import BertTokenizer
import numpy as np
#from PipelinePredictions import BERTClass

basepath = './PipelinePredictions/models/'

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 13)
        #TODO update with sigmoid

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

    def predict(self, ids, mask, token_type_ids):
        outputs = self(ids, mask, token_type_ids)
        _, predicted = torch.max(outputs, 1)
        return predicted

#LOAD MODEL HERE SO IT'S NOT LOADED EVERYTIME
modelo = torch.load(basepath + 'bert_model.h5', map_location=torch.device('cpu'))


def predictBert(strings):
    MAX_LEN = 200
    predsbert = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in strings:
        inputs = tokenizer.encode_plus(
                    line,
                    None,
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    truncation=True
                )
        ids = torch.tensor([inputs['input_ids']], dtype=torch.long)
        mask = torch.tensor([inputs['attention_mask']], dtype=torch.long)
        token_type_ids = torch.tensor([inputs["token_type_ids"]], dtype= torch.long)

        #linear output as it's trained with BCElogitloss
        outputs = modelo(ids, mask, token_type_ids)
        outputs = torch.sigmoid(outputs).detach().numpy()
        outputs = (np.array(outputs) >= 0.5).astype(int)
        predsbert.append(outputs.flatten())

    return predsbert
