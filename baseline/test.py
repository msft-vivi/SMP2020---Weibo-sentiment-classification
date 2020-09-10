# import torch
# from torch.nn import CrossEntropyLoss
# from transformers import BertModel, BertForPreTraining, BertConfig, BertTokenizer
# from net.bert_transfer_learning import BertTransferLearning
# from net.bert_uer import BertUER
# import torch
#
# config = BertConfig.from_pretrained('/mnt/yzf/yang/baseline/uer_weibo_model',num_labels=4)
# config.ignore_weights = ['classifier']
# args = {}
# config.ignore_weights = None
# model = BertUER.from_pretrained('/mnt/yzf/yang/baseline/uer_weibo_model',
#                                                       config=config, args=args)
# tokenizer = BertTokenizer.from_pretrained('/mnt/yzf/yang/baseline/uer_weibo_model')
# device = torch.device('cpu')
# model_weight = torch.load('/mnt/yzf/yang/baseline/uer_weibo_model/pytorch_model.bin')
# print(model.bert.embeddings.word_embeddings.weight.to(device) == model_weight['embedding.word_embedding.weight'].to(device))
# print(len(tokenizer), model.bert.embeddings.word_embeddings.weight.size())

import pickle
import os
import numpy as np
import json
def load_X_train(path):
    path = os.path.join('virus_k_fold_model', path)
    path = os.path.join(path, 'oof_test')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return np.array(data)


with open('voting_result2/virus/virus_result.txt', 'r', encoding='utf8') as f:
    target = json.load(f)
target_label = [int(example['label']) for example in target]

paths = ['roberta_wwm_ext_transfer_learning',
 'roberta_wwm_ext_lstm_attention_transfer_learning1',]

oof_1 = load_X_train(paths[0])
oof_2 = load_X_train(paths[1])

same_nums = (np.argmax(oof_1, axis=-1) == np.argmax(oof_2, axis=-1))
print(same_nums)
print(np.sum(same_nums)/oof_1.shape[0])