import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import time
import json
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import string
from collections import defaultdict
import os

'''
    This function prepares the numericalized data in the form of lists, to be used for training, test and evaluation.
        x:  list[num_docs, sentences_per_doc, sentence_embedding_dim] 
        y:  list[num_docs, sentences_per_doc]
'''
def prepare_data_new(train_data_dict, dev_data_dict, test_data_dict, args, dim, tag2idx=None):
    x_train, y_train = [], []
    x_dev, y_dev = [], []
    x_test, y_test = [], []
    
    word2idx = defaultdict(lambda: len(word2idx))
    if tag2idx is None:
        tag2idx = defaultdict(lambda: len(tag2idx))
        tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2
    

    # map the special symbols first
    word2idx['<pad>'], word2idx['<unk>'] = 0, 1

    # iterate over documents
    for doc_names in train_data_dict.keys():
        doc = train_data_dict[doc_names]
        embeddings, labels = doc['embeddings'], doc['labels']
        doc_x, doc_y = [], [] 
            
            # iterate over sentences
        for emb,label in zip(embeddings, labels):
            sent_x, sent_y = emb, label
            
            sent_x = list(map(float, sent_x[:dim]))
            sent_y = tag2idx[sent_y]

            if sent_x != []:
                doc_x.append(sent_x)
                doc_y.append(sent_y)
        
        x_train.append(doc_x)
        y_train.append(doc_y)
        
    # iterate over documents
    for doc_names in dev_data_dict.keys():
        doc = dev_data_dict[doc_names]
        embeddings, labels = doc['embeddings'], doc['labels']
        doc_x, doc_y = [], [] 
            
            # iterate over sentences
        for emb,label in zip(embeddings, labels):
            sent_x, sent_y = emb, label
            
            sent_x = list(map(float, sent_x[:dim]))
            sent_y = tag2idx[sent_y]

            if sent_x != []:
                doc_x.append(sent_x)
                doc_y.append(sent_y)
        
        x_dev.append(doc_x)
        y_dev.append(doc_y)
        
    # iterate over documents
    for doc_names in test_data_dict.keys():
        doc = test_data_dict[doc_names]
        embeddings, labels = doc['embeddings'], doc['labels']
        doc_x, doc_y = [], [] 
            
            # iterate over sentences
        for emb,label in zip(embeddings, labels):
            sent_x, sent_y = emb, label
            
            sent_x = list(map(float, sent_x[:dim]))
            sent_y = tag2idx[sent_y]

            if sent_x != []:
                doc_x.append(sent_x)
                doc_y.append(sent_y)
        
        x_test.append(doc_x)
        y_test.append(doc_y)

    return x_train, y_train, x_dev, y_dev, x_test, y_test, word2idx, tag2idx
