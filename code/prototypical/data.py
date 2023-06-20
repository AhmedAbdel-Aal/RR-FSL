import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch
import os
import random
import time
import datetime
import json
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import collections
class Tokenizer:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def encode(self, sentence):
        return self.tokenizer.encode_plus(sentence, padding='max_length', truncation=True, return_tensors='pt', max_length=256)

    
class DataCollector(Dataset):
    def __init__(self, file_path, tokenizer):
        self.compressed_label_mapper = {
        "Fact": 0,
        "Argument": 1, "Argument": 1,
        "RulingP": 2, "RulingL": 3, 
        "Ratio": 4,
        "None": -1,
        "Statute": 5,
        "Precedent": 6,
        "Issue": 0,"Dissent": 0
         }
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.load_documents()
        
    def load_documents(self):
        self.data = pd.read_csv(self.file_path)
        self.sentences = self.data['sentences'].tolist()
        labels = self.data['labels'].tolist()
        self.labels = [self.compressed_label_mapper[label] for label in labels]
    
    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
                
        sentences = self.sentences[idx]        
        label = torch.tensor(self.labels[idx])
        
        encoded_input = self.tokenizer.encode(sentences)
        input_ids = encoded_input['input_ids'].squeeze()
        attention_mask = encoded_input['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentences':sentences,
            'labels': label
        }
    
    
    
class SequenceClassificationDataset(Dataset):
    def __init__(self, file_path, labels_subset):
        self.l_mapper = {
            "Fact": 0,
            "ArgumentPetitioner": 1,
            "ArgumentRespondent": 1,
            "RulingByPresentCourt": 2,
            "RulingByLowerCourt": 3,
            "RatioOfTheDecision": 4,
            "None": -1,
            "Statute": 5,
            "PrecedentReliedUpon": 6,
            "PrecedentNotReliedUpon": 6,
            "PrecedentOverruled": 6,
            "Issue": 0,
            "Dissent": 0
        }
        self.file_path = file_path
        self.labels_subset = labels_subset
        self.data = self.load_documents()
        self.process_data()
        print(f"loaded data with {len(self.documents)} sentence embedding with labels subset {np.unique(self.doc_labels)}")
        
    def load_documents(self):
            with open(self.file_path, "rb") as file:
                data_dict = pickle.load(file)
            return [(doc_name, doc_data) for doc_name, doc_data in data_dict.items()]
    
    def process_data(self):
        self.documents = []
        self.doc_lengths = []
        self.doc_labels = []
        
        for doc_name, doc_data in self.data:
            doc_emb = doc_data["embeddings"]
            doc_label = doc_data["labels"]
            doc_label = [self.l_mapper[l] for l in doc_label]
            
            for emb, label in zip(doc_emb, doc_label):
                emb = torch.tensor(emb)
                if label in self.labels_subset:
                    self.documents.append(emb)
                    self.doc_labels.append(label)
        assert len(self.documents) == len(self.doc_labels)
                    

    def __len__(self):
        return len(self.documents)
    
    def get_labels(self):
        return self.doc_labels
    
    def __getitem__(self, index):
        document = self.documents[index]
        label = self.doc_labels[index]
        return {
            'embeddings': document,
            'labels': label
        }
        


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, N, K, shuffle = True):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - num_classes: number of random classes for each iteration
        - n_support: number of samples for each iteration for each class (support)
        - n_query: number of samples for each iteration for each class (query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.N = N
        self.n_support = K
        self.n_query = K
        self.shuffle = shuffle
        self.num_samples = self.n_support + self.n_query #(K+K)
        self.iterations = len(self.labels) // (self.num_samples)

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1
    
    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.num_samples
        cpi = self.N

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            if self.shuffle:
                batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
    
    
    
class FewShotBatchSampler:
    def __init__(self, dataset_targets, N_way, K_shot, include_query=False, shuffle=True, shuffle_once=False):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in
                           the beginning, but kept constant across iterations
                           (for validation)
        """
        super().__init__()
        self.dataset_targets = dataset_targets
        self.N_way = N_way
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2
        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch

        # Organize examples by class
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_shot

        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batches_per_class.values()) // self.N_way
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [
                i + p * self.num_classes for i, c in enumerate(self.classes) for p in range(self.batches_per_class[c])
            ]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        random.shuffle(self.class_list)

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = collections.defaultdict(int)
        for it in range(self.iterations):
            class_batch = self.class_list[it * self.N_way : (it + 1) * self.N_way]  # Select N classes for the batch
            index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                index_batch.extend(self.indices_per_class[c][start_index[c] : start_index[c] + self.K_shot])
                start_index[c] += self.K_shot
            if self.include_query:  # If we return support+query set, sort them so that they are easy to split
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations