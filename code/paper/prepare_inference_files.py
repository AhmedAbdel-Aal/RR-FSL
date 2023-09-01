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
from sklearn.metrics import f1_score
import copy



from utils import batch_to_tensor


def create_datastore(model, dataloader, device):
    outputs = []  # Initialize an empty list to store the iteration outputs
    labels = []
    model.to(device)
    model.eval()
    for batch in dataloader:
    with torch.no_grad():
        batch = batch_to_tensor(batch)
        for key, tensor in batch.items():
            batch[key] = tensor.to(device)
        label = batch['label_ids']

        # Forward pass
        _,output = model(batch, label, get_embeddings = True)
        outputs.append(output)
        labels.append(label)
        
    datastore_keys = torch.cat(outputs, dim=1) # torch.Size([1, M, 768])
    datastore_values = torch.cat(labels, dim=1) # torch.Size([1, M])
  
    return datastore_keys, datastore_values


def prepare_test_data(model, dataloader, device):
    embeddings = []  # Initialize an empty list to store the iteration outputs
    labels = []
    predicted_labels = []
    logits = []
    model.to(device)
    model.eval()
    for batch in dataloader:
    with torch.no_grad():
        batch = batch_to_tensor(batch)
        for key, tensor in batch.items():
            batch[key] = tensor.to(device)
        label = batch['label_ids']

        # Forward pass
        logit, embedding = model(batch, labels=label, get_embeddings = True)

        # Apply argmax to get the predicted labels
        tlogit = logit.transpose(1, 2)
        predicted_label = torch.argmax(tlogit, dim=1)

        logits.append(logit)
        embeddings.append(embedding)
        labels.append(label)
        predicted_labels.append(predicted_label)
        
    embeddings = torch.cat(embeddings, dim=1)
    labels = torch.cat(labels, dim=1)
    logits = torch.cat(logits, dim=1)
    predicted_labels = torch.cat(predicted_labels, dim=1)

    return embeddings, labels, logits, predicted_labels






def main():
    from data import SequenceClassificationDataset
    root = '/content/'
    train_dataset = SequenceClassificationDataset(Path('/content/drive/MyDrive/OS/train_scibert_IT.json'))
    dev_dataset = SequenceClassificationDataset(Path(root, 'dev_scibert_IT.json'))
    test_dataset = SequenceClassificationDataset(Path(root, 'test_scibert_IT.json'))
    
    batch_size=1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    from utils import load_checkpoint
    model_path = '/content/drive/MyDrive/OS/baseline_malik_it.pth'
    model = load_checkpoint(model_path, mconfig, device)
    
    datastore_keys, datastore_values = create_datastore(model, train_dataloader, device)    
    save_npy(datastore_keys,'/content/v0_it_datastore_keys.npy')
    save_npy(datastore_values,'/content/v0_it_datastore_values.npy')

    
    embeddings, labels, logits, predicted_labels = prepare_test_data(model, test_dataloader, device)    
    save_npy(embeddings,'/content/v0_it_test_embeddings.npy')
    save_npy(labels,'/content/v0_it_test_labels.npy')
    save_npy(logits,'/content/v0_it_test_logits.npy')
    save_npy(predicted_labels,'/content/v0_it_test_predicted_labels.npy')
    
    
    embeddings, labels, logits, predicted_labels = prepare_test_data(model, valid_dataloader, device)
    save_npy(embeddings,'/content/v0_it_val_embeddings.npy')
    save_npy(labels,'/content/v0_it_val_labels.npy')
    save_npy(logits,'/content/v0_it_val_logits.npy')
    save_npy(predicted_labels,'/content/v0_it_val_predicted_labels.npy')
    