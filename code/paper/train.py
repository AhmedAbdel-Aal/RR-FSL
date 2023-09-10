from allennlp.common.util import pad_sequence_to_length
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
import functools
import operator
from sklearn.metrics import classification_report

def batch_to_tensor(b):
    # convert to dictionary of tensors and pad the tensors
    max_sentence_len = 128
    result = {}
    for k, v in b.items():

        if k in ["input_ids", "attention_mask"]:
            # determine the max sentence len in the batch
            max_sentence_len = -1
            for sentence in v:
                sentence = torch.cat(sentence)
                max_sentence_len = max(len(sentence), max_sentence_len)
            # pad the sentences to max sentence len
            for i, sentence in enumerate(v):
                v[i] = pad_sequence_to_length(sentence, desired_length=max_sentence_len)
        
        if k != 'doc_name' and k != 'label_ids':
            result[k] = torch.tensor(v).unsqueeze(0).clone().detach()
        elif k == 'label_ids':
            result[k] = torch.tensor(v).clone().detach()
        else:
            result[k] = v
    return result



from sklearn.metrics import f1_score
import numpy as np

def training_step(model, optimizer, scheduler, data_loader, device, crf=False):
    model.train()  # Set the model to train mode
    train_loss = {'cls': 0}
    train_correct = 0
    train_total = 0

    all_labels = []
    all_predicted = []

    for batch_idx, batch in enumerate(data_loader):
        batch = batch_to_tensor(batch)
        
        # Handle an empty batch --> error in data preparation
        if batch["input_ids"].shape[1] == 0:
            print("Skipping an empty batch.")
            continue

        optimizer.zero_grad()  # Zero out gradients

        for key, tensor in batch.items():
            batch[key] = tensor.to(device)
        
        labels = batch['label_ids']

        # Forward pass
        outputs, embeddings = model(batch, labels, get_embeddings=True)

        # Calculate loss
        classification_loss = outputs['loss']

        train_loss['cls'] += classification_loss.item()

        if batch_idx % 10 == 0:
            print(f'After {batch_idx} steps: classification_loss {classification_loss.item()}')

        # Backward pass and optimization
        classification_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()

    # Calculate epoch statistics
    train_loss['cls'] = train_loss['cls'] / len(data_loader)

    return train_loss



from sklearn.metrics import f1_score
import functools
import operator
import torch

def validation_step(model, data_loader, device):
    model.eval()
    dev_loss = 0.0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch_to_tensor(batch)
            
            if batch["input_ids"].shape[1] == 0:
                print("Skipping an empty batch.")
                continue

            for key, tensor in batch.items():
                batch[key] = tensor.to(device)
            
            labels = batch['label_ids']
            outputs = model(batch)

            logits = outputs['logits'].squeeze()
            dev_loss += F.cross_entropy(logits, labels.squeeze()).item()
            
            predicted_label = outputs['predicted_label']

            # Debugging: Print shapes
            #print(f"Labels shape: {labels.shape}, Predicted shape: {predicted_label.shape}")

            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted_label.cpu().numpy())

    #all_predicted = functools.reduce(operator.iconcat, all_predicted, [])
    all_labels = functools.reduce(operator.iconcat, all_labels, [])
    
    #print(f"Length of all_labels: {len(all_labels)}, Length of all_predicted: {len(all_predicted)}")

    f1 = f1_score(all_labels, all_predicted, average='macro')
    dev_loss /= len(data_loader)
    
    return f1, dev_loss




from sklearn.metrics import classification_report

from sklearn.metrics import classification_report
import functools
import operator
import torch

def testing_step(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move model to device

    predictions = []
    true_labels = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch_to_tensor(batch)
            
            for key, tensor in batch.items():
                batch[key] = tensor.to(device)
            
            labels = batch['label_ids']
            
            # Forward pass
            outputs = model(batch)
            logits = outputs['logits'].squeeze()
            predicted_labels = outputs['predicted_label']

            # Store predictions and true labels
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            predicted_labels = predicted_labels.to('cpu')
            labels = labels.to('cpu')
            correct = (predicted_labels == labels).sum().item()
            test_correct += correct
            test_total += labels.shape[0]

    # Flatten lists
    #predictions = functools.reduce(operator.iconcat, predictions, [])
    true_labels = functools.reduce(operator.iconcat, true_labels, [])

    # Calculate statistics
    test_loss /= len(data_loader)
    test_accuracy = test_correct / test_total

    print(f"Testing Loss: {test_loss:.4f} - Testing Accuracy: {test_accuracy:.4f}")
    print(classification_report(true_labels, predictions))

    return test_loss, test_accuracy, predictions, true_labels
