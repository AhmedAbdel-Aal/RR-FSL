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
from allennlp.common.util import pad_sequence_to_length

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
        if k!='doc_name' and k!= 'label_ids':
            result[k] = torch.tensor(v).unsqueeze(0)
        elif k == 'label_ids':
            result[k] = torch.tensor(v)
        else:
            result[k] = v
    return result


def training_step(model, optimizer, scheduler, data_loader, device):
    model.train()  # Set the model to train mode
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    all_labels = []
    all_predicted = []

    for batch in data_loader:
        batch = batch_to_tensor(batch)
        for key, tensor in batch.items():
            batch[key] = tensor.to(device)
        labels = batch['label_ids']

        optimizer.zero_grad()

        # Forward pass
        output = model(batch, labels)


        #max_sequence_length = output.size(1)
        #tlengths = torch.tensor(lengths).unsqueeze(1)
        #mask = torch.arange(max_sequence_length).unsqueeze(0).to(device) < tlengths
        #masked_output = output['predicted_label']

        # Calculate loss
        #loss = criterion(masked_output, labels)
        loss = output['loss']
        train_loss += loss.item()

        # Calculate accuracy
        #predicted_labels = masked_output.argmax(dim=-1)
        #correct = (predicted_labels == labels).sum().item()
        #train_correct += correct
        #train_total += labels.shape[0]

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # save all epoch labels and predicted labels
        #all_labels.extend(labels.cpu())
        #all_predicted.extend(predicted_labels.cpu())
    scheduler.step()
    # Calculate epoch statistics
    train_loss /= len(data_loader)
    #train_accuracy = train_correct / train_total
    # calculate f1
    #f1 = f1_score(all_labels, all_predicted, average='weighted')
    return train_loss#, train_accuracy, f1


def validation_step(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    dev_loss = 0.0
    dev_correct = 0
    dev_total = 0
    all_labels = []
    all_predicted = []


    for batch in data_loader:
        with torch.no_grad():
            batch = batch_to_tensor(batch)
            for key, tensor in batch.items():
               batch[key] = tensor.to(device)
            labels = batch['label_ids']
            # Forward pass
            outputs = model(batch)
            #max_sequence_length = outputs.size(1)
            #tlengths = torch.tensor(lengths).unsqueeze(1)
            #mask = torch.arange(max_sequence_length).unsqueeze(0).to(device) < tlengths
            #masked_output = output['predicted_label']

            # Calculate loss
            #loss = criterion(masked_output, labels)
            predicted_label = outputs['predicted_label']
            #dev_loss += loss.item()

            # Calculate accuracy
            #predicted_labels = masked_output.argmax(dim=-1)
            #correct = (predicted_labels == labels).sum().item()
            #dev_correct += correct
            #dev_total += labels.shape[0]

            # save all epoch labels and predicted labels
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted_label.cpu().numpy())

    # Calculate epoch statistics
    #dev_loss /= len(data_loader)
    #dev_accuracy = dev_correct / dev_total
    # calculate f1
    all_predicted = functools.reduce(operator.iconcat, all_predicted, [])
    all_labels = functools.reduce(operator.iconcat, all_labels, [])
    print(len(all_predicted), len(all_labels))
    f1 = f1_score(all_labels, all_predicted, average='weighted')
    return f1


from sklearn.metrics import classification_report

def testing_step(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    for batch in data_loader:
        with torch.no_grad():
            
            batch = batch_to_tensor(batch)
            for key, tensor in batch.items():
               batch[key] = tensor.to(device)
            labels = batch['label_ids']
            # Forward pass
            outputs = model(batch)
            #max_sequence_length = outputs.size(1)
            #tlengths = torch.tensor(lengths).unsqueeze(1)
            #mask = torch.arange(max_sequence_length).unsqueeze(0).to(device) < tlengths
            #masked_output = outputs[mask]
            predicted_labels = outputs['predicted_label']

            # Calculate loss
            #loss = criterion(masked_output, labels)
            #test_loss += loss.item()

            # Store predictions and true labels
            #predicted_labels = masked_output.argmax(dim=-1)
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            predicted_labels.detach().cpu().numpy()
            labels = labels.to('cpu')

            correct = (predicted_labels == labels).sum().item()
            test_correct += correct
            test_total += labels.shape[0]

    predictions = functools.reduce(operator.iconcat, predictions, [])
    true_labels = functools.reduce(operator.iconcat, true_labels, [])

    test_loss /= len(data_loader)
    test_accuracy = test_correct / test_total

    print(f"Testing Loss: {test_loss:.4f} - Testing Accuracy: {test_accuracy:.4f}")
    print(classification_report(true_labels, predictions))
    return test_loss, test_accuracy, predictions, true_labels