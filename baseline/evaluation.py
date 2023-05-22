import time
import argparse
import logging
import numpy as np
import torch

from utils import load_config
from models import get_model
from torch.utils.data import Dataset, DataLoader
from data import Tokenizer, RhetoricalRoleDataset
from sklearn.metrics import classification_report


def get_loader(data_path, tokenizer, mapper, batch_size):
    dataset = RhetoricalRoleDataset(data_path, tokenizer, mapper=mapper)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logging.info('loaded dataset and dataloader')
    return loader

def predict(model, data_loader, device):
    logging.info('start predicting on loaded data')
    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []


    # Predict 
    for (step, batch) in enumerate(data_loader):
      # Add batch to CPU
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)


      # Telling the model not to compute or store gradients, saving memory and 
      # speeding up prediction
      with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

      logits = outputs[1]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = labels.to('cpu').numpy()

      # Store predictions and true labels
      predictions.append(logits)
      true_labels.append(label_ids)
    return predictions, true_labels    
    

def report(predictions, true_labels):
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_labels.flatten()
    logging.info(classification_report(labels_flat, pred_flat))
    

def evaluation(model, data_path, device, tokenizer, mapper, batch_size):
    loader = get_loader(data_path, tokenizer, mapper, batch_size)
    predictions, true_labels = predict(model, loader, device)
    report(predictions, true_labels)

    
def main(config):
    # setup logging
    logging_path = config['eval']['save_results_path']
    logging.basicConfig(filename=logging_path, level=logging.INFO)
    logging.info('============================================')
    logging.info(f'START OF EVALUATION AT {time.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # get configs
    model_path = config['output']['save_model_path']
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    tokenizer = Tokenizer(config['model']['name'])
    mapper = config['data']['mapper']
    batch_size = config['eval']['batch_size']
    device = config['eval']['device']
    # get model
    model_name = config['model']['name']
    model_save_name = config['model']['save_name']
    model = get_model(model_name, model_save_name, config['data']['num_label'])
    
    # load model checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # evaluate
    logging.info('IN DOMAIN TEST EVALUAION')
    evaluation(model, config['eval']['in_domain_test_path'], device, tokenizer, mapper, batch_size)
    logging.info('OUT OF DOMAIN TEST EVALUAION')
    evaluation(model, config['eval']['out_domain_test_path'], device, tokenizer, mapper, batch_size)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./baseline_CL.json", type=str)
    args = parser.parse_args()    

    # Load the config file
    config = load_config(args.config_path)

    # Call the train function
    main(config)        

    
 # python3 evaluation.py --config_path ./baseline_CL.json