import os
import argparse
import random
import time
import datetime
import json
import logging
import pandas as pd
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from models import get_model
from data import Tokenizer, RhetoricalRoleDataset
from utils import flat_accuracy, load_config, plot_training_numbers, save_model

def train(model, device, train_loader, valid_loader, train_dataset, config):
    # Access the configuration values
    lr = config['training']['learning_rate']
    max_grad_norm = config['training']['max_grad_norm']
    epochs = config['training']['epochs']
    num_total_steps = len(train_dataset)*epochs
    num_warmup_steps = config['training']['num_warmup_steps']
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
    # intialize optimizers and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)
    # set seed
    seed_val = config['training']['seed']
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)



    # Define the number of training and validation steps
    num_train_steps = len(train_loader)
    num_valid_steps = len(valid_loader)

    # Initialize lists to store losses
    train_epoch_losses = []
    train_epoch_acc = []

    valid_epoch_losses = []
    valid_epoch_acc = []


    # Training loop
    for epoch in range(epochs):
        logging.info('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        logging.info('Training...')

        model.train()  # Set the model to train mode
        train_loss = 0.0
        train_accuracy = 0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # get batch loss
            loss = outputs[0]
            train_loss += loss.item()
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # get batch acc
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()  
            batch_train_accuracy = flat_accuracy(logits, label_ids)
            train_accuracy += batch_train_accuracy          

            # Print the training loss every 40 steps
            if (step + 1) % 40 == 0:
                train_step_loss = train_loss / (step+1)
                logging.info(f"Epoch {epoch+1}/{epochs} - Step {step+1}/{num_train_steps} - Training Loss: {train_step_loss:.4f}")

        # train epoch numbers: (loss , accuracy)
        train_loss /= num_train_steps
        train_accuracy /= num_train_steps
        # storing valid epoch numbers for plots
        train_epoch_losses.append(train_loss)
        train_epoch_acc.append(train_accuracy)
        # Calculate the training epoch time
        epoch_time = time.time() - start_time
        # print epoch numbers
        logging.info(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Epoch Time: {epoch_time:.2f} seconds")

        # ------------------------------------------------------------------------------------

        # Validation loop
        logging.info("Running Validation...")
        model.eval()  # Set the model to evaluation mode
        valid_loss = 0.0
        nb_eval_examples = 0
        valid_accuracy = 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # get batch loss
                loss = outputs[0]
                valid_loss += loss.item()
                # get batch acc
                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()  
                batch_valid_accuracy = flat_accuracy(logits, label_ids)
                valid_accuracy += batch_valid_accuracy          

        # valid epoch numbers: (loss , accuracy)
        valid_loss /= num_valid_steps
        valid_accuracy /= num_valid_steps
        # storing valid epoch numbers for plots
        valid_epoch_losses.append(valid_loss)
        valid_epoch_acc.append(valid_accuracy)
        # Print the validation loss for each epoch
        logging.info(f"Epoch {epoch+1}/{epochs} - Validation Loss: {valid_loss:.4f} - Validation Accuracy: {valid_accuracy:.4f}")

    # save the model
    save(model, optimizer, config['output']['save_model_path'])

    # plot the numbers
    figure_save_path = Path(config['output']['plots_path'], config['model']['save_name'], 'loss.png')
    loss_data_tuple = [(train_epoch_losses, 'Training Loss'),(valid_epoch_losses, 'Validation Loss')]
    plot_training_numbers(loss_data_tuple, 'Epoch loss', 'Epoch', 'Loss', figure_save_path)
    
    figure_save_path = Path(config['output']['plots_path'], config['model']['save_name'], 'acc.png')
    loss_data_tuple = [(train_epoch_acc, 'Training Acc'),(valid_epoch_acc, 'Validation Acc')]
    plot_training_numbers(acc_data_tuple, 'Epoch Acc', 'Epoch', 'Acc', figure_save_path)

    
def main(config):
        logging_path = config['output']['log_path']
        logging.basicConfig(filename=logging_path, level=logging.INFO)
        logging.info('============================================')
        logging.info(f'START OF NEW TRAINING AT {time.strftime("%Y-%m-%d %H:%M:%S")}')

        # Define the model name for the tokenizer
        model_name = config['model']['name']
        model_save_name = config['model']['save_name']
        model = get_model(model_name, model_save_name, config['data']['num_label'])
        logging.info(f"loaded model: {model_name}")

        # Create an instance of the tokenizer
        tokenizer = Tokenizer(config['model']['name'])
        logging.info(f"loaded tokenizer: {model_name}")
        
        # assign device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"device used: {device}")
        
        # Create instances of the dataset and dataloader for (train, valid)
        batch_size = config['training']['batch_size']
        mapper = config['data']['mapper']
        
        train_data_path = Path(config['data']['root_path'], config['data']['train_data_path'])
        train_dataset = RhetoricalRoleDataset(train_data_path, tokenizer, mapper=mapper)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_data_path = Path(config['data']['root_path'], config['data']['valid_data_path'])
        valid_dataset = RhetoricalRoleDataset(valid_data_path, tokenizer,mapper=mapper)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        
        logging.info(f"loaded datasets and data loaders")
        # train the model
        train(model, device, train_loader, valid_loader, train_dataset, config)

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./baseline_CL.json", type=str)
    args = parser.parse_args()    

    # Load the config file
    config = load_config(args.config_path)

    # Call the train function
    main(config)        

    
 # python3 train.py --config_path ./baseline_CL.json