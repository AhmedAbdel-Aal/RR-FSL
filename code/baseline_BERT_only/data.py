import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def encode(self, sentence):
        return self.tokenizer.encode_plus(sentence, padding='max_length', truncation=True, return_tensors='pt', max_length=256)

class RhetoricalRoleDataset(Dataset):
    def __init__(self, data_path, tokenizer, mapper:str = 'compressed'):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_path)
        self.sentences = self.data['sentences'].tolist()
        self.labels = self.data['labels'].tolist()
        
        # Create the label mapping dictionary
        label_mapping = {}
        # Assign unique integer labels to each unique string label
        compressed_label_mapper = {
        "Fact": 0,
        "Argument": 1, "Argument": 1,
        "RulingP": 2, "RulingL": 3, 
        "Ratio": 4,
        "None": -1,
        "Statute": 5,
        "Precedent": 6,
        "Issue": 0,"Dissent": 0
         }
        full_label_mapper = {
        "Fact": 0,
        "ArgumentPetitioner": 1,
        "ArgumentRespondent": 2,
        "RulingByPresentCourt": 3,
        "RulingByLowerCourt": 4,
        "RatioOfTheDecision": 5,
        "None": 6,
        "Statute": 7,
        "PrecedentReliedUpon": 8,
        "PrecedentNotReliedUpon": 9,
        "PrecedentOverruled": 10,
        "Issue": 11,
        "Dissent": 12
        }
        if mapper == 'compressed':
          label_mapping = compressed_label_mapper
        else:
          label_mapping = full_label_mapper
        # Convert the string labels to integer labels using the mapping
        self.integer_labels = [label_mapping[label] for label in self.labels]


    def __len__(self):
        return len(self.sentences)
    
    def num_labels(self):
        return len(set(self.labels))

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.integer_labels[idx]

        encoded_input = self.tokenizer.encode(sentence)
        input_ids = encoded_input['input_ids'].squeeze()
        attention_mask = encoded_input['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }