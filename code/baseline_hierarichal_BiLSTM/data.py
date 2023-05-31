"""
A custome dataset class that applies bucketing for sentences representations in documents 

"""

import torch
from torch.utils.data import Dataset
import pickle


class Batch:
    def __init__(self, batch_dict):
        self.batch_dict = batch_dict
        
        
    def get_documents(self):
        return self.batch_dict['documents']
        
    def get_embeddings(self):
        return self.batch_dict['documents']['embeddings']
    
    def get_labels(self):
        return self.batch_dict['documents']['labels']
    
    def get_num_sentences(self):
        return self.batch_dict['num_sentences']

    def add_document(self, embeddings, labels):
        self.batch_dict['documents'].append({'embeddings': embeddings, 'labels': labels})
        self.batch_dict['num_sentences'] += len(labels)

    def borrow_document(self, other_batch):
        next_document = other_batch.get_document()
        print(type(next_document['embeddings']), len(next_document['embeddings']))
        self.add_document(next_document['embeddings'], next_document['labels'])
        other_batch.remove_document()

    def get_document(self):
        return self.batch_dict['documents'][0]

    def remove_document(self):
        print(type(self.batch_dict['documents']))
        self.batch_dict['documents'].pop(0)
        self.batch_dict['num_sentences'] -= len(self.get_document()['labels'])

    def pad_with_last_embedding(self, batch_size):
        last_embedding = self.get_document()['embeddings'][-1:]
        last_label = self.get_document()['labels'][-1:]

            
        i = 0
        #print(self.get_num_sentences(), batch_size)
        while self.get_num_sentences() < batch_size:
         #   print(i)
            i+=1
            self.add_document(last_embedding, last_label)
        #print('-----')


class CustomDataset(Dataset):
    def __init__(self, data_path, batch_size):
        self.data = self.load_data(data_path)
        self.batch_size = batch_size
        self.batches = self.bucket_and_batch_data()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]

    def load_data(self, data_path):
        # Load data from pkl file
        # Returns dictionary in the form of {document_name: ([embedding_list], [labels_list])}
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def bucket_and_batch_data(self):
        buckets = {}  # Dictionary to store buckets
        batches = []  # List to store batches

        for document_name, values in self.data.items():
            embeddings, labels = values['embeddings'], values['labels']
            document_length = len(embeddings)
            if document_length not in buckets:
                # Create a new bucket for unique document length
                buckets[document_length] = []

            bucket = buckets[document_length]  # a bucket is a list of batches
            num_batches_in_bucket = len(bucket)
            if num_batches_in_bucket == 0 or bucket[-1].get_num_sentences() >= self.batch_size:
                # Create a new batch if the bucket is empty or the current batch is full or overfilled
                batch_dict = {'documents': [], 'num_sentences': 0}
                batch = Batch(batch_dict)
                bucket.append(batch)
                batches.append(batch)
            else:
                # use the last batch in the bucket
                batch = bucket[-1]

            # Add the document to the batch
            print(embeddings[:5])
            batch.add_document(embeddings, labels)

        # Flatten all the batches into a single list
        all_batches = [batch for bucket in buckets.values() for batch in bucket]

        # Sort the flattened batches based on the length of the first document in each batch
        all_batches.sort(key=lambda x: x.get_num_sentences())
        return all_batches
        
        # Split overfilled batches
        all_batches_new = self.split_overfilled_batches(all_batches)
        #return all_batches_new
        
        # Borrow documents for the underfilled batches
        all_batches_new = self.borrow_documents(all_batches_new)
        #return all_batches_new
        
        # Pad the underfilled batches with the last embedding from the last document in the batch
        #all_batches_new = self.pad_underfilled_batches(all_batches_new)
        #return all_batches_new

        # Merge documents in the batch into one big document
        merged_batches = self.merge_documents(all_batches_new)

        return merged_batches

    def split_overfilled_batches(self, all_batches):
        all_batches_new = []
        for batch in all_batches:
            num_sentences = batch.get_num_sentences()
            if num_sentences > self.batch_size:
                num_batches = (num_sentences + self.batch_size - 1) // self.batch_size
                start_idx = 0
                for _ in range(num_batches):
                    end_idx = start_idx + self.batch_size
                    smaller_batch = Batch({
                        'documents': [{
                            'embeddings': batch.get_documents()[-1]['embeddings'][start_idx:end_idx],
                            'labels': batch.get_documents()[-1]['labels'][start_idx:end_idx]
                        }],
                        'num_sentences': len(batch.get_documents()[-1]['embeddings'][start_idx:end_idx])
                    })
                    all_batches_new.append(smaller_batch)
                    start_idx = end_idx
            else:
                all_batches_new.append(batch)

        return all_batches_new

    def borrow_documents(self, all_batches):
        for batch_idx, batch in enumerate(all_batches):
            while batch.get_num_sentences() < self.batch_size and batch_idx < len(all_batches) - 1:
                next_batch = all_batches[batch_idx + 1]
                if batch.get_num_sentences() + len(next_batch.get_document()['embeddings']) <= self.batch_size:
                    batch.borrow_document(next_batch)
                    if len(next_batch.get_documents()) == 0:
                        all_batches.pop(batch_idx + 1)
                else:
                    break
        return all_batches

    def pad_underfilled_batches(self, all_batches):
        for batch in all_batches:
            if batch.get_num_sentences() < self.batch_size:
                batch.pad_with_last_embedding(self.batch_size)
        return all_batches

    def merge_documents(self, all_batches):
        merged_batches = []
        for batch in all_batches:
            merged_batch = {'embeddings': [], 'labels': [], 'num_sentences': 0}
            for doc in batch.get_documents():
                merged_batch['embeddings'].extend(doc['embeddings'])
                merged_batch['labels'].extend(doc['labels'])
                merged_batch['num_sentences'] += len(doc['labels'])
            merged_batches.append(merged_batch)

        return merged_batches
