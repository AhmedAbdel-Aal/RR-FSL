import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

'''
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings. The feature vectors are actually context-aware sentence embeddings. These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''
class LSTM_model(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop = 0.5, device = 'cuda'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(hidden_size, num_classes)
        self.hidden = None
        self.device = device
        
    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))
    
    def forward(self, sequences):
        ## sequences: tensor[batch_size, max_seq_len, emb_dim]
        
        # initialize hidden state
        self.hidden = self.init_hidden(sequences.shape[0])
        
        # generate context-aware sentence embeddings (feature vectors)
        ## tensor[batch_size, max_seq_len, emb_dim] --> tensor[batch_size, max_seq_len, hidden_dim]
        x, self.hidden = self.lstm(sequences, self.hidden)
        x = self.dropout(x)
        
        # generate emission scores for each class at each sentence
        # tensor[batch_size, max_seq_len, hidden_dim] --> tensor[batch_size, max_seq_len, n_tags]
        x = self.head(x)
        return x