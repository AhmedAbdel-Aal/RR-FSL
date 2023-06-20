import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(input_size, hidden_size//2, batch_first=True, bidirectional=True)    
    def forward(self, x):
        # Apply the bidirectional LSTM
        outputs, _ = self.bilstm(x)
        return outputs