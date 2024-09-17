from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random

    
class WaterLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, num_layers=1):
        super(WaterLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        lstm_outputs, _ = self.lstm(x)
        predictions = self.sigmoid(self.fc(self.dropout(lstm_outputs)))
        return predictions

class BatchWaterLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0,num_layers=1):
        super(BatchWaterLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Batch Normalization for LSTM outputs
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        # Linear layer that maps from hidden state space to output space
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # sigmoid to push outputs to 0 and 1
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        
        out, _ = self.lstm(x)
        
        
        # Transform the output to be suitable for batch normalization
        out = out.contiguous().view(-1, self.hidden_dim)
        
        # Batch normalization
        out = self.bn(out)
        
        # Reshape back to the original shape
        out = out.view(x.size(0), -1, self.hidden_dim)
        
        # Decode the hidden state of the last time step        
        predictions = self.sigmoid(self.fc(out))
        
        return predictions


class WaterLSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, num_layers=1):
        super(WaterLSTM2, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim[0],
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.tanh = nn.Tanh()
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim[0],
            hidden_size=hidden_dim[1],
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim[1], output_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        lstm_outputs, _ = self.lstm1(x)
        lstm_outputs = self.tanh(lstm_outputs)
        lstm_outputs, _ = self.lstm2(lstm_outputs)
                
        predictions = self.sigmoid(self.fc(self.tanh(self.dropout(lstm_outputs))))
        return predictions

class AdvancedWaterLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5, 
                 num_fc_layers=1, intermediate_fc_sizes=[]):
        super(WaterLSTM, self).__init__()
        assert num_fc_layers - 1 == len(intermediate_fc_sizes)
        intermediate_fc_sizes.insert(0, hidden_dim)
        intermediate_fc_sizes.append(output_dim)
        
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.layers.append(self.lstm)
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        
        self.fc_layers = []
        for i in range(num_fc_layers):
            i_dim = intermediate_fc_sizes[i]
            o_dim = intermediate_fc_sizes[i+1]
            self.layers.append(nn.Linear(i_dim, o_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        self.signmoid = nn.Sigmoid()


    def forward(self, x):
        lstm_outputs, _ = self.lstm(x)
        predictions = self.fc(lstm_outputs)
        predictions = self.sigmoid(predictions)
        return predictions