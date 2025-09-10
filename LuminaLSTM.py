import torch 
import torch.nn as nn
import torch.nn.functional as F

class DrowsinessLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=1, dropout_prob=0.3):
        super(DrowsinessLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob) 
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  

    def forward(self, out):
        _, (hn, _) = self.lstm(out)
        hn = hn[-1]  
        out = self.dropout(hn)  
        out = F.relu(self.fc1(out))
        out = self.dropout(out)  
        out = self.fc2(out)  
        return out