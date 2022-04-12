from utils import count_parameters
import torch.nn as nn
import torch

class RNN_bidirectional(nn.Module):
    def __init__(self):
        super(RNN_bidirectional, self).__init__()
        self.rnn = nn.GRU(128, 64, num_layers=3, bidirectional=True, dropout=0.5, batch_first=True)
        self.FC = nn.Linear(128, 20)
        self.param_count = count_parameters(self)
        print(self.param_count)
    def forward(self, X):
        out, _ = self.rnn(X)
        out = torch.sigmoid(self.FC(out[:,-1,:]))
        return out