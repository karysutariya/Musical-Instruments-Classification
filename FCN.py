# from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
from torch.nn import init
from utils import count_parameters

# class Mlp(MLPClassifier):
#     def __init__(self, freq_bins, classes_num, emb_layers, hidden_units, drop_rate,lr,batch_size):
#         super.__init__(hidden_layer_sizes=(hidden_units,276,552,128,classes_num), activation='identity',
#                          solver='adam', batch_size=batch_size, learning_rate_init=lr,random_state=1,
#                          learning_rate='constant',  max_iter=300,)


class Fcn(nn.Module):
    def __init__(self, hidden_units, drop_rate, classes_num):
        super(Fcn, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(),
        )

        self.classify = nn.Linear(hidden_units, classes_num)
        self.param_count = count_parameters(self)
        print(self.param_count)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, X):
        out = self.embed(X)
        out = (out+X).mean(1)
        out = torch.sigmoid(self.classify(out))
        return out
