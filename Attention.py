import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import count_parameters
from utils import *

# super function returns a temporary object of the superclass that allows access to all of its methods to its child class
class Attention(nn.Module):
    def __init__(self, n_in, n_out):
        super(Attention, self).__init__()
        # super().__init__() for python3

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att,)
        init_layer(self.cla)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1), time_steps is the time it occurs
        """

        att = self.att(x)
        att = torch.sigmoid(att)

        # the initial/previous state of decoder to compute context_vector(CV)= previous_state_(i-1) * scores_i
        cla = self.cla(x)
        cla = torch.sigmoid(cla)

        # Check again... 0 seems along colums 0 e.g[ : , 0 ] means (more or less) [ first_row:last_row , column_0 ]. If you have a 2-dimensional list/matrix/array, this notation will give you all the values in column 0 (from all rows).
        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        # clamp(min=0) is exactly ReLU, but here clamping between min, and max....unlike ReLu(input, 0, inf)
        att = torch.clamp(att, epsilon, 1. - epsilon)

        # None allows operations between array of different dimmension by adding a new, empty dimension which automagically fit the size of the other array.
        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        x = F.hardtanh(x, 0., 1.)  # squashes between 0 and 1
        return x


class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, emb_layers, hidden_units, drop_rate):
        super(EmbeddingLayers, self).__init__()  # slow
        # super().__init__() for python3

        self.freq_bins = freq_bins  # freq_bins should no of instrument class
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        # creating an empty list to hold the layers
        # Holds submodules in a list, and will be visible by all module methods
        self.conv1x1 = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        # in_channels:freq_bins; Number of channels in the input; if image ->in_channels= total number of vocabulary set (instrument set), it will determine the number of vector space
        # Number of channels produced by the convolution->out_channels is the number of filter, each of size kernel_size
        # padding=(0, 0) bo padding
        for i in range(emb_layers):
            # for the first layer, in_channels = freq_bins
            in_channels = freq_bins if i == 0 else hidden_units
            conv = nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_units,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            self.conv1x1.append(conv)
            self.batchnorm.append(nn.BatchNorm2d(in_channels))

        # Append last batch-norm layer
        self.batchnorm.append(nn.BatchNorm2d(hidden_units))

        self.init_weights()

    def init_weights(self):

        for conv in self.conv1x1:
            init_layer(conv)

        for bn in self.batchnorm:
            init_bn(bn)

    def forward(self, input, return_layers=False):
        """input: (samples_num, time_steps, freq_bins), time_steps is the time it occurs
        """

        drop_rate = self.drop_rate

        # Multi-headed attention layer, each input is split into multiple heads which allows the network to simultaneously attend to different subsections of each embedding.
        # transpose to get dimensions batch_s * no_heads * sequenceLength * EmbdVectorLength
        # (samples_num, freq_bins, time_steps)
        x = input.transpose(1, 2)

        # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)
        # When you call contiguous() , it actually makes a copy of the tensor
        x = x[:, :, :, None].contiguous()
        # such that the order of its elements in memory
        # is the same as if it had been created from scratch with the same data.

        out = self.batchnorm[0](x)
        residual = x
        all_outs = [out]

        for i in range(len(self.conv1x1)):
            out = F.dropout(F.relu(self.batchnorm[i+1](self.conv1x1[i](out))),
                            p=drop_rate,
                            training=self.training)
            all_outs.append(out)
        out = out + residual
        if return_layers is False:
            # (samples_num, hidden_units, time_steps, 1)
            return out  # Final output plus original x

        else:
            return all_outs  # all outputs of each layer


class DecisionLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, emb_layers, hidden_units, drop_rate):

        super(DecisionLevelSingleAttention, self).__init__()
        # super().__init__() for python3

        self.emb = EmbeddingLayers(
            freq_bins=freq_bins,
            emb_layers=emb_layers,
            hidden_units=hidden_units,
            drop_rate=drop_rate)

        self.attention = Attention(
            n_in=hidden_units,
            n_out=classes_num)

        self.param_count = count_parameters(self)
        print(self.param_count)

    def init_weights(self):
        pass

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1), time_steps is the time it occurs
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, classes_num, time_steps, 1)
        output = self.attention(b1)

        return output
