import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable
from model import count_parameters
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=(1,1), padding=(0,0), groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Attention(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, n_in=128, n_out=20,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(n_in, dims[0], kernel_size=(1,1), stride=(1,1)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(1): #3
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=(1,1), stride=(1,1)),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(1): #4
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        #self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.norm = nn.LayerNorm(dims[0], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], n_out)
        self.head = nn.Linear(dims[0], n_out)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(1):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        #x = x[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        epsilon = 1e-7
        x = torch.clamp(x, epsilon, 1. - epsilon)
        x = x / torch.sum(x)
        x = F.hardtanh(x, 0., 1.)  #I used here because I was having error "values has to be between 0 and 1"
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x






class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, emb_layers, hidden_units, drop_rate):
        super(EmbeddingLayers, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        self.conv1x1 = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        for i in range(emb_layers):
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
        """input: (samples_num, time_steps, freq_bins)
        """

        drop_rate = self.drop_rate

        # (samples_num, freq_bins, time_steps)
        x = input.transpose(1, 2)
    # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)
        x = x[:, :, :, None].contiguous()

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
            return out

        else:
            return all_outs

        return x



class trainShallowConvNext(nn.Module):

    def __init__(self, freq_bins, classes_num, emb_layers, hidden_units, drop_rate):

        super(trainShallowConvNext, self).__init__()

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
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, classes_num, time_steps, 1)
        output = self.attention(b1)

        return output
