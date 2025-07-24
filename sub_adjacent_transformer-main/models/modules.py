import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back
    
class SmoothLayer(nn.Module):
    def __init__(self, num_features, kernel_size):
        """
        num_features: 特征维度（对应输入的 num_features）
        p_l: 卷积核大小（分块大小，对应公式里的 p_l）
        """
        super(SmoothLayer, self).__init__()
        # 一维卷积：输入通道=num_features，输出通道=num_features（保持特征数不变）
        # 卷积核大小=p_l，padding 设为 (p_l - 1) // 2 可保持输出长度和输入一致（window_size 不变）
        self.conv1d = nn.Conv1d(
            in_channels=num_features,  
            out_channels=num_features, 
            kernel_size=kernel_size,           
            padding=(kernel_size - 1) // 2,    
            bias=True  # 是否加偏置 b，对应公式里的 +b
        )

    def forward(self, x):
        """
        x: 输入张量，维度 [batch_size, window_size, num_features]
        """
        # 1. 转置维度：[batch_size, window_size, num_features] -> [batch_size, num_features, window_size]
        x_transposed = x.transpose(1, 2)  
        
        # 2. 一维卷积：在 window_size 维度做平滑
        x_conv = self.conv1d(x_transposed)  
        
        # 3. 转回原维度：[batch_size, num_features, window_size] -> [batch_size, window_size, num_features]
        x_smooth = x_conv.transpose(1, 2)  
        
        return x_smooth


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):   # x: (b, n, 3k)
        out, h = self.gru(x)   # (b, n, hid_dim), (n_layers, b, hid_dim)
        #out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer 
        h = h[-1, :, :]
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):  # x=(b,n,k)
        # x will be last hidden state of the GRU layer
        decoder_out = self.decoder(x)  # (b, n, hid_dim)
        out = self.fc(decoder_out)
        return out
