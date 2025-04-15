import torch
from torch.nn import Sequential as Seq, Linear, ReLU, Parameter
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.nn.functional as F


class AdaGCNConv(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                    add_self_loops=False, normalize=True, bias=True, init_method='all'):
        super(AdaGCNConv, self).__init__(aggr='add', node_dim=0) #  "Max" aggregation.
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias
        self.init_method = init_method

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_graph_logits_()

        self.reset_parameters()

    def _init_graph_logits_(self):
        if self.init_method == 'all':
            logits = .8 * torch.ones(self.num_nodes ** 2, 2)
            logits[:, 1] = 0
        elif self.init_method == 'random':
            logits = 1e-3 * torch.randn(self.num_nodes ** 2, 2)
        elif self.init_method == 'equal':
            logits = .5 * torch.ones(self.num_nodes ** 2, 2)
        else:
            raise NotImplementedError('Initial Method %s is not implemented' % self.init_method)
        
        self.register_parameter('logits', Parameter(logits, requires_grad=True))
    
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        if self.normalize:
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim),
                            self.improved, self.add_self_loops, dtype=x.dtype)

        z = torch.nn.functional.gumbel_softmax(self.logits, hard=True)
        
        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None, z=z)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_weight, z):
        if edge_weight is None:
            return x_j * z[:, 0].contiguous().view([-1] + [1] * (x_j.dim() - 1))
        else:
            return edge_weight.view([-1] + [1] * (x_j.dim() - 1)) * x_j * z[:, 0].contiguous().view([-1] + [1] * (x_j.dim() - 1))

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, seq_len, num_levels=1, device=torch.device('cuda:0')):
        super(GraphEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.device = device
        self.num_levels = num_levels
        
        self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len)
        
        source_nodes, target_nodes = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                source_nodes.append(j)
                target_nodes.append(i)
        self.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long, device=self.device)

    def forward(self, x):
        # x: >> (bsz, seq_len, num_nodes)
        # x = x.permute(0, 2, 1) # >> (bsz, num_nodes, seq_len)
        # x = self.gc_module(x.transpose(0, 1), self.edge_index).transpose(0, 1) # >> (bsz, num_nodes, seq_len)

        x = x.permute(2, 0, 1) # >> (num_nodes, bsz, seq_len)
        for i in range(self.num_levels):
            x = self.gc_module(x, self.edge_index) # >> (num_nodes, bsz, seq_len)

        x = x.permute(1, 2, 0)  # >> (bsz, seq_len, num_nodes)
        return x


class AdaGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super(AdaGATConv, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=False).to('cuda:0')
    
    def forward(self, x, edge_index):
        return self.gat(x, edge_index)
    
class GATEmbedding(nn.Module):
    def __init__(self, num_nodes, seq_len, num_levels=1, heads=2):
        super(GATEmbedding, self).__init__()
        self.num_node = num_nodes
        self.num_levels = num_levels
        self.seq_len = seq_len
        self.heads = heads

        self.gat_module = AdaGATConv(seq_len, seq_len, heads=heads).to('cuda:0')

        source_nodes, target_nodes = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                source_nodes.append(j)
                target_nodes.append(i)
        self.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to('cuda:0')

    def forward(self, x):
        x = x.permute(0, 2, 1) # >> (bsz, n_features, window_size)
        batch_size, num_nodes, seq_len = x.size()
        out = []
        for i in range(batch_size):
            for _ in range(self.num_levels):
                y = self.gat_module(x[i], self.edge_index)
                out.append(y)
        re = torch.stack(out, dim=0)
        return re.permute(0, 2, 1)
            