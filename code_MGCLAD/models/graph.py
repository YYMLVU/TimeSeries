import torch
from torch.nn import Sequential as Seq, Linear, ReLU, Parameter, Embedding
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.nn.functional as F


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index: [2, edge_num]
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i * node_num

    return batch_edge_index.long()

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


class DynamicGraphEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, seq_len, embed_dim=64, num_levels=1, topk=20, device=torch.device('cuda:0')):
        super(DynamicGraphEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.device = device
        self.num_levels = num_levels
        self.topk = topk
        self.embed_dim = embed_dim
        
        # Node embeddings to learn dynamic graph structure
        self.embedding = Embedding(num_nodes, embed_dim)
        # Initialize embeddings with Kaiming initialization
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        
        self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len)
        
        # Dynamic edge construction will happen during forward pass

    def forward(self, x):
        # x: >> (bsz, seq_len, num_nodes)
        batch_size = x.size(0)
        
        # Generate dynamic graph edges based on node embeddings
        # Get node embeddings
        all_embeddings = self.embedding(torch.arange(self.num_nodes).to(self.device))
        
        # Calculate cosine similarity between node embeddings
        weights = all_embeddings.view(self.num_nodes, -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        
        # Get top-k similar nodes for each node
        topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]
        
        # Create edge indices from topk similar nodes
        gated_i = torch.arange(0, self.num_nodes).T.unsqueeze(1).repeat(1, self.topk).flatten().to(self.device).unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        dynamic_edge_index = torch.cat((gated_j, gated_i), dim=0)
        
        # Create batch edge indices
        batch_edge_index = get_batch_edge_index(dynamic_edge_index, batch_size, self.num_nodes).to(self.device)
        
        # Process through GCN layers
        x = x.permute(2, 0, 1)  # >> (num_nodes, bsz, seq_len)
        
        # Repeat embeddings for batch dimension
        all_batch_embeddings = all_embeddings.repeat(batch_size, 1)
        
        for i in range(self.num_levels):
            x = self.gc_module(x, batch_edge_index)  # >> (num_nodes, bsz, seq_len)

        x = x.permute(1, 2, 0)  # >> (bsz, seq_len, num_nodes)
        return x

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


# class AdaGATConv(nn.Module):
#     def __init__(self, in_channels, out_channels, heads=2):
#         super(AdaGATConv, self).__init__()
#         self.gat = GATConv(in_channels, out_channels, heads=heads, concat=False).to('cuda:0')
    
#     def forward(self, x, edge_index):
#         return self.gat(x, edge_index)
    
# class GATEmbedding(nn.Module):
#     def __init__(self, num_nodes, seq_len, num_levels=1, heads=2):
#         super(GATEmbedding, self).__init__()
#         self.num_node = num_nodes
#         self.num_levels = num_levels
#         self.seq_len = seq_len
#         self.heads = heads

#         self.gat_module = AdaGATConv(seq_len, seq_len, heads=heads).to('cuda:0')

#         source_nodes, target_nodes = [], []
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 source_nodes.append(j)
#                 target_nodes.append(i)
#         self.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to('cuda:0')

#     def forward(self, x):
#         x = x.permute(0, 2, 1) # >> (bsz, n_features, window_size)
#         batch_size, num_nodes, seq_len = x.size()
#         out = []
#         for i in range(batch_size):
#             for _ in range(self.num_levels):
#                 y = self.gat_module(x[i], self.edge_index)
#                 out.append(y)
#         re = torch.stack(out, dim=0)
#         return re.permute(0, 2, 1)
            