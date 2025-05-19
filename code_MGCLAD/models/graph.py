import torch
from torch.nn import Sequential as Seq, Linear, ReLU, Parameter, Embedding
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaGCNConv(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                 add_self_loops=False, normalize=True, bias=True, init_method='all'):
        super(AdaGCNConv, self).__init__(aggr='add', node_dim=0)
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
        
        # 不再需要预初始化logits
        self.logits = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None):
        # x形状: [num_nodes, batch_size, seq_len]
        # edge_index形状: [2, num_edges]
        
        # 1. 计算特征相似度作为边权重
        x_norm = F.normalize(x, p=2, dim=-1)  # 归一化特征 [num_nodes, batch_size, seq_len]

        # Reshape x_norm to [num_nodes, batch_size * seq_len] if you want cross-node similarities
        # Or better, permute dimensions to get [batch_size, num_nodes, seq_len]
        x_norm_permuted = x_norm.permute(1, 0, 2)  # [batch_size, num_nodes, seq_len]

        # Compute similarity matrix
        similarity = torch.matmul(x_norm_permuted, x_norm_permuted.transpose(-1, -2))  # [batch_size, num_nodes, num_nodes]
        # Permute back if needed to get [num_nodes, batch_size, num_nodes]
        similarity = similarity.permute(1, 0, 2)  # Now matches comment shape [num_nodes, batch_size, num_nodes]
        mean_similarity = similarity.mean(dim=1)  # [num_nodes, num_nodes]
        
        # 2. 为每个节点选择前30%的边
        num_edges_per_node = int(self.num_nodes * 0.3)  # 30%的边
        if num_edges_per_node < 1:
            num_edges_per_node = 1  # 至少保留1条边

        # 获取每个节点的topk相似邻居(不包括自己)
        topk_values, topk_indices = torch.topk(mean_similarity, num_edges_per_node+1, dim=-1)
        
        # 构建边mask
        edge_mask = torch.zeros_like(mean_similarity, dtype=torch.bool)
        for i in range(self.num_nodes):
            neighbors = topk_indices[i][topk_indices[i] != i]  # 排除自连接
            if len(neighbors) > num_edges_per_node:
                neighbors = neighbors[:num_edges_per_node]
            edge_mask[i, neighbors] = True
        
        # 3. 应用边mask到edge_index
        src, dst = edge_index
        edge_valid = edge_mask[src, dst]
        pruned_edge_index = edge_index[:, edge_valid]
        pruned_edge_weight = mean_similarity[src, dst][edge_valid] if edge_weight is None else edge_weight[edge_valid]
        
        # 4. 归一化边权重
        if self.normalize:
            pruned_edge_index, pruned_edge_weight = gcn_norm(
                pruned_edge_index, pruned_edge_weight, x.size(self.node_dim),
                self.improved, self.add_self_loops, dtype=x.dtype)
        
        # 5. 特征变换
        x = torch.matmul(x, self.weight)
        
        # 6. 消息传递(不再需要logits，因为边已经通过相似度筛选)
        out = self.propagate(pruned_edge_index, x=x, edge_weight=pruned_edge_weight)
        
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1, 1) * x_j  # 直接使用相似度作为权重

class DynamicGraphEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, seq_len, num_levels=1, device=torch.device('cuda:0')):
        super(DynamicGraphEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.device = device
        self.num_levels = num_levels
        
        # 初始化全连接边索引
        source_nodes, target_nodes = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # 排除自连接
                    source_nodes.append(j)
                    target_nodes.append(i)
        self.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long, device=self.device)
        
        self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len)

    def forward(self, x):
        # 输入x形状: (bsz, seq_len, num_nodes)
        x = x.permute(2, 0, 1)  # >> (num_nodes, bsz, seq_len)
        
        # 进行图卷积
        for i in range(self.num_levels):
            x = self.gc_module(x, self.edge_index)  # >> (num_nodes, bsz, seq_len)
        
        x = x.permute(1, 2, 0)  # >> (bsz, seq_len, num_nodes)
        return x

# class GraphEmbedding(torch.nn.Module):
#     def __init__(self, num_nodes, seq_len, num_levels=1, device=torch.device('cuda:0')):
#         super(GraphEmbedding, self).__init__()
#         self.num_nodes = num_nodes
#         self.seq_len = seq_len
#         self.device = device
#         self.num_levels = num_levels
        
#         self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len)
        
#         source_nodes, target_nodes = [], []
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 source_nodes.append(j)
#                 target_nodes.append(i)
#         self.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long, device=self.device)

#     def forward(self, x):
#         # x: >> (bsz, seq_len, num_nodes)
#         # x = x.permute(0, 2, 1) # >> (bsz, num_nodes, seq_len)
#         # x = self.gc_module(x.transpose(0, 1), self.edge_index).transpose(0, 1) # >> (bsz, num_nodes, seq_len)

#         x = x.permute(2, 0, 1) # >> (num_nodes, bsz, seq_len)
#         for i in range(self.num_levels):
#             x = self.gc_module(x, self.edge_index) # >> (num_nodes, bsz, seq_len)
#         x = x.permute(1, 2, 0)  # >> (bsz, seq_len, num_nodes)
#         return x
            
# class AdaGCNConv(MessagePassing):
#     def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
#                     add_self_loops=False, normalize=True, bias=True, init_method='all'):
#         super(AdaGCNConv, self).__init__(aggr='add', node_dim=0) #  "Max" aggregation.
#         self.num_nodes = num_nodes
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.improved = improved
#         self.add_self_loops = add_self_loops
#         self.normalize = normalize
#         self.bias = bias
#         self.init_method = init_method

#         self.weight = Parameter(torch.Tensor(in_channels, out_channels))

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
        
#         self._init_graph_logits_()

#         self.reset_parameters()

#     def _init_graph_logits_(self):
#         if self.init_method == 'all':
#             logits = .8 * torch.ones(self.num_nodes ** 2, 2)
#             logits[:, 1] = 0
#         elif self.init_method == 'random':
#             logits = 1e-3 * torch.randn(self.num_nodes ** 2, 2)
#         elif self.init_method == 'equal':
#             logits = .5 * torch.ones(self.num_nodes ** 2, 2)
#         else:
#             raise NotImplementedError('Initial Method %s is not implemented' % self.init_method)
        
#         self.register_parameter('logits', Parameter(logits, requires_grad=True))
    
#     def reset_parameters(self):
#         glorot(self.weight)
#         zeros(self.bias)
    
#     def forward(self, x, edge_index, edge_weight=None):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
#         if self.normalize:
#             edge_index, edge_weight = gcn_norm(  # yapf: disable
#                             edge_index, edge_weight, x.size(self.node_dim),
#                             self.improved, self.add_self_loops, dtype=x.dtype)

#         z = torch.nn.functional.gumbel_softmax(self.logits, hard=True)
        
#         x = torch.matmul(x, self.weight)

#         # propagate_type: (x: Tensor, edge_weight: OptTensor)
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
#                              size=None, z=z)

#         if self.bias is not None:
#             out += self.bias

#         return out

#     def message(self, x_j, edge_weight, z):
#         if edge_weight is None:
#             return x_j * z[:, 0].contiguous().view([-1] + [1] * (x_j.dim() - 1))
#         else:
#             return edge_weight.view([-1] + [1] * (x_j.dim() - 1)) * x_j * z[:, 0].contiguous().view([-1] + [1] * (x_j.dim() - 1))

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
#                                    self.out_channels)