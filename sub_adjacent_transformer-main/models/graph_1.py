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


def get_batch_edge_index(edge_index, batch_num, node_num):
    """将边索引扩展到批次维度"""
    batch_edge_index = []
    for i in range(batch_num):
        batch_edge_index.append(edge_index + i * node_num)
    return torch.cat(batch_edge_index, dim=1)


class AdaGCNConv(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                 add_self_loops=False, normalize=True, bias=True, embed_dim=64):
        super(AdaGCNConv, self).__init__(aggr='add', node_dim=0)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias

        self.topk = 0.3 * num_nodes  # 设置topk为30%的节点
        if self.topk < 1:
            self.topk = 1
        self.topk = int(self.topk)  # 确保topk为整数
        
        self.embed_dim = embed_dim
        self.is_structural = True  # 是否计算结构系数

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 节点嵌入层
        self.embedding = Embedding(num_nodes, embed_dim)
        
        # 缓存边索引
        self.cache_edge_index = None
        self.learned_graph = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def compute_topk_graph(self, batch_num, device):
        """基于节点embedding计算topk相似性图"""
        # 获取所有节点的嵌入
        all_embeddings = self.embedding(torch.arange(self.num_nodes).to(device))
        
        # 计算余弦相似度矩阵
        weights = all_embeddings.view(self.num_nodes, -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
        cos_ji_mat = cos_ji_mat / (normed_mat + 1e-8)  # 避免除零

        # 获取topk相似的节点
        topk_num = min(self.topk, self.num_nodes)
        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

        # 构建边索引
        gated_i = torch.arange(0, self.num_nodes).unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

        # 扩展到批次维度
        batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, self.num_nodes).to(device)
        
        self.learned_graph = gated_edge_index
        return batch_gated_edge_index, all_embeddings

    def compute_structural_coeff(self, edge_index, lambda_val=1.0):
        """计算结构系数"""
        device = edge_index.device
        # 构建邻接矩阵
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=device, dtype=torch.float)
        adj[edge_index[0] % self.num_nodes, edge_index[1] % self.num_nodes] = 1
        adj = (adj + adj.t()).clamp(0, 1)
        
        # 添加自连接来构建邻居mask
        neighbor_mask = adj + torch.eye(self.num_nodes, device=device)
        neighbor_mask = neighbor_mask.clamp(0, 1)
        
        # 计算所有节点对的共同邻居数量
        common_neighbor_count = torch.mm(neighbor_mask, neighbor_mask.t())
        
        # 只保留有边连接且共同邻居数>1的节点对
        edge_mask = adj * (common_neighbor_count > 1)
        
        # 计算结构系数
        structural_coeff = torch.zeros_like(adj)
        
        valid_pairs = edge_mask > 0
        if valid_pairs.any():
            max_common = common_neighbor_count.max()
            if max_common > 0:
                normalized_common = common_neighbor_count / max_common
                structural_coeff[valid_pairs] = (normalized_common[valid_pairs] * 
                                            (common_neighbor_count[valid_pairs].float() ** lambda_val))
        
        return structural_coeff
    
    def forward(self, x, edge_weight=None):
        batch_num, node_num, all_feature = x.shape
        device = x.device

        # 重塑输入
        x = x.reshape(-1, all_feature)  # (batch_num * node_num, all_feature)

        # 计算基于embedding的topk图
        edge_index, all_embeddings = self.compute_topk_graph(batch_num, device)

        # 计算结构系数
        if self.is_structural:
            structural_coeff = self.compute_structural_coeff(edge_index=edge_index, lambda_val=1.0)
            # 为批次中的每条边分配结构系数
            src_nodes = edge_index[0] % self.num_nodes
            dst_nodes = edge_index[1] % self.num_nodes
            edge_weight = structural_coeff[src_nodes, dst_nodes]
        else:
            edge_weight = None  # 如果不需要结构系数，则不使用

        # 标准化边权重
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(0),
                self.improved, self.add_self_loops, dtype=x.dtype
            )

        # 线性变换
        x = torch.matmul(x, self.weight)

        # 消息传递
        out = self.propagate(
            edge_index, 
            x=x, 
            edge_weight=edge_weight,
            size=None, 
        )

        if self.bias is not None:
            out += self.bias

        # 重塑回原始形状
        out = out.reshape(batch_num, node_num, self.out_channels)  # (batch_num, node_num, out_channels)
        return out

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view([-1] + [1] * (x_j.dim() - 1)) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GraphEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, seq_len, num_levels=1, embed_dim=64, device=torch.device('cuda:0')):
        super(GraphEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.device = device
        self.num_levels = num_levels
        
        self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len, embed_dim=embed_dim)

    def forward(self, x):
        # x形状: (bsz, seq_len, num_nodes)
        x = x.permute(0, 2, 1).contiguous()  # (bsz, num_nodes, seq_len)
        
        for _ in range(self.num_levels):
            x = self.gc_module(x)
            
        x = x.permute(0, 2, 1).contiguous()  # (bsz, seq_len, num_nodes)
        return x
    

if __name__ == '__main__':
    # x:[batch_size, window_size, num_feature]
    x = torch.randn(4, 10, 16).to('cuda')
    layer1 = GraphEmbedding(num_nodes=16, seq_len=10, num_levels=1, embed_dim=64, device='cuda').to('cuda')
    out = layer1(x)
    print("Output shape:", out.shape)  # 应该是 [batch_size, seq_len, num_nodes]