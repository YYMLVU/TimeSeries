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
                 add_self_loops=False, normalize=True, bias=True):
        super(AdaGCNConv, self).__init__(aggr='add', node_dim=0)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias
        
        self.is_structural = True  # 是否计算结构系数
        
        # 注意力机制参数
        self.attention_dim = in_channels
        self.threshold = 0.1  # 边存在概率阈值
        
        # 线性变换层
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        
        # 注意力机制的线性变换层
        self.query_transform = Linear(in_channels, self.attention_dim)
        self.key_transform = Linear(in_channels, self.attention_dim)
        self.attention_head = Linear(self.attention_dim * 2, 1)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        # 初始化注意力机制参数
        torch.nn.init.xavier_uniform_(self.query_transform.weight)
        torch.nn.init.xavier_uniform_(self.key_transform.weight)
        torch.nn.init.xavier_uniform_(self.attention_head.weight)

    def compute_attention_graph(self, x, batch_num):
        """基于注意力机制动态构建图"""
        device = x.device
        batch_size, num_nodes, feature_dim = x.shape
        
        # 计算query和key
        queries = self.query_transform(x)  # (batch_size, num_nodes, attention_dim)
        keys = self.key_transform(x)       # (batch_size, num_nodes, attention_dim)
        
        # 计算所有节点对之间的注意力分数
        # 扩展维度以计算所有节点对
        queries_expanded = queries.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (batch_size, num_nodes, num_nodes, attention_dim)
        keys_expanded = keys.unsqueeze(1).expand(-1, num_nodes, -1, -1)        # (batch_size, num_nodes, num_nodes, attention_dim)
        
        # 拼接query和key
        attention_input = torch.cat([queries_expanded, keys_expanded], dim=-1)  # (batch_size, num_nodes, num_nodes, attention_dim*2)
        
        # 计算注意力分数
        attention_scores = self.attention_head(attention_input).squeeze(-1)  # (batch_size, num_nodes, num_nodes)
        
        # 使用sigmoid归一化为概率
        attention_probs = torch.sigmoid(attention_scores)  # (batch_size, num_nodes, num_nodes)
        
        # 移除自连接的影响（可选）
        mask = torch.eye(num_nodes, device=device).bool()
        attention_probs = attention_probs.masked_fill(mask.unsqueeze(0), 0)
        
        # 基于概率阈值筛选边
        edge_mask = attention_probs > self.threshold
        
        # 构建边索引和边权重
        batch_edge_indices = []
        batch_edge_weights = []
        
        for batch_idx in range(batch_size):
            # 获取当前batch的边
            edge_positions = edge_mask[batch_idx].nonzero(as_tuple=False)  # (num_edges, 2)
            
            if edge_positions.size(0) > 0:
                # 转换为edge_index格式 (2, num_edges)
                edge_index = edge_positions.t() + batch_idx * num_nodes
                edge_weight = attention_probs[batch_idx][edge_mask[batch_idx]]
                
                batch_edge_indices.append(edge_index)
                batch_edge_weights.append(edge_weight)
        
        if batch_edge_indices:
            final_edge_index = torch.cat(batch_edge_indices, dim=1)
            final_edge_weights = torch.cat(batch_edge_weights, dim=0)
        else:
            # 如果没有边，创建一个空的edge_index
            final_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            final_edge_weights = torch.empty(0, device=device)
        
        return final_edge_index, final_edge_weights, attention_probs

    def compute_structural_coeff(self, edge_index, attention_probs, lambda_val=1.0):
        """基于注意力概率计算结构系数"""
        if edge_index.size(1) == 0:
            return torch.empty(0, device=edge_index.device)
        
        device = edge_index.device
        batch_size = attention_probs.size(0)
        
        # 提取源节点和目标节点（考虑批次偏移）
        src_nodes = edge_index[0] % self.num_nodes
        dst_nodes = edge_index[1] % self.num_nodes
        batch_indices = edge_index[0] // self.num_nodes
        
        # 获取对应的注意力概率作为结构系数
        structural_coeff = attention_probs[batch_indices, src_nodes, dst_nodes]
        
        # 应用lambda缩放
        structural_coeff = structural_coeff ** lambda_val
        
        return structural_coeff
    
    def forward(self, x, edge_weight=None):
        batch_num, node_num, all_feature = x.shape
        device = x.device

        # 基于当前节点特征动态计算注意力图
        edge_index, attention_weights, attention_probs = self.compute_attention_graph(x, batch_num)
        
        # 如果没有边，返回原始特征
        if edge_index.size(1) == 0:
            return x

        # 重塑输入用于消息传递
        x_flat = x.reshape(-1, all_feature)  # (batch_num * node_num, all_feature)

        # 计算结构系数
        if self.is_structural:
            edge_weight = self.compute_structural_coeff(
                edge_index=edge_index, 
                attention_probs=attention_probs,
                lambda_val=1.0
            )
        else:
            edge_weight = attention_weights

        # 标准化边权重
        if self.normalize and edge_index.size(1) > 0:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x_flat.size(0),
                self.improved, self.add_self_loops, dtype=x_flat.dtype
            )

        # 线性变换
        x_flat = torch.matmul(x_flat, self.weight)

        # 消息传递
        out = self.propagate(
            edge_index, 
            x=x_flat, 
            edge_weight=edge_weight,
            size=None, 
        )

        if self.bias is not None:
            out += self.bias

        # 重塑回原始形状
        out = out.reshape(batch_num, node_num, self.out_channels)
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
    def __init__(self, num_nodes, seq_len, num_levels=3, device=torch.device('cuda:0')):
        super(GraphEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.device = device
        self.num_levels = num_levels
        
        self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len)

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
    layer1 = GraphEmbedding(num_nodes=16, seq_len=10, num_levels=3, device='cuda').to('cuda')
    out = layer1(x)
    print("Output shape:", out.shape)  # 应该是 [batch_size, seq_len, num_nodes]