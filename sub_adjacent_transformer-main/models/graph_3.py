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


class AdaGCNConv2(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                 add_self_loops=False, normalize=True, bias=True, embed_dim=64):
        super(AdaGCNConv2, self).__init__(aggr='add', node_dim=0)
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


class AdaGCNConv1(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                 add_self_loops=False, normalize=True, bias=True):
        super(AdaGCNConv1, self).__init__(aggr='add', node_dim=0)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias
        self.is_structural = True  # 是否计算结构系数

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # 直接初始化logits参数
        self.logits = torch.nn.Parameter(
            torch.zeros(num_nodes ** 2, 2), 
            requires_grad=True
        )
        
        self.reset_parameters()

    def _init_graph_logits_(self, x=None):
        # 计算节点特征相似度
        x_avg = x.mean(dim=1).to('cuda')  # 平均批次维度 -> (num_nodes, in_channels)
            
        # 计算余弦相似度
        x_norm = F.normalize(x_avg, p=2, dim=-1).to('cuda')  # 归一化特征 -> (num_nodes, in_channels)
        sim_matrix = torch.mm(x_norm, x_norm.transpose(0, 1)).to('cuda')  # (num_nodes, num_nodes)
        sim_matrix = (sim_matrix + 1) / 2  # 余弦相似度范围[-1,1] -> [0,1]
        sim_flat = sim_matrix.flatten().to('cuda')  # 展平为一维向量 -> (num_nodes^2,)

        # 设置logits: 第一列为保留边的概率，第二列为删除边的概率
        # logits = torch.zeros(self.num_nodes ** 2, 2, device='cuda:0')
        self.logits.data[:, 0] = sim_flat  # 第一列为相似度
        self.logits.data[:, 1] = 1 - sim_flat  # 第二列为1减去相似度
        self.logits.data = self.logits.data.to(x.device)

        # self.logits = torch.nn.Parameter(logits, requires_grad=True)
    
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        torch.nn.init.zeros_(self.logits)  # 初始化logits为零

    def compute_structural_coeff(self, edge_index, lambda_val=1.0): # 近似功能 Gpu大幅加速
        """
        完全向量化的结构系数计算 - 更高效的GPU版本
        """
        device = edge_index.device
        # 构建邻接矩阵
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=device, dtype=torch.float)
        adj[edge_index[0], edge_index[1]] = 1
        adj = (adj + adj.t()).clamp(0, 1)
        
        # 添加自连接来构建邻居mask
        neighbor_mask = adj + torch.eye(self.num_nodes, device=device)
        neighbor_mask = neighbor_mask.clamp(0, 1)
        
        # 计算所有节点对的共同邻居数量
        # common_neighbor_count[i,j] = 节点i和j的共同邻居数量
        common_neighbor_count = torch.mm(neighbor_mask, neighbor_mask.t())
        
        # 只保留有边连接且共同邻居数>1的节点对
        edge_mask = adj * (common_neighbor_count > 1)
        
        # 计算结构系数的近似版本 (基于共同邻居数量而非子图密度)
        # 这是一个简化版本，避免了复杂的子图计算
        structural_coeff = torch.zeros_like(adj)
        
        # 使用共同邻居数量作为密度的代理
        valid_pairs = edge_mask > 0
        if valid_pairs.any():
            # 归一化共同邻居数量
            max_common = common_neighbor_count.max()
            if max_common > 0:
                normalized_common = common_neighbor_count / max_common
                structural_coeff[valid_pairs] = (normalized_common[valid_pairs] * 
                                            (common_neighbor_count[valid_pairs].float() ** lambda_val))
        
        return structural_coeff
    
    def gumbel_sampling_to_edge_index(self, z):
        """
        将Gumbel-Softmax采样结果转换为edge_index
        
        参数:
        z (torch.Tensor): Gumbel-Softmax采样结果，形状为 [num_nodes^2, 2]
        
        返回:
        torch.Tensor: 边索引，形状为 [2, num_edges]
        """
        # 获取保留的边（第一列概率）
        edge_mask = z[:, 0]  # 形状: [num_nodes^2]
        
        # 重新reshape为邻接矩阵形式
        adj_matrix = edge_mask.view(self.num_nodes, self.num_nodes).to('cuda:0')
        
        # 找到所有非零边
        edge_indices = torch.nonzero(adj_matrix > 0.5, as_tuple=False).to('cuda:0')  # 阈值设为0.5
        
        if edge_indices.numel() == 0:
            # 如果没有边，创建一个自环避免空图
            edge_indices = torch.arange(self.num_nodes, device='cuda:0').unsqueeze(0).repeat(2, 1).t()
        
        # 转置得到[2, num_edges]格式
        learned_edge_index = edge_indices.t().contiguous().to('cuda:0')
        
        return learned_edge_index, adj_matrix

    def forward(self, x, edge_weight=None):
        # 延迟初始化logits（基于输入特征）
        if torch.all(self.logits == 0):
            self._init_graph_logits_(x)
        # print("self.logits.shape:", self.logits.shape)
        # print('self.logits:', self.logits)

        # Gumbel-Softmax采样
        z = torch.nn.functional.gumbel_softmax(self.logits, hard=True).to('cuda:0')
        # print('self.num_nodes:', self.num_nodes)
        # print("z.shape:", z.shape)
        # exit()

        # 将z转换为边索引
        edge_index, adj_matrix = self.gumbel_sampling_to_edge_index(z)
        # print("edge_index.shape:", edge_index.shape)
        # print("edge_index", edge_index[0, :10], edge_index[1, :10])  # 打印前10个边索引
        # print('adj_matrix:\n', adj_matrix, adj_matrix.shape)
        # exit()

        # 计算结构系数
        if self.is_structural:
            structural_coeff = self.compute_structural_coeff(edge_index=edge_index, lambda_val=1.0)
            src, dst = edge_index
            edge_weight = structural_coeff[src, dst]  # 获取对应边的结构系数
        else:
            edge_weight = None  # 如果不需要结构系数，则不使用

        # 标准化边权重
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim),
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

        return out

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view([-1] + [1] * (x_j.dim() - 1)) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
    
class AdaGCNConv3(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                 add_self_loops=False, normalize=True, bias=True):
        super(AdaGCNConv3, self).__init__(aggr='add', node_dim=0)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias
        
        self.is_structural = True  # 是否计算结构系数
        
        # 注意力机制参数
        self.attention_dim = in_channels  # 注意力维度
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

class AdaGCNConv4(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                 add_self_loops=False, normalize=True, bias=True, 
                 lambda_val=1, edge_ratio=0.3):
        super(AdaGCNConv4, self).__init__(aggr='add', node_dim=0)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias
        self.lambda_val = lambda_val
        self.edge_ratio = edge_ratio  # 可配置的边保留比例

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # 缓存机制
        self._cached_edge_index = None
        self._cached_edge_weight = None
        self._last_similarity_hash = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        # 改进1: 批量化相似度计算
        x_norm = F.normalize(x, p=2, dim=-1)
        similarity = self._compute_similarity_batch(x_norm)
        
        # 改进3: 更高效的边选择策略
        pruned_edge_index, pruned_edge_weight = self._select_edges_efficient(
            similarity)

        # 改进4: 优化的结构系数计算
        structural_coeffs = self._compute_structural_coefficients_optimized(
            pruned_edge_index, self.num_nodes)

        # 改进5: 自适应权重融合
        alpha = torch.sigmoid(torch.tensor(self.lambda_val))  # 学习权重平衡
        final_edge_weight = alpha * pruned_edge_weight + (1 - alpha) * structural_coeffs

        if self.normalize:
            pruned_edge_index, final_edge_weight = gcn_norm(
                pruned_edge_index, final_edge_weight, x.size(self.node_dim),
                self.improved, self.add_self_loops, dtype=x.dtype)

        # 改进6: 残差连接
        x_original = x
        x = torch.matmul(x, self.weight)
        out = self.propagate(pruned_edge_index, x=x, edge_weight=final_edge_weight)
        
        # 残差连接（如果维度匹配）
        if self.in_channels == self.out_channels:
            out = out + x_original

        if self.bias is not None:
            out += self.bias

        return out

    def _compute_similarity_batch(self, x_norm):
        """批量化相似度计算"""
        # 使用einsum进行更高效的批量矩阵乘法
        x_norm_permuted = x_norm.permute(1, 0, 2)  # [batch, nodes, features]
        similarity = torch.einsum('bif,bjf->bij', x_norm_permuted, x_norm_permuted)
        return similarity.mean(dim=0)  # [nodes, nodes]

    def _select_edges_efficient(self, similarity):
        """更高效的边选择策略"""
        num_edges_per_node = max(int(self.num_nodes * self.edge_ratio), 1)
        
        # 使用torch.topk的sorted=False选项提升性能
        topk_values, topk_indices = torch.topk(
            similarity, num_edges_per_node + 1, dim=-1, sorted=False)
        
        # 向量化的边mask构建
        row_indices = torch.arange(self.num_nodes, device=similarity.device)
        row_indices = row_indices.unsqueeze(1).expand(-1, num_edges_per_node + 1)
        
        # 排除自连接
        mask = topk_indices != row_indices
        
        # 确保有足够的有效邻居
        valid_counts = mask.sum(dim=1)
        min_valid = valid_counts.min().item()
        actual_edges_per_node = min(num_edges_per_node, min_valid)
        
        if actual_edges_per_node <= 0:
            actual_edges_per_node = 1
        
        # 重新构建有效邻居矩阵
        valid_neighbors = torch.zeros((self.num_nodes, actual_edges_per_node), 
                                    dtype=torch.long, device=similarity.device)
        
        for i in range(self.num_nodes):
            valid_mask = mask[i]
            valid_indices = topk_indices[i][valid_mask][:actual_edges_per_node]
            if len(valid_indices) < actual_edges_per_node:
                # 如果没有足够的邻居，用其他节点填充
                remaining = actual_edges_per_node - len(valid_indices)
                all_others = torch.arange(self.num_nodes, device=similarity.device)
                others = all_others[all_others != i][:remaining]
                valid_indices = torch.cat([valid_indices, others])
            valid_neighbors[i] = valid_indices
        
        # 构建边索引
        src_nodes = torch.arange(self.num_nodes, device=similarity.device)
        src_nodes = src_nodes.unsqueeze(1).expand(-1, actual_edges_per_node).flatten()
        dst_nodes = valid_neighbors.flatten()
        
        new_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        new_edge_weight = similarity[src_nodes, dst_nodes]
        
        return new_edge_index, new_edge_weight

    def _compute_structural_coefficients_optimized(self, edge_index, num_nodes):
        """优化的结构系数计算"""
        # 使用稀疏矩阵操作提升效率
        indices = edge_index
        values = torch.ones(edge_index.size(1), device=edge_index.device)
        adj_sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        
        # 稀疏矩阵乘法计算共同邻居
        common_neighbors_sparse = torch.sparse.mm(adj_sparse, adj_sparse.t())
        
        # 修复索引计算问题
        common_neighbors_dense = common_neighbors_sparse.to_dense()
        edge_common = common_neighbors_dense[edge_index[0], edge_index[1]]
        
        # 计算结构系数
        structural_coeffs = torch.pow(2 + edge_common.float(), self.lambda_val)
        
        # 行归一化 - 确保索引在有效范围内
        row_sum = torch.zeros(num_nodes, device=structural_coeffs.device)
        
        # 添加边界检查
        valid_indices = (edge_index[0] >= 0) & (edge_index[0] < num_nodes)
        if valid_indices.any():
            row_sum.scatter_add_(0, edge_index[0][valid_indices], structural_coeffs[valid_indices])
            norm_coeffs = structural_coeffs / (row_sum[edge_index[0]] + 1e-8)
        else:
            norm_coeffs = structural_coeffs
        
        return norm_coeffs

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1, 1) * x_j

class GraphEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, seq_len, num_levels=3, device=torch.device('cuda:0')):
        super(GraphEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.device = device
        self.num_levels = num_levels
        
        self.gc_module_1 = AdaGCNConv1(num_nodes, seq_len, seq_len) # 动态构图
        self.gc_module_2 = AdaGCNConv2(num_nodes, seq_len, seq_len, embed_dim=64) # 节点嵌入 topk
        self.gc_module_3 = AdaGCNConv3(num_nodes, seq_len, seq_len) # 注意力分数分配
        self.gc_module_4 = AdaGCNConv4(num_nodes, seq_len, seq_len, lambda_val=1, edge_ratio=0.3) # topk

        # 注意力权重层
        self.attention_weights = torch.nn.Sequential(
            torch.nn.Linear(seq_len * 4, 4),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x形状: (bsz, seq_len, num_nodes)
        v = x.permute(2, 0, 1) # (num_nodes, bsz, seq_len)
        u = x.permute(0, 2, 1) # (bsz, num_nodes, seq_len)

        # self.gc_module_1.to(self.device)
        # self.gc_module_2.to(self.device)
        # self.gc_module_3.to(self.device)
        # self.gc_module_4.to(self.device)
        
        for _ in range(self.num_levels):
            x_1 = self.gc_module_1(v).permute(1, 0, 2)
            x_2 = self.gc_module_2(u)
            x_3 = self.gc_module_3(u)
            x_4 = self.gc_module_4(v).permute(1, 0, 2)

            batch_size = x.shape[0]
            # 拼接特征用于注意力计算
            concat_features = torch.cat([x_1, x_2, x_3, x_4], dim=2)  # (bsz, num_nodes, seq_len * 4)
            concat_features = concat_features.reshape(-1, self.seq_len * 4)  # (bsz * num_nodes, seq_len * 4)

            # 计算权重
            weights = self.attention_weights(concat_features)
            weights = weights.reshape(batch_size, self.num_nodes, 4, 1)  # (bsz, num_nodes, 4, 1)

            # 重塑特征以应用注意力
            x_list = [x_1.unsqueeze(2), x_2.unsqueeze(2), x_3.unsqueeze(2), x_4.unsqueeze(2)]  # (bsz, num_nodes, 1, seq_len)
            x_stack = torch.cat(x_list, dim=2)  # (bsz, num_nodes, 4, seq_len)

            # 加权求和
            weighted_features = x_stack * weights
            fused_features = torch.sum(weighted_features, dim=2)  # (bsz, num_nodes, seq_len)

            u = fused_features
            v = fused_features.permute(1, 0, 2)
     
        out = fused_features.permute(0, 2, 1)
        return out  

if __name__ == '__main__':
    # x:[batch_size, window_size, num_feature]
    x = torch.randn(4, 10, 16).to('cuda')
    layer1 = GraphEmbedding(num_nodes=16, seq_len=10, num_levels=1, device='cuda').to('cuda')
    out = layer1(x)
    print("Output shape:", out.shape)  # 应该是 [batch_size, seq_len, num_nodes]