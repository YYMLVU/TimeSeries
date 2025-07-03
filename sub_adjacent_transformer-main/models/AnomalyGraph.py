import torch
import torch.nn as nn
from torch.nn import Parameter, Linear, Sequential
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, scatter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import math
import networkx as nx

# def segment_coo(src, index, out, reduce='sum'):
#     """
#     高效的COO格式分段聚合函数
#     类似torch_scatter.scatter但更轻量
    
#     Args:
#         src: 源数据 [num_edges, features]
#         index: 目标索引 [num_edges]
#         out: 输出缓冲区 [num_nodes, features]
#         reduce: 聚合方式 ('sum', 'mean', 'max')
#     """
#     # 获取唯一索引和反向索引
#     unique, inverse, counts = torch.unique(
#         index, return_inverse=True, return_counts=True)
    
#     # 根据索引对源数据进行分组
#     grouped_src = src.new_zeros(unique.size(0), src.size(1))
#     grouped_src.index_add_(0, inverse, src)
    
#     # 应用聚合操作
#     if reduce == 'sum':
#         pass  # index_add已经完成求和
#     elif reduce == 'mean':
#         grouped_src = grouped_src / counts.float().view(-1, 1)
#     elif reduce == 'max':
#         max_values = torch.zeros_like(grouped_src)
#         max_values.scatter_reduce_(0, inverse.unsqueeze(-1).expand_as(src), 
#                                  src, reduce='amax', include_self=False)
#         grouped_src = max_values
    
#     # 将结果映射回输出缓冲区
#     out[unique] = grouped_src
    
#     return out

# class GraphLayer(MessagePassing):
#     def __init__(self, in_channels, out_channels, heads=1, concat=True,
#                  negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
#         super(GraphLayer, self).__init__(aggr='add', **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout

#         self.__alpha__ = None

#         self.lin = Linear(in_channels, heads * out_channels, bias=False)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()

#         # GraphSNN特有的结构感知参数 - 修复维度问题
#         if concat:
#             self.struct_transform = Linear(heads * out_channels, heads * out_channels)
#         else:
#             self.struct_transform = Linear(out_channels, out_channels)

#         self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
#         self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
#         self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
#         self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))
#         # 在类初始化中添加
#         self.struct_gate = Parameter(torch.Tensor([0.5]))
        
#         # 结构系数的注意力参数
#         self.att_struct = Parameter(torch.Tensor(1, heads, out_channels))

#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.lin.weight)
#         glorot(self.struct_transform.weight)
#         glorot(self.att_i)
#         glorot(self.att_j)
#         glorot(self.att_struct)
        
#         zeros(self.att_em_i)
#         zeros(self.att_em_j)
#         zeros(self.bias)

#     def forward(self, x, edge_index, edge_weights=None, embedding=None, return_attention_weights=False):
#         """"""
#         if torch.is_tensor(x):
#             x = self.lin(x)
#             x = (x, x)
#         else:
#             x = (self.lin(x[0]), self.lin(x[1]))

#         # 先移除自环
#         edge_index, edge_weights = remove_self_loops(edge_index, edge_weights)
        
#         # 添加自环，并为自环设置权重
#         num_nodes = x[1].size(self.node_dim)
#         edge_index, edge_weights = add_self_loops(edge_index, edge_attr=edge_weights, 
#                                                   fill_value=1.0, num_nodes=num_nodes)

#         # GraphSNN的结构化消息传递
#         out = self.propagate(edge_index, x=x, embedding=embedding,
#                              edge_weights=edge_weights,
#                              edges=edge_index,  # 传递完整的边信息
#                              return_attention_weights=return_attention_weights)        

#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         # GraphSNN的结构变换 - 在BatchNorm之前应用
#         out = self.struct_transform(out)

#         if self.bias is not None:
#             out = out + self.bias

#         # 修复BatchNorm维度问题
#         if self.concat:
#             # 如果concat=True，需要重新reshape用于BatchNorm
#             out_for_bn = out.view(-1, self.heads, self.out_channels).mean(dim=1)
#             out_for_bn = self.bn(out_for_bn)
#             out_for_bn = self.relu(out_for_bn)
#             # 恢复原始形状
#             out = out + out_for_bn.unsqueeze(1).repeat(1, self.heads).view(-1, self.heads * self.out_channels)
#         else:
#             out = self.bn(out)
#             out = self.relu(out)
        
#         if return_attention_weights:
#             alpha, self.__alpha__ = self.__alpha__, None
#             return out, (edge_index, alpha)
#         else:
#             return out

#     def message(self, x_i, x_j, edge_index_i, size_i,
#                 embedding=None, edges=None, edge_weights=None, return_attention_weights=False):

#         x_i = x_i.view(-1, self.heads, self.out_channels)
#         x_j = x_j.view(-1, self.heads, self.out_channels)

#         # GraphSNN核心：应用结构系数到消息计算
#         if edge_weights is not None:
#             # 确保结构权重的维度与消息数量匹配
#             if edge_weights.size(0) != x_j.size(0):
#                 # 如果维度不匹配，使用默认权重
#                 struct_weight = torch.ones(x_j.size(0), device=x_j.device).view(-1, 1, 1)
#             else:
#                 struct_weight = edge_weights.view(-1, 1, 1)
                
#             # x_j_struct = x_j * struct_weight
#             x_j_struct = (1 - self.struct_gate) * x_j + self.struct_gate * (x_j * struct_weight)
#         else:
#             x_j_struct = x_j

#         # 构建注意力机制的键值对
#         if embedding is not None and edges is not None:
#             # 安全地获取embedding索引 - 修复索引获取逻辑
#             try:
#                 embedding_i = embedding[edge_index_i]
#                 # 使用edges的第二行作为目标节点索引
#                 if edges.dim() > 1 and edges.size(0) > 1:
#                     # 获取对应的目标节点embedding
#                     edge_target_idx = edges[1]
#                     if edge_target_idx.size(0) == edge_index_i.size(0):
#                         embedding_j = embedding[edge_target_idx]
#                     else:
#                         embedding_j = embedding[edge_index_i]  # fallback
#                 else:
#                     embedding_j = embedding[edge_index_i]
#             except (IndexError, RuntimeError):
#                 # 如果索引出错，使用默认处理
#                 embedding_i = embedding[edge_index_i]
#                 embedding_j = embedding[edge_index_i]
                
#             embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
#             embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

#             key_i = torch.cat((x_i, embedding_i), dim=-1)
#             key_j = torch.cat((x_j_struct, embedding_j), dim=-1)
            
#             cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
#             cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)
#         else:
#             key_i = x_i
#             key_j = x_j_struct
#             cat_att_i = self.att_i
#             cat_att_j = self.att_j

#         # GraphSNN的增强注意力计算
#         alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        
#         # 添加结构感知的注意力分量
#         if edge_weights is not None and edge_weights.size(0) == x_j.size(0):
#             struct_att = (x_j_struct * self.att_struct).sum(-1) * edge_weights.view(-1, 1)
#             alpha = alpha + struct_att

#         alpha = alpha.view(-1, self.heads, 1)
#         alpha = F.leaky_relu(alpha, self.negative_slope)
        
#         self.node_dim = 0
#         alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

#         if return_attention_weights:
#             self.__alpha__ = alpha

#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
#         return x_j_struct * alpha.view(-1, self.heads, 1)



#     def __repr__(self):
#         return '{}({}, {}, heads={})'.format(self.__class__.__name__,
#                                              self.in_channels,
#                                              self.out_channels, self.heads)

# def get_batch_edge_index(org_edge_index, batch_num, node_num):
#     # org_edge_index:(2, edge_num)
#     edge_index = org_edge_index.clone().detach()
#     edge_num = org_edge_index.shape[1]
#     batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

#     for i in range(batch_num):
#         batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

#     return batch_edge_index.long()

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         if hasattr(m, 'weight'):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out')
#         if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
#             nn.init.constant_(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         if hasattr(m, 'weight') and m.weight is not None:
#             m.weight.data.normal_(1.0, 0.02)
#         if hasattr(m, 'bias') and m.bias is not None:
#             m.bias.data.fill_(0)

# class TCN1d(nn.Module): # 多尺度卷积
#     def __init__(self, feature_num, kernel_size=3, dilation=1):
#         super(TCN1d, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=feature_num, 
#                                out_channels=feature_num, 
#                                kernel_size=3, 
#                                dilation=dilation, 
#                                padding=((3-1)*dilation)//2, 
#                                groups=feature_num) 
        
#         self.bn1 = nn.BatchNorm1d(feature_num)

#         self.conv2 = nn.Conv1d(in_channels=feature_num, 
#                                out_channels=feature_num, 
#                                kernel_size=5, 
#                                dilation=dilation, 
#                                padding=((5-1)*dilation)//2, 
#                                groups=feature_num) 
#         self.bn2 = nn.BatchNorm1d(feature_num)
        
#         self.conv3 = nn.Conv1d(in_channels=feature_num, 
#                                out_channels=feature_num, 
#                                kernel_size=7, 
#                                dilation=dilation, 
#                                padding=((7-1)*dilation)//2, 
#                                groups=feature_num)
#         self.bn3 = nn.BatchNorm1d(feature_num)
#         self.apply(weights_init)        

        
        
#     def forward(self, x):
#         # Multi scale
#         y1 = F.relu(self.bn1(self.conv1(x)))
#         y2 = F.relu(self.bn2(self.conv2(x)))
#         y3 = F.relu(self.bn3(self.conv3(x)))

#         y = (y1 + y2 + y3) / 3

#         return x + y

# class DynamicGraphEmbedding(nn.Module):
#     def __init__(self, in_channels, out_channels, num_nodes, topk=20, heads=1, concat=False,
#                  negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, lambda_val=1.0, 
#                  enable_subgraph_features=True):
#         super(DynamicGraphEmbedding, self).__init__()
#         self.graph_layer = GraphLayer(in_channels, out_channels, heads=heads,
#                                       concat=concat, negative_slope=negative_slope,
#                                       dropout=dropout, bias=bias, inter_dim=inter_dim)
        
#         self.lambda_val = lambda_val
#         self.topk = topk
#         self.enable_subgraph_features = enable_subgraph_features
#         self.MSConv = TCN1d(num_nodes)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         embed_dim = out_channels
#         self.embedding = nn.Embedding(num_nodes, embed_dim)
#         self.dropout = nn.Dropout(dropout)
        
#         # GraphSNN特有的子图特征提取器
#         if self.enable_subgraph_features:
#             self.subgraph_mlp = nn.Sequential(
#                 nn.Linear(in_channels, in_channels),
#                 nn.ReLU(),
#                 nn.Linear(in_channels, in_channels)
#             )
        
#         self.init_params()
    
#     def init_params(self):
#         nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        
#     def forward(self, x, return_attention_weights=False):
#         x = x.permute(0, 2, 1).contiguous()  # [batch_size, num_feature, window_size]
#         x = self.MSConv(x)  # Apply multi-scale convolution
#         batch_size, num_feature, window_size = x.size()
#         x = x.view(-1, window_size)
        
#         # GraphSNN的动态图构建
#         all_embeddings = self.embedding(torch.arange(num_feature).to(self.device))
#         weights_arr = all_embeddings.detach().clone()
#         all_embeddings = all_embeddings.repeat(batch_size, 1)

#         weights = weights_arr.view(num_feature, -1)
        
#         # 计算节点相似度（用于图构建）
#         cos_ji_mat = torch.matmul(weights, weights.T)
#         normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
#         cos_ji_mat = cos_ji_mat / (normed_mat + 1e-8)

#         topk_num = self.topk
#         topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

#         # 构建动态图的边
#         gated_i = torch.arange(0, num_feature).unsqueeze(1).expand(-1, topk_num).flatten().to(self.device).unsqueeze(0)
#         gated_j = topk_indices_ji.flatten().unsqueeze(0)
#         gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
        
#         # 构建无向图
#         gated_edge_index = torch.cat([
#             gated_edge_index, 
#             gated_edge_index.flip(0)
#         ], dim=1)

#         batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_size, num_feature).to(self.device)
        
#         # GraphSNN的核心：计算结构系数
#         edge_weights = self._compute_structural_coefficients(batch_gated_edge_index, batch_size, num_feature)
        
#         # GraphSNN的子图特征增强
#         if self.enable_subgraph_features:
#             x = self._enhance_with_subgraph_features(x, batch_gated_edge_index, edge_weights, batch_size, num_feature)

#         # 通过GraphSNN层进行消息传递
#         if return_attention_weights:
#             out, (new_edge_index, attn_edge) = self.graph_layer(
#                 x, batch_gated_edge_index, edge_weights=edge_weights, 
#                 embedding=all_embeddings, return_attention_weights=return_attention_weights
#             )
#         else:
#             out = self.graph_layer(
#                 x, batch_gated_edge_index, edge_weights=edge_weights, 
#                 embedding=all_embeddings, return_attention_weights=return_attention_weights
#             )
            
#         out = out.view(batch_size, num_feature, -1) # [batch_size, num_feature, out_channels]
#         out = out.permute(0, 2, 1)

#         if return_attention_weights:
#             return out, (new_edge_index, attn_edge)
#         else:
#             return out
    
#     def _compute_structural_coefficients(self, batch_gated_edge_index, batch_size, num_feature):
#         """向量化计算GraphSNN的结构系数（批处理优化版）"""
#         # 获取总节点数和总边数
#         total_nodes = batch_size * num_feature
#         num_edges = batch_gated_edge_index.size(1)
        
#         # 创建批处理索引 [0,0,...,0,1,1,...,1,...,batch_size-1,...]
#         batch_idx = torch.div(batch_gated_edge_index[0], num_feature, rounding_mode='floor')
        
#         # 创建节点局部索引 (0到num_feature-1范围内)
#         local_nodes = batch_gated_edge_index % num_feature
        
#         # 创建批处理邻接矩阵（稀疏表示）
#         adj_indices = torch.stack([
#             batch_idx * num_feature + local_nodes[0],
#             batch_idx * num_feature + local_nodes[1]
#         ], dim=0)
        
#         adj_values = torch.ones(num_edges, device=batch_gated_edge_index.device)
#         adj_size = (batch_size * num_feature, batch_size * num_feature)
#         adj = torch.sparse_coo_tensor(adj_indices, adj_values, adj_size)
        
#         # 计算共同邻居 - 使用矩阵乘法 (A * A^T)
#         # 注意：这里只计算每个批次内部的邻居关系
#         adj_dense = adj.to_dense()
#         common_neighbors = torch.matmul(adj_dense, adj_dense.t())
        
#         # 提取每条边的共同邻居数
#         edge_common = common_neighbors[
#             batch_gated_edge_index[0], 
#             batch_gated_edge_index[1]
#         ]
        
#         # 计算结构系数 (|V_ij| = 2 + |N(i) ∩ N(j)|)
#         structural_coeffs = (2 + edge_common) ** self.lambda_val
        
#         # 行归一化（按源节点）
#         row_sum = torch.zeros(total_nodes, device=structural_coeffs.device)
#         row_sum.scatter_add_(0, batch_gated_edge_index[0], structural_coeffs)
        
#         # 避免除零错误
#         norm_coeffs = structural_coeffs / (row_sum[batch_gated_edge_index[0]] + 1e-8)
        
#         return norm_coeffs
    
#     def _enhance_with_subgraph_features(self, x, edge_index, edge_weights, batch_size, num_feature):
#         """向量化计算子图特征（批处理优化版）"""
#         if not self.enable_subgraph_features or edge_weights is None:
#             return x

#         # 准备批处理数据
#         total_nodes = x.size(0)

#         # 获取源节点特征
#         source_idx = edge_index[0]
#         source_features = x[source_idx]

#         # 获取邻居特征并应用结构权重
#         neighbor_features = x[edge_index[1]]
#         weighted_neighbors = neighbor_features * edge_weights.view(-1, 1)

#         # 使用segment_coo进行高效聚合
#         aggregated = segment_coo(
#             src=weighted_neighbors,
#             index=source_idx,
#             out=torch.zeros_like(x),
#             reduce='sum'
#         )

#         # 应用子图特征变换
#         subgraph_feat = self.subgraph_mlp(x + aggregated)

#         # 残差连接 (原始特征 + 0.1 * 子图特征)
#         enhanced_feat = x + 0.1 * subgraph_feat

#         return enhanced_feat

class AdaGCNConv(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                 add_self_loops=False, normalize=True, bias=True, init_method='all', lambda_val=1.0):
        super(AdaGCNConv, self).__init__(aggr='add', node_dim=0)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias
        self.init_method = init_method
        self.lambda_val = lambda_val  # 结构系数参数λ

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
    
    # def compute_structural_coeff(self, edge_index):
    #     """计算结构系数矩阵"""
    #     # 创建对称邻接矩阵（无自环）
    #     adj = torch.zeros((self.num_nodes, self.num_nodes), 
    #                      device=edge_index.device)
    #     adj[edge_index[0], edge_index[1]] = 1
    #     adj = (adj + adj.t()).clamp(0, 1)  # 确保对称
        
    #     # 计算共同邻居矩阵
    #     common_neighbors = torch.mm(adj, adj.t())
        
    #     # 计算结构系数
    #     structural_coeff = torch.zeros_like(adj, dtype=torch.float)
    #     for i in range(self.num_nodes):
    #         for j in range(self.num_nodes):
    #             if adj[i, j] == 1:  # 只处理存在的边
    #                 cn = common_neighbors[i, j].item()
    #                 n_nodes = 2 + cn  # |V_uv| = 2 + |N(u)∩N(v)|
    #                 n_edges_approx = 1 + 2 * cn  # 近似边数 (u-v, u-cn, v-cn)
                    
    #                 # 避免除以零
    #                 if n_nodes > 1:
    #                     density = n_edges_approx / (n_nodes * (n_nodes - 1))
    #                     structural_coeff[i, j] = density * (n_nodes ** self.lambda_val)
        
    #     return structural_coeff
    
    # def compute_structural_coeff(self, edge_index):
    #     """
    #     基于共同邻居子图密度的结构系数计算
        
    #     参数:
    #     edge_index (torch.Tensor): 边索引，形状为 [2, num_edges]
    #     num_nodes (int): 节点数量
    #     lambda_val (float): 幂指数参数
        
    #     返回:
    #     torch.Tensor: 结构系数矩阵，形状为 [num_nodes, num_nodes]
    #     """
    #     # 构建邻接矩阵
    #     adj = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
    #     adj[edge_index[0], edge_index[1]] = 1
    #     adj = (adj + adj.t()).clamp(0, 1)  # 确保对称
        
    #     # 创建图和子图
    #     G = nx.from_numpy_array(adj.cpu().numpy())
    #     sub_graphs = []
        
    #     for i in range(self.num_nodes):
    #         s_indexes = [i]
    #         for j in range(self.num_nodes):
    #             if adj[i, j] == 1:
    #                 s_indexes.append(j)
    #         sub_graphs.append(G.subgraph(s_indexes))
        
    #     # 获取子图节点列表和邻接矩阵
    #     subgraph_nodes_list = [list(sub_g.nodes) for sub_g in sub_graphs]
    #     sub_graphs_adj = [nx.adjacency_matrix(sub_g).toarray() for sub_g in sub_graphs]
        
    #     # 计算结构系数矩阵
    #     new_adj = torch.zeros(self.num_nodes, self.num_nodes, device=edge_index.device)
        
    #     for node in range(self.num_nodes):
    #         sub_adj = torch.tensor(sub_graphs_adj[node], dtype=torch.float, device=edge_index.device)
    #         nodes_list = subgraph_nodes_list[node]
            
    #         for neighbor_idx in range(len(nodes_list)):
    #             neighbor = nodes_list[neighbor_idx]
    #             if neighbor == node:
    #                 continue
                    
    #             # 计算共同邻居
    #             c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[neighbor])
    #             if neighbor in c_neighbors:
    #                 c_neighbors_list = list(c_neighbors)
    #                 count = torch.tensor(0.0, device=edge_index.device)
                    
    #                 # 计算共同邻居子图中的边数
    #                 for i, item1 in enumerate(nodes_list):
    #                     if item1 in c_neighbors:
    #                         for item2 in c_neighbors_list:
    #                             j = nodes_list.index(item2)
    #                             count += sub_adj[i, j]

    #                 # 计算结构系数 (避免除以零)
    #                 if len(c_neighbors) > 1:
    #                     new_adj[node, neighbor] = count / 2
    #                     new_adj[node, neighbor] /= (len(c_neighbors) * (len(c_neighbors) - 1))
    #                     new_adj[node, neighbor] *= (len(c_neighbors) ** self.lambda_val)
        
    #     return new_adj

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
    
    def forward(self, x, edge_index, edge_weight=None):
        # x形状: [num_nodes, batch_size, seq_len]
        # edge_index形状: [2, num_edges]
        
        # 1. 计算特征相似度作为边权重
        x_norm = F.normalize(x, p=2, dim=-1)  # 归一化特征 [num_nodes, batch_size, win_size]

        # Reshape x_norm to [num_nodes, batch_size * win_size] if you want cross-node similarities
        # Or better, permute dimensions to get [batch_size, num_nodes, win_size]
        x_norm_permuted = x_norm.permute(1, 0, 2)  # [batch_size, num_nodes, win_size]
        
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

        # 3. 计算结构系数 (仅对剪枝后的边)
        structural_coeff = self.compute_structural_coeff(pruned_edge_index)
        # 4. 结合相似度和结构系数
        if edge_weight is None:
            edge_weight = mean_similarity[src, dst][edge_valid]
        # 获取结构系数值
        src_pruned, dst_pruned = pruned_edge_index
        structural_values = structural_coeff[src_pruned, dst_pruned]
        # 融合相似度和结构系数 (加权相加)
        fused_edge_weight = edge_weight + structural_values
        # 5. 归一化
        if self.normalize:
            pruned_edge_index, fused_edge_weight = gcn_norm(
                pruned_edge_index, fused_edge_weight, x.size(self.node_dim),
                self.improved, self.add_self_loops, dtype=x.dtype)

        # pruned_edge_weight = mean_similarity[src, dst][edge_valid] if edge_weight is None else edge_weight[edge_valid]
        # # 4. 归一化边权重
        # if self.normalize:
        #     pruned_edge_index, pruned_edge_weight = gcn_norm(
        #         pruned_edge_index, pruned_edge_weight, x.size(self.node_dim),
        #         self.improved, self.add_self_loops, dtype=x.dtype)
        
        # 5. 特征变换
        x = torch.matmul(x, self.weight)
        
        # 6. 消息传递(不再需要logits，因为边已经通过相似度筛选)
        out = self.propagate(pruned_edge_index, x=x, edge_weight=fused_edge_weight)
        
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1, 1) * x_j  # 直接使用相似度作为权重
    
# class AdaGCNConv(MessagePassing):
#     def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
#                  add_self_loops=False, normalize=True, bias=True, init_method='all', lambda_val=0.5):
#         super(AdaGCNConv, self).__init__(aggr='add', node_dim=0)
#         self.num_nodes = num_nodes
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.improved = improved
#         self.add_self_loops = add_self_loops
#         self.normalize = normalize
#         self.bias = bias
#         self.init_method = init_method
#         self.lambda_val = lambda_val

#         self.weight = Parameter(torch.Tensor(in_channels, out_channels))

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.logits = None
#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.weight)
#         zeros(self.bias)

#     def forward(self, x, edge_index, edge_weight=None):
#         x_norm = F.normalize(x, p=2, dim=-1)
#         x_norm_permuted = x_norm.permute(1, 0, 2)
#         similarity = torch.matmul(x_norm_permuted, x_norm_permuted.transpose(-1, -2))
#         similarity = similarity.permute(1, 0, 2)
#         mean_similarity = similarity.mean(dim=1)

#         num_edges_per_node = int(self.num_nodes * 0.3)
#         num_edges_per_node = max(num_edges_per_node, 1)
#         topk_values, topk_indices = torch.topk(mean_similarity, num_edges_per_node + 1, dim=-1)

#         edge_mask = torch.zeros_like(mean_similarity, dtype=torch.bool)
#         for i in range(self.num_nodes):
#             neighbors = topk_indices[i][topk_indices[i] != i]
#             neighbors = neighbors[:num_edges_per_node]
#             edge_mask[i, neighbors] = True

#         src, dst = edge_index
#         edge_valid = edge_mask[src, dst]
#         pruned_edge_index = edge_index[:, edge_valid]
#         pruned_edge_weight = mean_similarity[src, dst][edge_valid] if edge_weight is None else edge_weight[edge_valid]

#         structural_coeffs = self._compute_structural_coefficients(pruned_edge_index, self.num_nodes)

#         pruned_edge_weight = pruned_edge_weight * structural_coeffs

#         if self.normalize:
#             pruned_edge_index, pruned_edge_weight = gcn_norm(
#                 pruned_edge_index, pruned_edge_weight, x.size(self.node_dim),
#                 self.improved, self.add_self_loops, dtype=x.dtype)

#         x = torch.matmul(x, self.weight)
#         out = self.propagate(pruned_edge_index, x=x, edge_weight=pruned_edge_weight)

#         if self.bias is not None:
#             out += self.bias

#         return out

#     def message(self, x_j, edge_weight):
#         return edge_weight.view(-1, 1, 1) * x_j

#     def _compute_structural_coefficients(self, edge_index, num_nodes):
#         adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
#         adj[edge_index[0], edge_index[1]] = 1
#         common_neighbors = torch.matmul(adj, adj)
#         edge_common = common_neighbors[edge_index[0], edge_index[1]]
#         structural_coeffs = (2 + edge_common.float()) ** self.lambda_val
#         row_sum = torch.zeros(num_nodes, device=structural_coeffs.device)
#         row_sum.scatter_add_(0, edge_index[0], structural_coeffs)
#         norm_coeffs = structural_coeffs / (row_sum[edge_index[0]] + 1e-8)
#         return norm_coeffs

# class AdaGCNConv(MessagePassing):
#     def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
#                  add_self_loops=False, normalize=True, bias=True, init_method='all', 
#                  lambda_val=0.5, edge_ratio=0.3, use_cache=True):
#         super(AdaGCNConv, self).__init__(aggr='add', node_dim=0)
#         self.num_nodes = num_nodes
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.improved = improved
#         self.add_self_loops = add_self_loops
#         self.normalize = normalize
#         self.bias = bias
#         self.init_method = init_method
#         self.lambda_val = lambda_val
#         self.edge_ratio = edge_ratio  # 可配置的边保留比例
#         self.use_cache = use_cache    # 是否使用缓存

#         self.weight = Parameter(torch.Tensor(in_channels, out_channels))

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         # 缓存机制
#         self._cached_edge_index = None
#         self._cached_edge_weight = None
#         self._last_similarity_hash = None

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.weight)
#         zeros(self.bias)

#     def forward(self, x, edge_index, edge_weight=None):
#         # 改进1: 批量化相似度计算
#         x_norm = F.normalize(x, p=2, dim=-1)
#         similarity = self._compute_similarity_batch(x_norm)
        
#         # 改进2: 缓存机制 - 如果相似度没有显著变化，复用边索引
#         similarity_hash = self._compute_hash(similarity) if self.use_cache else None
        
#         if (self.use_cache and self._last_similarity_hash is not None and 
#             abs(similarity_hash - self._last_similarity_hash) < 1e-6):
#             pruned_edge_index = self._cached_edge_index
#             pruned_edge_weight = self._cached_edge_weight
#         else:
#             # 改进3: 更高效的边选择策略
#             pruned_edge_index, pruned_edge_weight = self._select_edges_efficient(
#                 similarity, edge_index, edge_weight)
            
#             if self.use_cache:
#                 self._cached_edge_index = pruned_edge_index
#                 self._cached_edge_weight = pruned_edge_weight
#                 self._last_similarity_hash = similarity_hash

#         # 改进4: 优化的结构系数计算
#         structural_coeffs = self._compute_structural_coefficients_optimized(
#             pruned_edge_index, self.num_nodes)

#         # 改进5: 自适应权重融合
#         alpha = torch.sigmoid(torch.tensor(self.lambda_val))  # 学习权重平衡
#         final_edge_weight = alpha * pruned_edge_weight + (1 - alpha) * structural_coeffs

#         if self.normalize:
#             pruned_edge_index, final_edge_weight = gcn_norm(
#                 pruned_edge_index, final_edge_weight, x.size(self.node_dim),
#                 self.improved, self.add_self_loops, dtype=x.dtype)

#         # 改进6: 残差连接
#         x_original = x
#         x = torch.matmul(x, self.weight)
#         out = self.propagate(pruned_edge_index, x=x, edge_weight=final_edge_weight)
        
#         # 残差连接（如果维度匹配）
#         if self.in_channels == self.out_channels:
#             out = out + x_original

#         if self.bias is not None:
#             out += self.bias

#         return out

#     def _compute_similarity_batch(self, x_norm):
#         """批量化相似度计算"""
#         # 使用einsum进行更高效的批量矩阵乘法
#         x_norm_permuted = x_norm.permute(1, 0, 2)  # [batch, nodes, features]
#         similarity = torch.einsum('bif,bjf->bij', x_norm_permuted, x_norm_permuted)
#         return similarity.mean(dim=0)  # [nodes, nodes]

#     def _select_edges_efficient(self, similarity, edge_index, edge_weight):
#         """更高效的边选择策略"""
#         num_edges_per_node = max(int(self.num_nodes * self.edge_ratio), 1)
        
#         # 使用torch.topk的sorted=False选项提升性能
#         topk_values, topk_indices = torch.topk(
#             similarity, num_edges_per_node + 1, dim=-1, sorted=False)
        
#         # 向量化的边mask构建
#         row_indices = torch.arange(self.num_nodes, device=similarity.device)
#         row_indices = row_indices.unsqueeze(1).expand(-1, num_edges_per_node + 1)
        
#         # 排除自连接
#         mask = topk_indices != row_indices
        
#         # 确保有足够的有效邻居
#         valid_counts = mask.sum(dim=1)
#         min_valid = valid_counts.min().item()
#         actual_edges_per_node = min(num_edges_per_node, min_valid)
        
#         if actual_edges_per_node <= 0:
#             actual_edges_per_node = 1
        
#         # 重新构建有效邻居矩阵
#         valid_neighbors = torch.zeros((self.num_nodes, actual_edges_per_node), 
#                                     dtype=torch.long, device=similarity.device)
        
#         for i in range(self.num_nodes):
#             valid_mask = mask[i]
#             valid_indices = topk_indices[i][valid_mask][:actual_edges_per_node]
#             if len(valid_indices) < actual_edges_per_node:
#                 # 如果没有足够的邻居，用其他节点填充
#                 remaining = actual_edges_per_node - len(valid_indices)
#                 all_others = torch.arange(self.num_nodes, device=similarity.device)
#                 others = all_others[all_others != i][:remaining]
#                 valid_indices = torch.cat([valid_indices, others])
#             valid_neighbors[i] = valid_indices
        
#         # 构建边索引
#         src_nodes = torch.arange(self.num_nodes, device=similarity.device)
#         src_nodes = src_nodes.unsqueeze(1).expand(-1, actual_edges_per_node).flatten()
#         dst_nodes = valid_neighbors.flatten()
        
#         new_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
#         new_edge_weight = similarity[src_nodes, dst_nodes]
        
#         return new_edge_index, new_edge_weight

#     def _compute_structural_coefficients_optimized(self, edge_index, num_nodes):
#         """优化的结构系数计算"""
#         # 使用稀疏矩阵操作提升效率
#         indices = edge_index
#         values = torch.ones(edge_index.size(1), device=edge_index.device)
#         adj_sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        
#         # 稀疏矩阵乘法计算共同邻居
#         common_neighbors_sparse = torch.sparse.mm(adj_sparse, adj_sparse.t())
        
#         # 修复索引计算问题
#         common_neighbors_dense = common_neighbors_sparse.to_dense()
#         edge_common = common_neighbors_dense[edge_index[0], edge_index[1]]
        
#         # 计算结构系数
#         structural_coeffs = torch.pow(2 + edge_common.float(), self.lambda_val)
        
#         # 行归一化 - 确保索引在有效范围内
#         row_sum = torch.zeros(num_nodes, device=structural_coeffs.device)
        
#         # 添加边界检查
#         valid_indices = (edge_index[0] >= 0) & (edge_index[0] < num_nodes)
#         if valid_indices.any():
#             row_sum.scatter_add_(0, edge_index[0][valid_indices], structural_coeffs[valid_indices])
#             norm_coeffs = structural_coeffs / (row_sum[edge_index[0]] + 1e-8)
#         else:
        #     norm_coeffs = structural_coeffs
        
        # return norm_coeffs
    
    # def _compute_hash(self, tensor):
    #     """计算张量的简单哈希值用于缓存"""
    #     return torch.sum(tensor).item()

    # def message(self, x_j, edge_weight):
    #     return edge_weight.view(-1, 1, 1) * x_j

class DynamicGraphEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, seq_len, num_levels=1, device=torch.device('cuda:0'), lambda_val=1.0):
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
        
        self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len, lambda_val=lambda_val)

    def forward(self, x):
        # 输入x形状: (bsz, seq_len, num_nodes)
        x = x.permute(2, 0, 1)  # >> (num_nodes, bsz, seq_len)
        
        # 进行图卷积
        for i in range(self.num_levels):
            x = self.gc_module(x, self.edge_index)  # >> (num_nodes, bsz, seq_len)
        
        x = x.permute(1, 2, 0)  # >> (bsz, seq_len, num_nodes)
        return x

# class DynamicGraphEmbedding(torch.nn.Module):
#     def __init__(self, num_nodes, seq_len, num_levels=1, device=torch.device('cuda:0'), lambda_val=1.0):
#         super(DynamicGraphEmbedding, self).__init__()
#         self.num_nodes = num_nodes
#         self.seq_len = seq_len
#         self.device = device
#         self.num_levels = num_levels
        
#         # 初始化全连接边索引
#         source_nodes, target_nodes = [], []
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 if i != j:  # 排除自连接
#                     source_nodes.append(j)
#                     target_nodes.append(i)
#         self.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long, device=self.device)
        
#         # 修复AdaGCNConv的初始化参数
#         self.gc_module = AdaGCNConv(
#             num_nodes=num_nodes, 
#             in_channels=seq_len, 
#             out_channels=seq_len, 
#             lambda_val=lambda_val,
#             edge_ratio=0.3,
#             use_cache=False  # 暂时禁用缓存以避免问题
#         ).to(device)

#     def forward(self, x):
#         # 输入x形状: (bsz, seq_len, num_nodes)
#         x = x.permute(2, 0, 1)  # >> (num_nodes, bsz, seq_len)
        
#         # 进行图卷积
#         for i in range(self.num_levels):
#             x = self.gc_module(x, self.edge_index)  # >> (num_nodes, bsz, seq_len)
        
#         x = x.permute(1, 2, 0)  # >> (bsz, seq_len, num_nodes)
#         return x


if __name__ == '__main__':
    # x:[batch_size, window_size, num_feature]
    x = torch.randn(4, 10, 16).to('cuda')
    # # switch to [batch_size, num_feature, window_size]
    # x = x.permute(0, 2, 1).contiguous()
    # # switch to [batch_size * num_feature, window_size] -> [batch_size * num_nodes, window_size]
    # x = x.view(-1, 10)
    # edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
    #                            [1, 2, 3, 4, 5, 0]], dtype=torch.long).to('cuda')
    # cache_edge_index = get_batch_edge_index(edge_index, 4, 16).to('cuda')
    # embedding = torch.randn(64, 5).to('cuda')  # Example embedding
    # layer = GraphLayer(in_channels=10, out_channels=5, heads=2, concat=False, dropout=0.1).to('cuda')
    # out, (new_edge_index, alpha) = layer(x, cache_edge_index, embedding, return_attention_weights=True)
    # print("Output shape:", out.shape)
    # layer = DynamicGraphEmbedding(in_channels=10, out_channels=10, num_nodes=16, topk=3, heads=2, concat=False, dropout=0.1, lambda_val=1.0).to('cuda')
    # out, (new_edge_index, attn_edge) = layer(x, return_attention_weights=True)
    # print("Output shape:", out.shape)
    layer = DynamicGraphEmbedding(num_nodes=16, seq_len=10, num_levels=1, device='cuda').to('cuda')
    out = layer(x)
    print("Output shape:", out.shape)  # 应该是 [batch_size, seq_len, num_nodes]
