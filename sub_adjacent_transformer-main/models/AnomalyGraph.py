import torch
import torch.nn as nn
from torch.nn import Parameter, Linear, Sequential
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, scatter
from torch_geometric.nn.inits import glorot, zeros
import math

def segment_coo(src, index, out, reduce='sum'):
    """
    高效的COO格式分段聚合函数
    类似torch_scatter.scatter但更轻量
    
    Args:
        src: 源数据 [num_edges, features]
        index: 目标索引 [num_edges]
        out: 输出缓冲区 [num_nodes, features]
        reduce: 聚合方式 ('sum', 'mean', 'max')
    """
    # 获取唯一索引和反向索引
    unique, inverse, counts = torch.unique(
        index, return_inverse=True, return_counts=True)
    
    # 根据索引对源数据进行分组
    grouped_src = src.new_zeros(unique.size(0), src.size(1))
    grouped_src.index_add_(0, inverse, src)
    
    # 应用聚合操作
    if reduce == 'sum':
        pass  # index_add已经完成求和
    elif reduce == 'mean':
        grouped_src = grouped_src / counts.float().view(-1, 1)
    elif reduce == 'max':
        max_values = torch.zeros_like(grouped_src)
        max_values.scatter_reduce_(0, inverse.unsqueeze(-1).expand_as(src), 
                                 src, reduce='amax', include_self=False)
        grouped_src = max_values
    
    # 将结果映射回输出缓冲区
    out[unique] = grouped_src
    
    return out

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # GraphSNN特有的结构感知参数 - 修复维度问题
        if concat:
            self.struct_transform = Linear(heads * out_channels, heads * out_channels)
        else:
            self.struct_transform = Linear(out_channels, out_channels)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))
        # 在类初始化中添加
        self.struct_gate = Parameter(torch.Tensor([0.5]))
        
        # 结构系数的注意力参数
        self.att_struct = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.struct_transform.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        glorot(self.att_struct)
        
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weights=None, embedding=None, return_attention_weights=False):
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        # 先移除自环
        edge_index, edge_weights = remove_self_loops(edge_index, edge_weights)
        
        # 添加自环，并为自环设置权重
        num_nodes = x[1].size(self.node_dim)
        edge_index, edge_weights = add_self_loops(edge_index, edge_attr=edge_weights, 
                                                  fill_value=1.0, num_nodes=num_nodes)

        # GraphSNN的结构化消息传递
        out = self.propagate(edge_index, x=x, embedding=embedding,
                             edge_weights=edge_weights,
                             edges=edge_index,  # 传递完整的边信息
                             return_attention_weights=return_attention_weights)        

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # GraphSNN的结构变换 - 在BatchNorm之前应用
        out = self.struct_transform(out)

        if self.bias is not None:
            out = out + self.bias

        # 修复BatchNorm维度问题
        if self.concat:
            # 如果concat=True，需要重新reshape用于BatchNorm
            out_for_bn = out.view(-1, self.heads, self.out_channels).mean(dim=1)
            out_for_bn = self.bn(out_for_bn)
            out_for_bn = self.relu(out_for_bn)
            # 恢复原始形状
            out = out + out_for_bn.unsqueeze(1).repeat(1, self.heads).view(-1, self.heads * self.out_channels)
        else:
            out = self.bn(out)
            out = self.relu(out)
        
        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding=None, edges=None, edge_weights=None, return_attention_weights=False):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        # GraphSNN核心：应用结构系数到消息计算
        if edge_weights is not None:
            # 确保结构权重的维度与消息数量匹配
            if edge_weights.size(0) != x_j.size(0):
                # 如果维度不匹配，使用默认权重
                struct_weight = torch.ones(x_j.size(0), device=x_j.device).view(-1, 1, 1)
            else:
                struct_weight = edge_weights.view(-1, 1, 1)
                
            # x_j_struct = x_j * struct_weight
            x_j_struct = (1 - self.struct_gate) * x_j + self.struct_gate * (x_j * struct_weight)
        else:
            x_j_struct = x_j

        # 构建注意力机制的键值对
        if embedding is not None and edges is not None:
            # 安全地获取embedding索引 - 修复索引获取逻辑
            try:
                embedding_i = embedding[edge_index_i]
                # 使用edges的第二行作为目标节点索引
                if edges.dim() > 1 and edges.size(0) > 1:
                    # 获取对应的目标节点embedding
                    edge_target_idx = edges[1]
                    if edge_target_idx.size(0) == edge_index_i.size(0):
                        embedding_j = embedding[edge_target_idx]
                    else:
                        embedding_j = embedding[edge_index_i]  # fallback
                else:
                    embedding_j = embedding[edge_index_i]
            except (IndexError, RuntimeError):
                # 如果索引出错，使用默认处理
                embedding_i = embedding[edge_index_i]
                embedding_j = embedding[edge_index_i]
                
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j_struct, embedding_j), dim=-1)
            
            cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
            cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)
        else:
            key_i = x_i
            key_j = x_j_struct
            cat_att_i = self.att_i
            cat_att_j = self.att_j

        # GraphSNN的增强注意力计算
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        
        # 添加结构感知的注意力分量
        if edge_weights is not None and edge_weights.size(0) == x_j.size(0):
            struct_att = (x_j_struct * self.att_struct).sum(-1) * edge_weights.view(-1, 1)
            alpha = alpha + struct_att

        alpha = alpha.view(-1, self.heads, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        self.node_dim = 0
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j_struct * alpha.view(-1, self.heads, 1)



    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class TCN1d(nn.Module): # 多尺度卷积
    def __init__(self, feature_num, kernel_size=3, dilation=1):
        super(TCN1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_num, 
                               out_channels=feature_num, 
                               kernel_size=3, 
                               dilation=dilation, 
                               padding=((3-1)*dilation)//2, 
                               groups=feature_num) 
        
        self.bn1 = nn.BatchNorm1d(feature_num)

        self.conv2 = nn.Conv1d(in_channels=feature_num, 
                               out_channels=feature_num, 
                               kernel_size=5, 
                               dilation=dilation, 
                               padding=((5-1)*dilation)//2, 
                               groups=feature_num) 
        self.bn2 = nn.BatchNorm1d(feature_num)
        
        self.conv3 = nn.Conv1d(in_channels=feature_num, 
                               out_channels=feature_num, 
                               kernel_size=7, 
                               dilation=dilation, 
                               padding=((7-1)*dilation)//2, 
                               groups=feature_num)
        self.bn3 = nn.BatchNorm1d(feature_num)
        self.apply(weights_init)        

        
        
    def forward(self, x):
        # Multi scale
        y1 = F.relu(self.bn1(self.conv1(x)))
        y2 = F.relu(self.bn2(self.conv2(x)))
        y3 = F.relu(self.bn3(self.conv3(x)))

        y = (y1 + y2 + y3) / 3

        return x + y

class DynamicGraphEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, topk=20, heads=1, concat=False,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, lambda_val=1.0, 
                 enable_subgraph_features=True):
        super(DynamicGraphEmbedding, self).__init__()
        self.graph_layer = GraphLayer(in_channels, out_channels, heads=heads,
                                      concat=concat, negative_slope=negative_slope,
                                      dropout=dropout, bias=bias, inter_dim=inter_dim)
        
        self.lambda_val = lambda_val
        self.topk = topk
        self.enable_subgraph_features = enable_subgraph_features
        self.MSConv = TCN1d(num_nodes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embed_dim = out_channels
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # GraphSNN特有的子图特征提取器
        if self.enable_subgraph_features:
            self.subgraph_mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, in_channels)
            )
        
        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        
    def forward(self, x, return_attention_weights=True):
        x = x.permute(0, 2, 1).contiguous()  # [batch_size, num_feature, window_size]
        x = self.MSConv(x)  # Apply multi-scale convolution
        batch_size, num_feature, window_size = x.size()
        x = x.view(-1, window_size)
        
        # GraphSNN的动态图构建
        all_embeddings = self.embedding(torch.arange(num_feature).to(self.device))
        weights_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.repeat(batch_size, 1)

        weights = weights_arr.view(num_feature, -1)
        
        # 计算节点相似度（用于图构建）
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
        cos_ji_mat = cos_ji_mat / (normed_mat + 1e-8)

        topk_num = self.topk
        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

        # 构建动态图的边
        gated_i = torch.arange(0, num_feature).unsqueeze(1).expand(-1, topk_num).flatten().to(self.device).unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
        
        # 构建无向图
        gated_edge_index = torch.cat([
            gated_edge_index, 
            gated_edge_index.flip(0)
        ], dim=1)

        batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_size, num_feature).to(self.device)
        
        # GraphSNN的核心：计算结构系数
        edge_weights = self._compute_structural_coefficients(batch_gated_edge_index, batch_size, num_feature)
        
        # GraphSNN的子图特征增强
        if self.enable_subgraph_features:
            x = self._enhance_with_subgraph_features(x, batch_gated_edge_index, edge_weights, batch_size, num_feature)

        # 通过GraphSNN层进行消息传递
        if return_attention_weights:
            out, (new_edge_index, attn_edge) = self.graph_layer(
                x, batch_gated_edge_index, edge_weights=edge_weights, 
                embedding=all_embeddings, return_attention_weights=return_attention_weights
            )
        else:
            out = self.graph_layer(
                x, batch_gated_edge_index, edge_weights=edge_weights, 
                embedding=all_embeddings, return_attention_weights=return_attention_weights
            )
            
        out = out.view(batch_size, num_feature, -1) # [batch_size, num_feature, out_channels]
        out = out.permute(0, 2, 1)

        if return_attention_weights:
            return out, (new_edge_index, attn_edge)
        else:
            return out
    
    def _compute_structural_coefficients(self, batch_gated_edge_index, batch_size, num_feature):
        """向量化计算GraphSNN的结构系数（批处理优化版）"""
        # 获取总节点数和总边数
        total_nodes = batch_size * num_feature
        num_edges = batch_gated_edge_index.size(1)
        
        # 创建批处理索引 [0,0,...,0,1,1,...,1,...,batch_size-1,...]
        batch_idx = torch.div(batch_gated_edge_index[0], num_feature, rounding_mode='floor')
        
        # 创建节点局部索引 (0到num_feature-1范围内)
        local_nodes = batch_gated_edge_index % num_feature
        
        # 创建批处理邻接矩阵（稀疏表示）
        adj_indices = torch.stack([
            batch_idx * num_feature + local_nodes[0],
            batch_idx * num_feature + local_nodes[1]
        ], dim=0)
        
        adj_values = torch.ones(num_edges, device=batch_gated_edge_index.device)
        adj_size = (batch_size * num_feature, batch_size * num_feature)
        adj = torch.sparse_coo_tensor(adj_indices, adj_values, adj_size)
        
        # 计算共同邻居 - 使用矩阵乘法 (A * A^T)
        # 注意：这里只计算每个批次内部的邻居关系
        adj_dense = adj.to_dense()
        common_neighbors = torch.matmul(adj_dense, adj_dense.t())
        
        # 提取每条边的共同邻居数
        edge_common = common_neighbors[
            batch_gated_edge_index[0], 
            batch_gated_edge_index[1]
        ]
        
        # 计算结构系数 (|V_ij| = 2 + |N(i) ∩ N(j)|)
        structural_coeffs = (2 + edge_common) ** self.lambda_val
        
        # 行归一化（按源节点）
        row_sum = torch.zeros(total_nodes, device=structural_coeffs.device)
        row_sum.scatter_add_(0, batch_gated_edge_index[0], structural_coeffs)
        
        # 避免除零错误
        norm_coeffs = structural_coeffs / (row_sum[batch_gated_edge_index[0]] + 1e-8)
        
        return norm_coeffs
    
    def _enhance_with_subgraph_features(self, x, edge_index, edge_weights, batch_size, num_feature):
        """向量化计算子图特征（批处理优化版）"""
        if not self.enable_subgraph_features or edge_weights is None:
            return x

        # 准备批处理数据
        total_nodes = x.size(0)

        # 获取源节点特征
        source_idx = edge_index[0]
        source_features = x[source_idx]

        # 获取邻居特征并应用结构权重
        neighbor_features = x[edge_index[1]]
        weighted_neighbors = neighbor_features * edge_weights.view(-1, 1)

        # 使用segment_coo进行高效聚合
        aggregated = segment_coo(
            src=weighted_neighbors,
            index=source_idx,
            out=torch.zeros_like(x),
            reduce='sum'
        )

        # 应用子图特征变换
        subgraph_feat = self.subgraph_mlp(x + aggregated)

        # 残差连接 (原始特征 + 0.1 * 子图特征)
        enhanced_feat = x + 0.1 * subgraph_feat

        return enhanced_feat

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
    layer = DynamicGraphEmbedding(in_channels=10, out_channels=10, num_nodes=16, topk=3, heads=2, concat=False, dropout=0.1, lambda_val=1.0).to('cuda')
    out, (new_edge_index, attn_edge) = layer(x, return_attention_weights=True)
    print("Output shape:", out.shape)
