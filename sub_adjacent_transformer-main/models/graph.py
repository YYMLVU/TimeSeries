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
import networkx as nx

class AdaGCNConv1(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                 add_self_loops=False, normalize=True, bias=True, init_method='all'):
        super(AdaGCNConv1, self).__init__(aggr='add', node_dim=0)
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
        x_norm = F.normalize(x, p=2, dim=-1)  # 归一化特征 [num_nodes, batch_size, feature_dim]

        # Reshape x_norm to [num_nodes, batch_size * feature_dim] if you want cross-node similarities
        # Or better, permute dimensions to get [batch_size, num_nodes, feature_dim]
        x_norm_permuted = x_norm.permute(1, 0, 2)  # [batch_size, num_nodes, feature_dim]
        
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
        
        self.gc_module = AdaGCNConv1(num_nodes, seq_len, seq_len)

    def forward(self, x):
        # 输入x形状: (bsz, seq_len, num_nodes)
        x = x.permute(2, 0, 1)  # >> (num_nodes, bsz, seq_len)
        
        # 进行图卷积
        for i in range(self.num_levels):
            x = self.gc_module(x, self.edge_index)  # >> (num_nodes, bsz, seq_len)
        
        x = x.permute(1, 2, 0)  # >> (bsz, seq_len, num_nodes)
        return x
            
class AdaGCNConv(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False, 
                 add_self_loops=False, normalize=True, bias=True, init_method='similarity'):
        super(AdaGCNConv, self).__init__(aggr='add', node_dim=0)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias
        self.init_method = init_method
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

    # def compute_structural_coeff(self, edge_index, lambda_val=1.0): # 原论文 cpu
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
    #     adj = torch.zeros((self.num_nodes, self.num_nodes), device='cuda:0')
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
    #     new_adj = torch.zeros(self.num_nodes, self.num_nodes, device='cuda:0')

    #     for node in range(self.num_nodes):
    #         sub_adj = torch.tensor(sub_graphs_adj[node], dtype=torch.float, device='cuda:0')
    #         nodes_list = subgraph_nodes_list[node]
            
    #         for neighbor_idx in range(len(nodes_list)):
    #             neighbor = nodes_list[neighbor_idx]
    #             if neighbor == node:
    #                 continue
                    
    #             # 计算共同邻居
    #             c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[neighbor])
    #             if neighbor in c_neighbors:
    #                 c_neighbors_list = list(c_neighbors)
    #                 count = torch.tensor(0.0, device='cuda:0')
                    
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
    #                     new_adj[node, neighbor] *= (len(c_neighbors) ** lambda_val)
        
    #     return new_adj

    # def compute_structural_coeff(self, edge_index, lambda_val=1.0): # 一样功能 Gpu
    #     """
    #     基于共同邻居子图密度的结构系数计算 - GPU加速版本
        
    #     参数:
    #     edge_index (torch.Tensor): 边索引，形状为 [2, num_edges]
    #     lambda_val (float): 幂指数参数
        
    #     返回:
    #     torch.Tensor: 结构系数矩阵，形状为 [num_nodes, num_nodes]
    #     """
    #     device = edge_index.device
        
    #     # 1. 构建邻接矩阵 (GPU上)
    #     adj = torch.zeros((self.num_nodes, self.num_nodes), device=device, dtype=torch.float)
    #     adj[edge_index[0], edge_index[1]] = 1
    #     adj = (adj + adj.t()).clamp(0, 1)  # 确保对称且二值化
        
    #     # 2. 计算每个节点的邻居集合 (包括自己)
    #     # neighbor_mask[i, j] = 1 表示节点j是节点i的邻居或者j=i
    #     neighbor_mask = adj.clone()
    #     neighbor_mask.fill_diagonal_(1)  # 添加自连接
        
    #     # 3. 为每个节点对计算共同邻居
    #     # 使用广播和矩阵运算来并行计算所有节点对的共同邻居
    #     # neighbor_mask_expanded: [num_nodes, 1, num_nodes]
    #     # neighbor_mask_expanded_T: [1, num_nodes, num_nodes]
    #     neighbor_mask_expanded = neighbor_mask.unsqueeze(1)  # [num_nodes, 1, num_nodes]
    #     neighbor_mask_expanded_T = neighbor_mask.unsqueeze(0)  # [1, num_nodes, num_nodes]
        
    #     # 计算共同邻居: 两个节点都与第三个节点相邻
    #     common_neighbors = neighbor_mask_expanded * neighbor_mask_expanded_T  # [num_nodes, num_nodes, num_nodes]
        
    #     # 4. 计算共同邻居子图中的边数
    #     # 对于节点对(i,j)，计算其共同邻居子图中的边数
    #     new_adj = torch.zeros((self.num_nodes, self.num_nodes), device=device, dtype=torch.float)
        
    #     for i in range(self.num_nodes):
    #         for j in range(self.num_nodes):
    #             if i == j:
    #                 continue
                    
    #             # 获取节点i和j的共同邻居
    #             common_neighbors_ij = common_neighbors[i, j]  # [num_nodes]
    #             common_neighbor_indices = torch.nonzero(common_neighbors_ij, as_tuple=True)[0]
                
    #             if len(common_neighbor_indices) <= 1:
    #                 continue
                    
    #             # 计算共同邻居子图的邻接矩阵
    #             subgraph_adj = adj[common_neighbor_indices][:, common_neighbor_indices]
                
    #             # 计算子图中的边数 (每条边被计算两次，所以除以2)
    #             edge_count = subgraph_adj.sum() / 2
                
    #             # 计算结构系数
    #             num_common = len(common_neighbor_indices)
    #             if num_common > 1:
    #                 max_edges = num_common * (num_common - 1) / 2  # 完全图的边数
    #                 if max_edges > 0:
    #                     density = edge_count / max_edges
    #                     new_adj[i, j] = density * (num_common ** lambda_val)
        
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

class GraphEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, seq_len, num_levels=1, device=torch.device('cuda:0')):
        super(GraphEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.device = device
        self.num_levels = num_levels
        
        # 使用相似度初始化
        self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len, init_method='similarity')

    def forward(self, x):
        # x形状: (bsz, seq_len, num_nodes)
        x = x.permute(2, 0, 1)  # (num_nodes, bsz, seq_len)
        
        for _ in range(self.num_levels):
            x = self.gc_module(x)
            
        x = x.permute(1, 2, 0)  # (bsz, seq_len, num_nodes)
        return x

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
    layer1 = GraphEmbedding(num_nodes=16, seq_len=10, num_levels=1, device='cuda').to('cuda')
    out = layer1(x)
    print('logits:', layer1.gc_module.logits[:10])  # 打印前10个
    print("Output shape:", out.shape)  # 应该是 [batch_size, seq_len, num_nodes]
    # save_path = './sub_adjacent_transformer-main/models/graph_embedding.pt'
    torch.save(layer1.state_dict(), './sub_adjacent_transformer-main/models/graph_embedding.pt')
    # # 查看保存的logits参数
    # print('Saved logits:', layer.gc_module.logits[:10])
    # load pt
    layer2 = GraphEmbedding(num_nodes=16, seq_len=10, num_levels=1, device='cuda').to('cuda')
    layer2.load_state_dict(torch.load('./sub_adjacent_transformer-main/models/graph_embedding.pt'))
    print('logits:', layer2.gc_module.logits[:10])  # 打印前10个
    out = layer2(x)
    print("Output shape after loading:", out.shape)  # 应该是 [batch_size, seq_len, num_nodes]


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