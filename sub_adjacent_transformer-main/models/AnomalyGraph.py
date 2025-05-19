# 输入：x.shape - [batch_size, windows_size, num_features]
# 输出：out.shape - [batch_size, windows_size, num_features] and 输出每个节点相连边的权重总和
# 实现一个图注意力神经网络（Graph Attention Neural Network），用于处理时间序列数据
# 网络设计要求：1. 图构建：将每个特征视为图中的一个节点，构建一个全连接图。节点间的关系通过边权重来表示。
# 2. 边权重赋值：使用图注意力机制为图中的每条边赋予权重。可以参考GraphSAGE或GAT等模型来实现边权重的计算。
# 3. 图重构：根据边权重对图进行重构，得到输出out。
# 损失函数：1. 重构损失：out和x之间的差异
# 2.将原本的子邻域注意力矩阵视为一种全连接图，子邻域中的低注意力分数对应图中的弱连接边 mylossnew_change
# 因为for循环每个batch导致模型速度过慢，所以尽量采用批处理的方式来实现

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyGraph(nn.Module):
    """
    Graph Attention Neural Network for time series data.
    Input: x.shape - [batch_size, windows_size, num_features]
    Output: out.shape - [batch_size, windows_size, num_features] and the sum of edge weights for each node
    """
    def __init__(self, win_size, num_features, hidden_dim=64, activation="relu", dropout=0.1):
        super(AnomalyGraph, self).__init__()
        self.win_size = win_size
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Feature transformation layers
        self.feature_proj = nn.Linear(win_size, hidden_dim)
        
        # Edge attention mechanism
        self.edge_attn1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.edge_attn2 = nn.Linear(hidden_dim, 1)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, win_size)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, win_size, num_features]
        Returns:
            out: Output tensor of shape [batch_size, win_size, num_features]
            edge_weights_sum: Sum of edge weights for each node
        """
        batch_size, win_size, num_features = x.shape
        identity = x
        
        # Transpose to work with features as nodes
        # [batch_size, num_features, win_size]
        x_transposed = x.transpose(1, 2)
        
        # Project each feature (node) to hidden dimension
        # [batch_size, num_features, hidden_dim]
        node_features = self.feature_proj(x_transposed)
        
        # Calculate edge attention scores between all pairs of nodes
        # Create all possible pairs of nodes
        node_i = node_features.unsqueeze(2).expand(-1, -1, num_features, -1)  # [batch, features, features, hidden]
        node_j = node_features.unsqueeze(1).expand(-1, num_features, -1, -1)  # [batch, features, features, hidden]
        
        # Concatenate node pairs
        # [batch_size, num_features, num_features, hidden_dim*2]
        node_pairs = torch.cat([node_i, node_j], dim=-1)
        
        # Calculate attention scores
        # [batch_size, num_features, num_features, hidden_dim]
        attn_hidden = self.activation(self.edge_attn1(node_pairs))
        # [batch_size, num_features, num_features, 1]
        attn_scores = self.edge_attn2(attn_hidden)
        # [batch_size, num_features, num_features]
        attn_scores = attn_scores.squeeze(-1)
        
        # Set self-attention to zero - we don't need self-loops
        mask = torch.eye(num_features, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        attn_scores = attn_scores.masked_fill(mask.bool(), float('-inf'))
        
        # Apply softmax to get edge weights
        # [batch_size, num_features, num_features]
        edge_weights = F.softmax(attn_scores, dim=-1)
        
        # Calculate sum of edge weights per node for anomaly detection
        # [batch_size, num_features]，统计每个节点的入边权重和
        edge_weights_sum = edge_weights.sum(dim=1)
        
        # Message passing - each node aggregates messages from its neighbors
        # [batch_size, num_features, hidden_dim]
        messages = torch.bmm(edge_weights, node_features)
        
        # Project back to original space
        # [batch_size, num_features, win_size]
        out_features = self.output_proj(messages)
        
        # Transpose back to original shape
        # [batch_size, win_size, num_features]
        out = out_features.transpose(1, 2)
        
        # Add residual connection and apply normalization
        out = self.norm(out + identity)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Expand edge_weights_sum to match dimensions for output
        # [batch_size, win_size, num_features]
        edge_weights_sum_expanded = edge_weights_sum.unsqueeze(1).expand(-1, win_size, -1)
        
        return out, edge_weights_sum_expanded

def AnomalyGraphLoss(edge_weights_sum, span=None, one_side=True):
    """
    Calculate anomaly loss based on edge weights in the graph.
    Lower edge weight sums indicate potential anomalies (weak connections).
    
    Args:
        edge_weights_sum: Sum of edge weights for each node [batch_size, win_size, num_features]
        span: Window size for contextual comparison (unused but kept for API compatibility)
        one_side: Flag for one-sided comparison (unused but kept for API compatibility)
    
    Returns:
        Loss tensor of shape [batch_size, win_size]
    """
    # Normalize edge weights across features
    # Low values indicate weak connections (potential anomalies)
    normalized_weights = edge_weights_sum / (edge_weights_sum.max(dim=-1, keepdim=True)[0] + 1e-6)
    
    # Calculate loss - inverse of average normalized weights
    # High loss = potential anomaly (weak connections across features)
    loss = 1.0 - torch.mean(normalized_weights, dim=-1)
    
    return loss  # [batch_size, win_size]