import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch.nn.functional as F

from models.modules import ConvLayer, ReconstructionModel
from models.transformer import EncoderLayer, MultiHeadAttention
from models.graph import DynamicGraphEmbedding
from model.AnomalyTransformer import AnomalyTransformer


'''

results, 2graph,  win=8  :
(F1, P, R)
--msl: (95.08,93.48,96.73)--20230324_104633   2.7   30 epoch
--smap:(90.32,95.62,85.58)--20230324_105432   6.0   30
--swat:(90.19,93.72,86.92)--20230331_172744   4.5   30
--wadi:(89.76,91.78,87.84)--20230329_105629   6.9   30
-------------------------------------------
results, 2graph,  win=16  :              
(F1, P, R)
--msl: (95.49,95.10,95.89)--20230323_173247   3.1
--smap:(91.83,90.81,92.88)--20230311_220757   6.0
--swat:(91.19,96.76,86.23)--20230322_102823   4.8
--wadi:(92.64,98.00,87.84)--20230329_095534   9.3
-------------------------------------------
results, 2graph,  win=32  :
(F1, P, R)
--msl: (95.00,94.18,95.84)--20230321_102427   4.8
--smap:(93.07,93.42,92.71)--20230327_171113   7.8
--swat:(91.87,94.81,89.11)--20230321_134845   7.5
--wadi:(88.31,88.79,87.84)--20230329_111323   18.0
--------------------------------------------
results, 2graph,  win=64  :
(F1, P, R)
--msl: (93.47,93.55,93.39)--20230320_094825    time:9.7
--smap:(90.30,95.55,85.60)--20230320_171625    time:11.0
--swat:(91.11,93.20,89.11)--20230331_174459    time:14.4
--wadi:(86.90,85.99,87.84)--20230317_180052    time:39.6
--------------------------------------------

---sensitivity---
graph layer=1  msl   (95.49,96.10,95.89)--20230323_173247     3.1
graph layer=2  msl   (94.34,94.82,93.86)--20230330_114206     4.3
graph layer=3  msl   (93.98,96.80,91.15)--20230329_165015     5.4
graph layer=4  msl   (93.49,93.90,93.10)--20230329_165631     6.6
graph layer=5  msl   (94.97,93.53,96.46)--20230329_171313     7.5
graph layer=6  msl   (93.08,93.09,93.09)--20230329_173156     8.8
graph layer=7  msl   (92.36,94.44,90.37)--20230329_174651     9.8
graph layer=8  msl   (92.81,92.14,93.48)--20230329_180412     10.8


---ablation---
results, 2graph cl, without transformer (use rnn)  win=16 
(F1, P, R)
--msl: (94.18,94.23,94.13)--20230328_150712    time:3.0    
--smap:(91.37,97.76,85.60)--20230328_171742    time:5.8
--swat:(89.62,90.14,89.11)--20230328_170613    time:4.7
--wadi:(86.29,84.80,87.84)--20230329_104913    9.5
--------------------------------------------

results, 2graph cl, without cross loss  win=16  :
(F1, P, R)
--msl: (94.55,92.35,96.86)--20230328_171229       2.8
--smap:(88.69,95.41,82.86)--20230329_103604       5.2
--swat:(89.15,94.98,83.99)--20230329_101612       4.4
--wadi:(90.76,93.88,87.84)--20230328_164119       9.0



results, 2 graph, without Contrastive learning  win=16  :
(F1, P, R)
--msl: (91.19,93.97,88.55)--20230329_123018  time: 2.1
--smap:(88.24,94.46,82.79)--20230329_123931      4.1
--swat:(89.27,95.25,83.99)--20230329_123437      3.3
--wadi:(88.79,89.76,87.84)--20230329_112802      7.2


results, only using intra graph cl win=16  :
(F1, P, R)
--msl: (90.96,91.96,89.98)--20230329_141006  time: 1.9
--smap:(87.71,96.50,80.38)--20230329_141727        4.2
--swat:(87.84,95.78,81.12)--20230329_143828        3.0
--wadi:(85.84,83.93,87.84)--20230329_144515   4.4


results, only using inter graph cl win=16  :
(F1, P, R)
--msl: (84.57,76.78,94.09)--20230329_152426   2.2
--smap:(80.80,86.91,75.49)--20230329_151117   4.0
--swat:(86.83,99.04,77.30)--20230329_150300   3.4
--wadi:(80.10,73.61,87.84)--20230329_145626   7.7



'''
# import torch.nn.functional as F

# def sim(h1, h2):
#     z1 = F.normalize(h1, dim=-1, p=2)
#     z2 = F.normalize(h2, dim=-1, p=2)
#     return torch.mm(z1, z2.t())

# def contrastive_loss(h1, h2):
#     f = lambda x: torch.exp(x)
#     inter_sim = f(sim(h1, h2))
#     return -torch.log(inter_sim.diag() /
#                      (inter_sim.sum(dim=-1) - inter_sim.diag()))


class ProjectionLayer(nn.Module):
    """Encoder的一层"""

    def __init__(self, n_feature, num_heads, dropout=0.1):
        super(ProjectionLayer, self).__init__()

        self.attention = MultiHeadAttention(n_feature, num_heads, dropout)

    def forward(self, inputs, attn_mask=None):

        context, _ = self.attention(inputs, inputs, inputs, attn_mask)
        return context

# 改进的 MaskGenerator 类
class MaskGenerator(nn.Module):
    """更先进的生成器网络，创建用于特征和时序维度的增强掩码"""
    
    def __init__(self, window_size, n_features, hidden_dim=128):
        super(MaskGenerator, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        
        # 更强大的编码器，使用残差连接
        self.encoder = nn.Sequential(
            nn.Linear(window_size * n_features, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 特征掩码解码器
        self.feature_mask_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_features),
            nn.Sigmoid()
        )
        
        # 时序掩码解码器
        self.temporal_mask_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, window_size),
            nn.Sigmoid()
        )
        
        # 注意力掩码解码器 (一种更细粒度的掩码)
        self.attention_mask_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, window_size * n_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 将输入展平
        x_flat = x.reshape(batch_size, -1)
        
        # 编码
        encoded = self.encoder(x_flat)
        
        # 生成三种不同的掩码
        feature_mask = self.feature_mask_decoder(encoded)  # [batch_size, n_features]
        temporal_mask = self.temporal_mask_decoder(encoded)  # [batch_size, window_size]
        attention_mask = self.attention_mask_decoder(encoded)  # [batch_size, window_size * n_features]
        attention_mask = attention_mask.view(batch_size, self.window_size, self.n_features)
        
        # 扩展维度以便应用掩码
        feature_mask = feature_mask.unsqueeze(1).expand(-1, self.window_size, -1)  # [batch_size, window_size, n_features]
        temporal_mask = temporal_mask.unsqueeze(2).expand(-1, -1, self.n_features)  # [batch_size, window_size, n_features]
        
        # 随机混合不同的掩码类型
        mask_weights = torch.rand(batch_size, 1, 1, device=x.device)
        final_mask = (mask_weights < 0.33).float() * feature_mask + \
                     ((mask_weights >= 0.33) & (mask_weights < 0.66)).float() * temporal_mask + \
                     (mask_weights >= 0.66).float() * attention_mask
        
        # 应用掩码
        masked_x = x * final_mask
        
        return masked_x, final_mask
    
class GANFeatureAugmenter(nn.Module):
    def __init__(self, window_size, num_features, lr=0.0002, device='cuda'):
        super(GANFeatureAugmenter, self).__init__()
        self.generator = MaskGenerator(window_size=window_size, n_features=num_features)
        self.device = device
        self.generator.to(device)
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # 增强策略
        self.augmentation_types = ['mask', 'noise', 'mixup', 'cutout']
        self.current_strategy = 'mask'
        
        # 超参数
        self.mask_ratio = 0.3  # 要掩码的特征比例
        self.noise_scale = 0.2  # 添加噪声的尺度
        self.mixup_alpha = 0.4  # mixup的alpha参数
        self.temperature = 0.1  # 对比损失温度
        
        # 运行统计
        self.running_loss = 0.0
        self.updates = 0
        self.augmentation_stats = {strategy: 0 for strategy in self.augmentation_types}
        
    def manual_augment(self, x, strategy=None):
        """手动应用特定的增强策略"""
        batch_size = x.shape[0]
        window_size = x.shape[1]
        num_features = x.shape[2]
        
        if strategy is None:
            strategy = random.choice(self.augmentation_types)
        
        self.augmentation_stats[strategy] += 1
        self.current_strategy = strategy
        
        if strategy == 'mask':
            # 随机掩码一些特征
            mask = torch.ones_like(x)
            for b in range(batch_size):
                # 每个批次随机掩码不同的特征
                feature_idx = random.sample(range(num_features), 
                                          int(num_features * self.mask_ratio))
                mask[b, :, feature_idx] = 0.0
            return x * mask
            
        elif strategy == 'noise':
            # 添加高斯噪声
            noise = torch.randn_like(x) * self.noise_scale * x.std(dim=1, keepdim=True)
            return x + noise
            
        elif strategy == 'mixup':
            # Mixup批次内的样本
            perm = torch.randperm(batch_size)
            mixed_x = self.mixup_alpha * x + (1 - self.mixup_alpha) * x[perm]
            return mixed_x
            
        elif strategy == 'cutout':
            # 时间序列的Cutout (删除连续的时间步长)
            x_aug = x.clone()
            for b in range(batch_size):
                length = random.randint(5, max(6, int(window_size * 0.2)))
                start = random.randint(0, window_size - length)
                x_aug[b, start:start+length, :] = 0
            return x_aug
            
        return x
    
    def aug_feature(self, x, training=True):
        """使用GAN网络增强输入特征"""
        if not training:
            return x
        
        # 有50%的概率使用手动增强，50%的概率使用GAN生成的增强
        if random.random() < 0.5:
            return self.manual_augment(x)
        else:
            masked_x, _ = self.generator(x)
            return masked_x
    
    def train_gan(self, x, output, rec_loss):
        """训练生成器创建能最大化重构损失的增强"""
        self.g_optimizer.zero_grad()
        
        # 使用detach创建新的计算图
        x_detached = x.detach()
        output_detached = output.detach()
        
        # 生成掩码版本并得到掩码
        masked_x, mask = self.generator(x_detached)
        
        # 计算重构误差 (对生成器而言，误差越大越好)
        criterion = nn.MSELoss()
        aug_loss = criterion(masked_x, output_detached)
        
        # 添加正则化项，使掩码更加稀疏
        sparsity_loss = torch.mean(mask)  # 鼓励更多的零（更多掩码）
        diversity_loss = -torch.mean(torch.std(mask, dim=0))  # 鼓励不同样本有不同的掩码
        
        # 综合损失 (我们希望最大化重构误差，同时保持稀疏和多样性)
        g_loss = -aug_loss + 0.1 * sparsity_loss + 0.05 * diversity_loss
        
        g_loss.backward()
        self.g_optimizer.step()
        
        self.running_loss += aug_loss.item()
        self.updates += 1
        
        return aug_loss.item()
    
    def get_stats(self):
        """返回训练统计信息"""
        avg_loss = self.running_loss / max(1, self.updates)
        stats = {
            'avg_loss': avg_loss,
            'updates': self.updates,
            'augmentation_counts': self.augmentation_stats
        }
        return stats
    
    def forward(self, x, training=True):
        return self.aug_feature(x, training)

class MODEL_CGRAPH_TRANS(nn.Module):
    """ MODEL_CGRAPH_TRANS model class.
    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        dropout=0.2,
        device = 'cuda:0'
    ):
        super(MODEL_CGRAPH_TRANS, self).__init__()
        self.window_size = window_size
        self.out_dim = out_dim
        self.f_dim = n_features
        self.decoder_type = 0  # 0:transformer   1:rnn
        self.use_cross_loss = True  # 
        self.use_contrastive = True
        self.use_intra_graph = True  
        self.use_inter_graph = True
        self.device = device

        # preprocessing
        self.conv = ConvLayer(n_features, 7)

        # augmentation learnable parameters
        self.param = nn.Linear(self.f_dim, self.f_dim)

        self.num_levels = 1
        # inter embedding module based on GNN
        # self.inter_module = GraphEmbedding(num_nodes=n_features, seq_len=window_size, num_levels=self.num_levels, device=torch.device(device))
        self.inter_module = DynamicGraphEmbedding(num_nodes=n_features, seq_len=window_size, num_levels=self.num_levels, device=torch.device(device))
        # self.inter_module = GATEmbedding(num_nodes=n_features, seq_len=window_size).to(device)

        # intra embedding module based on GNN
        # self.intra_module = GraphEmbedding(num_nodes=window_size, seq_len=n_features, num_levels=self.num_levels, device=torch.device(device))
        self.intra_module = AnomalyTransformer(win_size=self.window_size, enc_in=n_features, c_out=n_features, output_attention=False, attn_mode=0).to(device)

        # projection head
        #self.proj_head_inter = ProjectionLayer(n_feature=self.f_dim, num_heads=1, dropout=dropout)
        self.proj_head_inter = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(),
            nn.Linear(self.f_dim, self.f_dim)
        )

        # projection head
        #self.proj_head_intra = ProjectionLayer(n_feature=window_size, num_heads=1, dropout=dropout)
        self.proj_head_intra = nn.Sequential(
            nn.Linear(window_size, window_size),
            nn.ReLU(),
            nn.Linear(window_size, window_size)
        )

        # projection head
        self.proj_head3 = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(),
            nn.Linear(self.f_dim, self.f_dim)
        )

        self.fusion_linear = nn.Linear(self.f_dim*2, self.f_dim)

        self.inter_latent = None 
        self.intra_latent = None
        self.fused_latent = None

        # decoder
        if self.decoder_type == 0:
            self.decoder = EncoderLayer(n_feature=self.f_dim, num_heads=1, hid_dim=self.f_dim, dropout=dropout)
        elif self.decoder_type == 1:
            self.decoder = ReconstructionModel(in_dim=self.f_dim, hid_dim=100, out_dim=self.f_dim, n_layers=1, dropout=dropout)
        self.linear = nn.Linear(self.f_dim, self.out_dim)

        # TODO: ADD!
        # GAN特征增强器
        self.gan_augment = GANFeatureAugmenter(
            window_size=window_size,
            num_features=n_features,
            lr=0.0002,
            device=device
        )
        
        # GAN控制参数
        self.use_gan = True  # 是否使用GAN增强
        self.gan_train_freq = 5  # 每n批次训练一次GAN
        self.gan_warmup_epochs = 0  # GAN训练前的预热轮数


    def aug_feature1(self, input_feat, drop_dim = 2, drop_percent = 0.1):
        aug_input_feat = copy.deepcopy(input_feat)
        drop_dim = random.randint(1, 2)
        total_num = aug_input_feat.shape[drop_dim]
        drop_feat_num = int(total_num * drop_percent)
        drop_idx = random.sample([i for i in range(total_num)], drop_feat_num)
        #drop_dim = np.random.randint(1, 3)
        #print('drop_dim:', drop_dim)
        if drop_dim == 1:
            aug_input_feat[:, drop_idx, :] = 0.
        elif drop_dim == 2:
            aug_input_feat[:, :, drop_idx] = 0.

        return aug_input_feat


    def aug_feature(self, input_feat, drop_dim = 2, drop_percent = 0.1):
        aug_input_feat = copy.deepcopy(input_feat.detach())
        drop_dim = random.randint(1, 2)
        total_num = aug_input_feat.shape[drop_dim]
        drop_feat_num = int(total_num * drop_percent)
        # mask 
        if drop_dim == 1:
            ind = total_num - drop_feat_num
            aug_input_feat[:, ind:, :] = 0.
        elif drop_dim == 2:
            drop_idx = random.sample([i for i in range(total_num)], drop_feat_num)
            aug_input_feat[:, :, drop_idx] = 0.

        return aug_input_feat


    def aug_feature2(self, input_feat, drop_dim = 2, drop_percent = 0.1):
        aug_input_feat = copy.deepcopy(input_feat)
        drop_dim = random.randint(1, 2)
        total_num = aug_input_feat.shape[drop_dim]
        drop_feat_num = int(total_num * drop_percent)
        # mask 
        if drop_dim == 1:
            ind = total_num - drop_feat_num
            aug_input_feat[:, ind:, :] = 0.
        elif drop_dim == 2:
            drop_idx = random.sample([i for i in range(total_num)], drop_feat_num)
            aug_input_feat[:, :, drop_idx] = 0.

        return aug_input_feat


    def aug_feature3(self, input_feat, drop_dim = 2, drop_percent = 0.1):
        aug_input_feat = copy.deepcopy(input_feat)
        drop_dim = 1#random.randint(1, 2)
        total_num = aug_input_feat.shape[drop_dim]
        drop_feat_num = int(total_num * drop_percent)
        # mask 
        if drop_dim == 1:
            ind = total_num - drop_feat_num
            aug_input_feat[:, ind:, :] = 0.
        elif drop_dim == 2:
            p = self.param(aug_input_feat)
            sp = F.gumbel_softmax(p, hard=True)
            aug_input_feat = sp*input_feat

        return aug_input_feat


    # def loss_cl_s(self, z1, z2):
    #     batch_size, w, k = z1.size()
    #     T = 0.5
    #     x1 = z1.contiguous().view(batch_size, -1)
    #     x2 = z2.contiguous().view(batch_size, -1).detach()
        
    #     x1_abs = x1.norm(dim=1)
    #     x2_abs = x2.norm(dim=1)

    #     sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    #     sim_matrix = torch.exp(sim_matrix / T)
    #     pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    #     loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    #     loss = - torch.log(loss).mean()

    #     return loss

    def timelag_sigmoid(self, T, sigma=1):
        dist = np.arange(T)
        dist = np.abs(dist - dist[:, np.newaxis])
        matrix = 2 / (1 + np.exp(sigma * dist))
        matrix = np.where(matrix < 1e-6, 0, matrix)
        return matrix
    
    def dup_matrix(self, matrix):
        mat0 = torch.tril(matrix, diagonal=-1)[:,:-1]
        mat0 += torch.triu(matrix, diagonal=1)[:,1:]
        mat1 = torch.cat([mat0, matrix], dim=1)
        mat2 = torch.cat([matrix, mat0], dim=1)
        return mat1, mat2

    def loss_cl_s_change(self, z1, z2):
        timelag = self.timelag_sigmoid(z1.shape[1])
        timelag = torch.tensor(timelag, device=self.device)
        timelag_L, timelag_R = self.dup_matrix(timelag)
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=1)
        sim = torch.matmul(z, z.transpose(1, 2))
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        t = torch.arange(T, device=self.device)
        loss = torch.sum(logits[:, t]*timelag_L)
        loss += torch.sum(logits[:, T + t]*timelag_R)
        loss /= (2*B*T)
        return loss

    def loss_cl_s(self, z1, z2, z_neg=None):
        batch_size, w, k = z1.size()
        T = 0.5
        x1 = z1.contiguous().view(batch_size, -1)
        x2 = z2.contiguous().view(batch_size, -1).detach()
        
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        
        # Calculate positive similarity
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        
        # If negative samples are provided, incorporate them
        if z_neg is not None:
            x_neg = z_neg.contiguous().view(batch_size, -1).detach()
            x_neg_abs = x_neg.norm(dim=1)
            
            # Calculate negative similarity
            neg_sim_matrix = torch.einsum('ik,jk->ij', x1, x_neg) / torch.einsum('i,j->ij', x1_abs, x_neg_abs)
            neg_sim_matrix = torch.exp(neg_sim_matrix / T)
            
            # Total similarity is positive + negative
            denominator = sim_matrix.sum(dim=1) + neg_sim_matrix.sum(dim=1) - pos_sim
        else:
            denominator = sim_matrix.sum(dim=1) - pos_sim
        
        loss = pos_sim / denominator
        loss = - torch.log(loss).mean()
        
        return loss


    def loss_cl(self, z1, z2):
        batch_size, w, k = z1.size()
        T = 0.5
        x1 = z1.contiguous().view(batch_size, -1)
        x2 = z2.contiguous().view(batch_size, -1)
        
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_a = torch.exp(sim_matrix_a / T)
        pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
        loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
        loss_a = - torch.log(loss_a).mean()

        sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
        sim_matrix_b = torch.exp(sim_matrix_b / T)
        pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
        loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
        loss_b = - torch.log(loss_b).mean()

        loss = (loss_a + loss_b) / 2
        return loss


    def forward(self, x, x_aug_neg=None, training=True):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # print('x:', x.shape)
        x = self.conv(x)
        # if training and x_aug is not None:
        #     x_aug = self.conv(x_aug)
        # else:
        #     x_aug = x

        if training:
            x_aug = self.gan_augment(x, training=True)
        else:
            x_aug = x
        
        # intra graph
        if self.use_intra_graph:
            # enc_intra = self.intra_module(x.permute(0, 2, 1))   # >> (b, k, n)
            enc_intra = self.intra_module(x).permute(0, 2, 1)   # >> (b, k, n)

            # projection head
            enc_intra = self.proj_head_intra(enc_intra).permute(0, 2, 1)

        # inter graph
        if self.use_inter_graph:
            enc_inter = self.inter_module(x)  # >> (b, n, k)

            # projection head
            enc_inter = self.proj_head_inter(enc_inter)

        if training and self.use_contrastive:
    
            if self.use_intra_graph:
                # intra aug
                # enc_intra_aug = self.intra_module(x_aug.permute(0, 2, 1))  # >> (b, k, n)
                enc_intra_aug = self.intra_module(x_aug).permute(0, 2, 1)
                # projection head
                enc_intra_aug = self.proj_head_intra(enc_intra_aug).permute(0, 2, 1)
                # contrastive loss
                loss_intra_in = self.loss_cl_s_change(enc_intra, enc_intra_aug)

            if self.use_inter_graph:
                # inter aug
                enc_inter_aug = self.inter_module(x_aug)
                # projection head
                enc_inter_aug = self.proj_head_inter(enc_inter_aug)
                # contrastive loss
                loss_inter_in = self.loss_cl_s(enc_inter, enc_inter_aug)

            # if x_aug_neg is not None:
            #     x_aug_neg = self.conv(x_aug_neg)
            #     enc_inter_aug_neg = self.inter_module(x_aug_neg)
            #     enc_inter_aug_neg = self.proj_head_inter(enc_inter_aug_neg)

            #     enc_intra_aug_neg = self.intra_module(x_aug_neg).permute(0, 2, 1)
            #     enc_intra_aug_neg = self.proj_head_intra(enc_intra_aug_neg).permute(0, 2, 1)
            #     loss_intra_in = self.loss_cl_s(enc_intra, enc_intra_aug, enc_intra_aug_neg)
            #     loss_inter_in = self.loss_cl_s(enc_inter, enc_inter_aug, enc_inter_aug_neg)
            # else:
            #     loss_intra_in = self.loss_cl_s(enc_intra, enc_intra_aug)
            #     loss_inter_in = self.loss_cl_s(enc_inter, enc_inter_aug)

        # projection head 3
        # enc_intra = self.proj_head3(enc_intra)
        # enc_inter = self.proj_head3(enc_inter)

        # contrastive loss
        if training and self.use_contrastive:
            loss_cl = 0
            if self.use_intra_graph:
                loss_cl += loss_intra_in
            if self.use_inter_graph:
                loss_cl += loss_inter_in
            if self.use_cross_loss and self.use_intra_graph and self.use_inter_graph:
                loss_cross = self.loss_cl(enc_intra, enc_inter)
                loss_cl += loss_cross
        else:
            loss_cl = torch.zeros([1]).cuda()

        # fuse
        if self.use_intra_graph and self.use_inter_graph:
            enc = torch.cat([enc_inter, enc_intra], dim=-1)
            enc = self.fusion_linear(enc)
            # =============
            self.inter_latent = enc_inter
            self.intra_latent = enc_intra
            self.fused_latent = enc 
            # =============
        elif self.use_intra_graph:
            enc = enc_intra
        elif self.use_inter_graph:
            enc = enc_inter

        # decoder
        if self.decoder_type == 0:
            dec, _ = self.decoder(enc)
        elif self.decoder_type == 1:
            dec = self.decoder(enc)
        out = self.linear(dec)

        return out, loss_cl


