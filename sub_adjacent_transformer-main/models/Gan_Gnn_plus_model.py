import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch.nn.functional as F
from models.transformer import EncoderLayer
from models.modules import ConvLayer
from models.graph import GraphEmbedding
from models.gnn_plus import create_gnn_plus_complete_model
# from models.AnomalyGraph import DynamicGraphEmbedding
from model.AnomalyTransformer import AnomalyTransformer
from models.contrastive_loss import local_infoNCE, global_infoNCE
from models.PHilayer import PHilayer


class MaskGenerator(nn.Module):
    """具有自动学习增强强度的生成器，能根据数据特征自适应调整各增强方式的强度"""
    
    def __init__(self, window_size, n_features, hidden_dim=128, noise_dim=64):
        super(MaskGenerator, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.noise_dim = noise_dim
        
        # 基础生成器网络 - 生成各种变换的参数
        self.base_net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2)
        )
        
        # 各种增强方式的参数生成器（同之前）
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, window_size * n_features),
            nn.Sigmoid()
        )
        
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, window_size * n_features),
            nn.Softplus()
        )
        
        self.shift_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, n_features),
            nn.Tanh()
        )
        
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, n_features),
            nn.Softplus()
        )
        
        self.mix_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # 可学习的强度参数 - 初始值设为0.5（中等强度）
        self.mask_strength = nn.Parameter(torch.tensor(0.5))
        self.noise_strength = nn.Parameter(torch.tensor(0.5))
        self.shift_strength = nn.Parameter(torch.tensor(0.5))
        self.scale_strength = nn.Parameter(torch.tensor(0.5))
        self.mix_strength = nn.Parameter(torch.tensor(0.5))
        
        # 强度参数的温度系数，控制学习速度和强度范围
        self.strength_temp = 1.0
        
    def get_normalized_strengths(self):
        """将强度参数归一化到[0, 1]范围"""
        # 使用sigmoid确保输出在[0,1]之间
        mask_strength = torch.sigmoid(self.mask_strength * self.strength_temp)
        noise_strength = torch.sigmoid(self.noise_strength * self.strength_temp)
        scale_strength = torch.sigmoid(self.scale_strength * self.strength_temp)
        # shift+mix = 1
        shift_strength = torch.sigmoid(self.shift_strength * self.strength_temp)
        mix_strength = 1.0 - shift_strength
        
        return {
            'mask': mask_strength,
            'noise': noise_strength,
            'shift': shift_strength,
            'scale': scale_strength,
            'mix': mix_strength
        }
    
    def forward(self, x, batch_size, device):
        """生成多种增强策略的组合"""
        strengths = self.get_normalized_strengths()
        z = torch.randn(batch_size, self.noise_dim, device=device)
        base_features = self.base_net(z)
        
        # 1. 掩码增强
        mask = self.mask_head(base_features).view(batch_size, self.window_size, self.n_features)
        mask = 1.0 - (1.0 - mask) * strengths['mask'] * 0.3  # 限制掩码强度
        
        # 2. 噪声增强
        noise_magnitude = self.noise_head(base_features).view(batch_size, self.window_size, self.n_features)
        noise = torch.randn_like(x) * noise_magnitude * strengths['noise'] * 0.05
        
        # 3. 时间扭曲（改进版）
        time_warp_strength = strengths['shift']
        warped_x = self._apply_time_warp(x, time_warp_strength)
        
        # 4. 幅度缩放
        scale_factors = self.scale_head(base_features).unsqueeze(1)
        scale_factors = 1.0 + (scale_factors - 0.5) * 0.2 * strengths['scale']
        
        # 5. 频域增强（改进版）
        freq_aug_x = self._apply_frequency_augmentation(x, strengths['mix'])
        
        # 安全的组合多种增强
        # 确保所有权重之和为1
        w1 = 1 - strengths['mix'] - strengths['shift']
        w1 = torch.clamp(w1, 0.1, 0.8)  # 限制权重范围
        w2 = strengths['mix'] * 0.5
        w3 = strengths['shift'] * 0.5
        
        # 标准化权重
        total_weight = w1 + w2 + w3
        w1, w2, w3 = w1/total_weight, w2/total_weight, w3/total_weight
        
        # 组合增强
        aug_x = x * mask * scale_factors + noise
        aug_x = aug_x * w1 + freq_aug_x * w2 + warped_x * w3
        
        return aug_x, strengths
    
    def _apply_time_warp(self, x, strength):
        """时间扭曲增强 - 修复版本"""
        batch_size, seq_len, n_features = x.shape
        warp_steps = int(seq_len * 0.1 * strength)
        
        if warp_steps == 0:
            return x
            
        warped_x = x.clone()
        for i in range(batch_size):
            # 随机选择扭曲点，确保有足够的空间进行位移
            warp_point = torch.randint(warp_steps, seq_len - warp_steps, (1,)).item()
            shift = torch.randint(-warp_steps, warp_steps + 1, (1,)).item()
            
            if shift > 0:
                # 正向位移：将后面的数据向前移动
                if warp_point + shift < seq_len:
                    remaining_len = seq_len - warp_point - shift
                    warped_x[i, warp_point:warp_point + remaining_len] = x[i, warp_point + shift:seq_len]
                    # 用最后一个值填充剩余部分
                    if warp_point + remaining_len < seq_len:
                        warped_x[i, warp_point + remaining_len:] = x[i, seq_len-1:seq_len].repeat(seq_len - warp_point - remaining_len, 1)
            elif shift < 0:
                # 负向位移：将前面的数据向后移动
                abs_shift = abs(shift)
                if warp_point - abs_shift >= 0:
                    source_len = seq_len - warp_point
                    target_len = seq_len - warp_point + abs_shift
                    # 确保不超出边界
                    copy_len = min(source_len, target_len, seq_len - (warp_point - abs_shift))
                    warped_x[i, warp_point - abs_shift:warp_point - abs_shift + copy_len] = x[i, warp_point:warp_point + copy_len]
                    
        return warped_x
    
    def _apply_frequency_augmentation(self, x, strength):
        """频域增强 - 修复版本"""
        if strength < 0.01:  # 如果强度太小，直接返回原数据
            return x
            
        # 简单的频域滤波
        fft_x = torch.fft.fft(x, dim=1)
        
        # 随机滤波，限制强度范围
        filter_ratio = min(strength * 0.1, 0.5)  # 限制最大滤波比例
        filter_mask = torch.rand_like(fft_x.real) > filter_ratio
        fft_x = fft_x * filter_mask.to(fft_x.dtype)
        
        result = torch.fft.ifft(fft_x, dim=1).real
        return result


# 辅助损失函数：引导强度参数的学习
def strength_guidance_loss(generator, real_data, fake_data, discriminator, alpha=0.1):
    """
    强度引导损失，鼓励生成器找到最佳增强强度
    
    参数:
        generator: 生成器实例
        real_data: 真实数据
        fake_data: 生成器产生的增强数据
        discriminator: 判别器
        alpha: 平衡系数
    """
    # 1. 真实性损失：希望增强数据既不能太像真实数据（否则无增强效果）
    # 也不能太不像（否则会破坏数据分布）
    real_pred = discriminator(real_data)
    fake_pred = discriminator(fake_data)
    
    # 理想情况下，增强数据的判别值应略低于真实数据
    authenticity_loss = F.mse_loss(fake_pred, 0.8 * real_pred)
    
    # 2. 多样性损失：鼓励不同增强方式有差异化的强度
    strengths = generator.get_normalized_strengths()
    strength_values = torch.stack(list(strengths.values()))
    
    # 希望强度参数不要都趋向相同值，增加多样性
    diversity_loss = alpha * torch.var(strength_values)
    
    return authenticity_loss + diversity_loss
    

class Discriminator(nn.Module):
    """改进的时间序列判别器"""
    
    def __init__(self, window_size, n_features, hidden_dim=128):
        super(Discriminator, self).__init__()
        
        # 1D卷积层用于时间序列特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(window_size // 4)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * (window_size // 4), hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 光谱归一化用于稳定训练
        self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        """应用光谱归一化"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.utils.spectral_norm(module)
    
    def forward(self, x):
        # x: (batch_size, window_size, n_features)
        x = x.permute(0, 2, 1)  # (batch_size, n_features, window_size)
        
        conv_out = self.conv_layers(x)
        conv_out = conv_out.flatten(1)
        
        return self.fc_layers(conv_out)

# class Discriminator(nn.Module):
#     """Wasserstein GAN-GP discriminator for evaluating augmented samples"""

#     def __init__(self, window_size, n_features, hidden_dim=128):
#         super(Discriminator, self).__init__()
        
#         self.model = nn.Sequential(
#             nn.Linear(window_size * n_features, hidden_dim * 2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim, 1)  # No sigmoid for Wasserstein GAN
#         )
    
#     def forward(self, x):
#         batch_size = x.size(0)
#         x_flat = x.view(batch_size, -1)
#         return self.model(x_flat)


# Gradient penalty function for WGAN-GP
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP"""
    # Random weight term for interpolation
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    # Interpolated samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Calculate discriminator output for interpolated samples
    d_interpolated = discriminator(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    # gradients = gradients.view(gradients.size(0), -1)
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

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
        self.use_contrastive = True
        self.use_intra_graph = True  
        self.use_inter_graph = True # 消融
        self.device = device

        # preprocessing
        self.conv = ConvLayer(n_features, 7)
        self.fusion_linear = nn.Linear(self.f_dim*2, self.f_dim)
        # decoder
        self.decoder = EncoderLayer(n_feature=self.f_dim, num_heads=1, hid_dim=self.f_dim, dropout=dropout)
        self.linear = nn.Linear(self.f_dim, self.out_dim)

        self.num_levels = 1
        # inter embedding module based on GNN
        self.inter_module = GraphEmbedding(num_nodes=n_features, seq_len=window_size, num_levels=self.num_levels, device=torch.device(device)) # graph
        # self.inter_module = DynamicGraphEmbedding(num_nodes=n_features, seq_len=window_size, num_levels=self.num_levels, device=torch.device(device)) # graph
        # self.inter_module = DynamicGraphEmbedding(num_nodes=n_features, seq_len=window_size, num_levels=self.num_levels, device=torch.device(device), lambda_val=1.0).to(device) # anomalygraph
        # self.inter_module = None
        # self.inter_module = DynamicGraphEmbedding(in_channels=self.window_size, out_channels=self.window_size, num_nodes=n_features, topk=20, heads=2, concat=False, dropout=0.1, lambda_val=1.0).to('cuda') # anomalygraph

        
        # config = { # gcn
        #     'num_nodes': n_features,
        #     'seq_len': window_size,
        #     'device': torch.device(device),
        #     'mp_type': 'gcn',
        #     'num_layers': self.num_levels,
        #     'hidden_dim': window_size * 2,
        #     'dropout': 0.1,
        #     'use_positional_encoding': True,
        #     'use_edge_features': True,
        #     'use_dynamic_edges': True
        # }
        # config = { # gat
        #     'num_nodes': n_features,
        #     'seq_len': window_size,
        #     'device': torch.device(device),
        #     'mp_type': 'gat',
        #     'num_layers': 1,
        #     'hidden_dim': window_size * 2,
        #     'heads': 8,
        #     'dropout': 0.1,
        #     'use_positional_encoding': True,
        #     'use_edge_features': False,  # GAT自带注意力机制
        #     'use_dynamic_edges': False
        # }
        # config = { # sage
        #     'num_nodes': n_features,
        #     'seq_len': window_size,
        #     'device': torch.device(device),
        #     'mp_type': 'sage',
        #     'num_layers': 2,
        #     'hidden_dim': window_size * 2,
        #     'dropout': 0.15,
        #     'use_positional_encoding': True,
        #     'use_edge_features': True,
        #     'use_dynamic_edges': True
        # }

        # self.inter_module = create_gnn_plus_complete_model(config).to(device)

        # intra embedding module based on GNN
        # self.intra_module = GraphEmbedding(num_nodes=window_size, seq_len=n_features, num_levels=self.num_levels, device=torch.device(device))
        self.intra_module = AnomalyTransformer(win_size=self.window_size, enc_in=n_features, c_out=n_features, e_layers=3, linear_attn=True).to(device)

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


        # TODO: ADD!
        # WGAN-GP components for augmentation
        self.mask_generator = MaskGenerator(window_size, n_features)
        self.discriminator = Discriminator(window_size, n_features)
        self.lambda_gp = 10  # Gradient penalty coefficient
        
        # WGAN-GP training params
        self.n_critic = 5  # Number of discriminator updates per generator update
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.wgan_initialized = False

        # Training phase control
        self.training_phase = 'warmup'  # 'warmup' -> 'gan_pretrain' -> 'alternate'
        self.warmup_epochs = 1
        self.gan_pretrain_epochs = 1
        self.recon_loss_weight = 0.1  # Weight for reconstruction loss in generator
        self.philayer = PHilayer(hidden_dim=self.f_dim, output_dim=self.f_dim*4).to(device)

    def init_wgan_optimizers(self):
        """Initialize optimizers for WGAN-GP"""
        if not self.wgan_initialized:
            # Using RMSprop as recommended for WGAN
            self.generator_optimizer = torch.optim.RMSprop(
                self.mask_generator.parameters(), 
                lr=0.0001
            )
            self.discriminator_optimizer = torch.optim.RMSprop(
                self.discriminator.parameters(),
                lr=0.00005
            )
            self.wgan_initialized = True

    def set_training_phase(self, phase):
        """Set training phase: 'warmup', 'gan_pretrain', or 'alternate'"""
        self.training_phase = phase
        print(f"Training phase set to: {phase}")

    def freeze_main_model(self):
        """Freeze main model parameters except GAN components"""
        gan_param_ids = set()
        for param in self.mask_generator.parameters():
            gan_param_ids.add(id(param))
        for param in self.discriminator.parameters():
            gan_param_ids.add(id(param))
        
        for param in self.parameters():
            if id(param) not in gan_param_ids:
                param.requires_grad = False

    def unfreeze_main_model(self):
        """Unfreeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_gan_model(self):
        """Freeze GAN parameters"""
        for param in self.mask_generator.parameters():
            param.requires_grad = False
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def unfreeze_gan_model(self):
        """Unfreeze GAN parameters"""
        for param in self.mask_generator.parameters():
            param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def extract_features_for_recon(self, data_input):
        """Extract features for reconstruction loss calculation"""
        features = self.conv(data_input)
        if self.use_inter_graph:
            inter_features = self.inter_module(features)
            projected_features = self.proj_head_inter(data_input + inter_features)
        else:
            projected_features = features
        
        if self.use_intra_graph:
            # final_features, _, _ = self.intra_module(projected_features)
            intra_features, _, _ = self.intra_module(features)
            intra_features = intra_features.permute(0, 2, 1)  # (b, n, k)
            intra_features = self.proj_head_intra(intra_features).permute(0, 2, 1)  # (b, k, n)
            # Apply PHI layer for feature transformation
            intra_features, kl_loss, _ = self.philayer(intra_features)  # (b, k, n)
            if self.use_inter_graph:
                final_features = self.fusion_linear(torch.cat([projected_features, intra_features], dim=-1))  # (b, n, k)
            else:
                final_features = intra_features
        else:
            final_features = projected_features
        final_features, _ = self.decoder(final_features)  # (b, n, k)
        final_features = self.linear(final_features)  # (b, n, out_dim)
        
        return final_features

    def update_wgan_with_recon(self, x):
        """Updated WGAN training with reconstruction loss"""
        if not self.wgan_initialized:
            self.init_wgan_optimizers()
        
        batch_size = x.size(0)
        real_data = x.detach()
        
        # Train discriminator
        for _ in range(self.n_critic):
            self.discriminator_optimizer.zero_grad()
            
            with torch.no_grad():
                real_features = self.extract_features_for_recon(real_data)
                
                fake_data, strengths = self.mask_generator(real_data, batch_size, self.device)
                # fake_data = real_data * masks
                fake_features = self.extract_features_for_recon(fake_data)
            
            real_validity = self.discriminator(real_features.detach())
            fake_validity = self.discriminator(fake_features.detach())
            
            gradient_penalty = compute_gradient_penalty(
                self.discriminator, real_features.detach(), fake_features.detach(), self.device
            )
            
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + self.lambda_gp * gradient_penalty
            d_loss.backward()
            self.discriminator_optimizer.step()
        
        # Train generator with both adversarial and reconstruction loss
        self.generator_optimizer.zero_grad()
        
        fake_data_for_g, strengths = self.mask_generator(real_data, batch_size, self.device)
        # fake_data_for_g = real_data * masks
        
        # Extract features with gradients enabled for generator
        fake_features_for_g = self.extract_features_for_recon(fake_data_for_g)
        
        # Adversarial loss
        fake_validity = self.discriminator(fake_features_for_g)
        adversarial_loss = -torch.mean(fake_validity)
        
        # Reconstruction loss
        # real_features_for_recon = self.extract_features_for_recon(real_data)
        recon_loss = F.mse_loss(fake_features_for_g, real_data.detach())
        
        # Combined generator loss (adversarial loss as primary, reconstruction as auxiliary)
        g_loss = adversarial_loss + self.recon_loss_weight * recon_loss + 0.1 * strength_guidance_loss(
            self.mask_generator, real_data, fake_data_for_g, self.discriminator
        )
        
        g_loss.backward()
        self.generator_optimizer.step()
        
        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'adversarial_loss': adversarial_loss.item(),
            'reconstruction_loss': recon_loss.item()
        }
    
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

    def aug_feature_adaptive(self, input_feat, drop_percent=0.1, epoch=0):
        """Adaptive feature augmentation based on training phase"""
        if self.training_phase == 'warmup':
            return self.aug_feature(input_feat, drop_percent)
        elif self.training_phase in ['gan_pretrain', 'alternate']:
            return self.aug_feature_wgan(input_feat, drop_percent, epoch)
        else:
            return input_feat

    def aug_feature_wgan(self, input_feat, drop_percent=0.1, epoch=0):
        """WGAN-based feature augmentation"""
        batch_size = input_feat.size(0)
        
        with torch.no_grad():
            aug_input_feat, _ = self.mask_generator(input_feat, batch_size, self.device)
        
        return aug_input_feat

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
            x_aug = self.aug_feature_adaptive(x)
            # x_aug = self.aug_feature(x)
        else:
            x_aug = x

        # intra graph
        if self.use_intra_graph:
            # enc_intra = self.intra_module(x.permute(0, 2, 1))   # >> (b, k, n)
            enc_intra, queries_list_intra, keys_list_intra = self.intra_module(x)   # >> (b, k, n)
            enc_intra = enc_intra.permute(0, 2, 1)   # >> (b, n, k)

            # projection head
            enc_intra = self.proj_head_intra(enc_intra).permute(0, 2, 1)
            enc_intra,kl_loss, _  = self.philayer(enc_intra)  # Apply PHI layer for feature transformation

        # inter graph
        if self.use_inter_graph:
            enc_inter = self.inter_module(x)  # >> (b, n, k)
            # enc_inter, queries_list_inter, keys_list_inter = self.inter_module(x)

            # projection head
            enc_inter = self.proj_head_inter(enc_inter)

        if training and self.use_contrastive:
    
            if self.use_intra_graph:
                # intra aug
                # enc_intra_aug = self.intra_module(x_aug.permute(0, 2, 1))  # >> (b, k, n)
                enc_intra_aug, queries_list_aug, keys_list_aug = self.intra_module(x_aug)

                # for i in range(len(queries_list_aug)):
                #     queries_list_intra[i] = queries_list_intra[i] + queries_list_aug[i]
                #     keys_list_intra[i] = keys_list_intra[i] + keys_list_aug[i]
                #     # print("queries_list_intra[0].shape:", queries_list_intra[0].shape)
                #     # print("keys_list_intra[0].shape:", keys_list_intra[0].shape)
                #     # print(len(queries_list_aug))
                    
                enc_intra_aug = enc_intra_aug.permute(0, 2, 1)   # >> (b, n, k)
                # projection head
                enc_intra_aug = self.proj_head_intra(enc_intra_aug).permute(0, 2, 1)
                enc_intra_aug, _, _  = self.philayer(enc_intra_aug)  # Apply PHI layer for feature transformation
                # contrastive loss
                loss_intra_in = self.loss_cl_s_change(enc_intra, enc_intra_aug)

            if self.use_inter_graph:
                # inter aug
                enc_inter_aug = self.inter_module(x_aug)
                # enc_inter_aug, queries_list_inter_aug, keys_list_inter_aug = self.inter_module(x_aug)

                # for i in range(len(queries_list_inter_aug)):
                #     queries_list_inter[i] = queries_list_inter[i] + queries_list_inter_aug[i]
                #     keys_list_inter[i] = keys_list_inter[i] + keys_list_inter_aug[i]
                #     # print("queries_list_inter[0].shape:", queries_list_inter[0].shape)
                #     # print("keys_list_inter[0].shape:", keys_list_inter[0].shape)
                #     # print(len(queries_list_inter_aug))
                #     # exit()

                # projection head
                enc_inter_aug = self.proj_head_inter(enc_inter_aug)
                # contrastive loss
                loss_inter_in = local_infoNCE(enc_inter, enc_inter_aug, k=16)*0.5 + global_infoNCE(enc_inter, enc_inter_aug)

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
                # loss_cl += loss_intra_in
                # loss_cl += kl_loss
                loss_intra_in = loss_intra_in
                loss_intra_in += 0.5 * kl_loss
            if self.use_inter_graph:
                # loss_cl += loss_inter_in
                loss_inter_in = loss_inter_in
            if self.use_intra_graph and self.use_inter_graph:
                loss_cross = self.loss_cl(enc_intra, enc_inter)
                loss_cl += loss_cross
        else:
            loss_cl = torch.zeros([1]).cuda()
            loss_intra_in = torch.zeros([1]).cuda()
            loss_inter_in = torch.zeros([1]).cuda()

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
        dec, _ = self.decoder(enc)
        out = self.linear(dec)

        # # Combine queries and keys from both branches for anomaly detection
        # if self.use_intra_graph and self.use_inter_graph:
        #     # Combine attention from both branches
        #     combined_queries = queries_list_intra + queries_list_inter
        #     combined_keys = keys_list_intra + keys_list_inter
        #     return out, loss_cl, combined_queries, combined_keys
        # elif self.use_intra_graph:
        #     return out, loss_cl, queries_list_intra, keys_list_intra
        # elif self.use_inter_graph:
        #     return out, loss_cl, queries_list_inter, keys_list_inter
        # else:
        #     return out, loss_cl, [], []
        return out, loss_cl, queries_list_intra, keys_list_intra, loss_intra_in, loss_inter_in
        # return out, loss_cl+loss_intra_in+loss_inter_in, queries_list_intra, keys_list_intra