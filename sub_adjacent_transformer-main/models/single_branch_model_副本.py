import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch.nn.functional as F

from models.modules import ConvLayer
from models.graph import GraphEmbedding, DynamicGraphEmbedding
# from models.AnomalyGraph import DynamicGraphEmbedding
from model.AnomalyTransformer import AnomalyTransformer
from models.contrastive_loss import local_infoNCE, global_infoNCE



# Enhanced MaskGenerator for Wasserstein GAN-GP
class MaskGenerator(nn.Module):
    """Wasserstein GAN-GP based mask generator for feature augmentation"""
    
    def __init__(self, window_size, n_features, hidden_dim=128, noise_dim=64):
        super(MaskGenerator, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.noise_dim = noise_dim
        
        # Generator architecture
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, window_size * n_features),
            nn.Sigmoid()  # Output range [0, 1] for multiplicative masking
        )
        
    def forward(self, batch_size, device):
        # Generate random noise
        z = torch.randn(batch_size, self.noise_dim, device=device)
        
        # Generate mask
        mask = self.generator(z)
        mask = mask.view(batch_size, self.window_size, self.n_features)
        
        # Apply threshold to create binary mask with some randomness
        # We don't use hard threshold to keep gradients flowing
        return mask


class Discriminator(nn.Module):
    """Wasserstein GAN-GP discriminator for evaluating augmented samples"""

    def __init__(self, window_size, n_features, hidden_dim=128):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(window_size * n_features, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)  # No sigmoid for Wasserstein GAN
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.model(x_flat)


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
    gradients = gradients.view(gradients.size(0), -1)
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
        self.use_contrastive = False
        self.use_intra_graph = True  
        self.use_inter_graph = True # 消融
        self.device = device

        # preprocessing
        self.conv = ConvLayer(n_features, 7)

        self.num_levels = 1
        # inter embedding module based on GNN
        self.inter_module = GraphEmbedding(num_nodes=n_features, seq_len=window_size, num_levels=self.num_levels, device=torch.device(device)) # graph
        # self.inter_module = DynamicGraphEmbedding(num_nodes=n_features, seq_len=window_size, num_levels=self.num_levels, device=torch.device(device)) # graph
        # self.inter_module = DynamicGraphEmbedding(num_nodes=n_features, seq_len=window_size, num_levels=self.num_levels, device=torch.device(device), lambda_val=1.0).to(device) # anomalygraph
        # self.inter_module = None
        # self.inter_module = DynamicGraphEmbedding(in_channels=self.window_size, out_channels=self.window_size, num_nodes=n_features, topk=20, heads=2, concat=False, dropout=0.1, lambda_val=1.0).to('cuda') # anomalygraph

        # intra embedding module based on GNN
        # self.intra_module = GraphEmbedding(num_nodes=window_size, seq_len=n_features, num_levels=self.num_levels, device=torch.device(device))
        self.intra_module = AnomalyTransformer(win_size=self.window_size, enc_in=n_features, c_out=n_features, e_layers=3, linear_attn=True).to(device)

        # projection head
        self.proj_head_inter = nn.Sequential(
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

    def update_wgan(self, x):
        """Train WGAN-GP for one step"""
        if not self.wgan_initialized:
            self.init_wgan_optimizers()
        
        batch_size = x.size(0)
        real_data = x.detach()
        
        # Generate masks and apply them to create augmentations
        for _ in range(self.n_critic):
            self.discriminator_optimizer.zero_grad()
            
            # Generate masks
            masks = self.mask_generator(batch_size, self.device)
            
            # Apply masks to input data to create fake samples
            fake_samples = x * masks
            
            # Discriminator predictions
            real_validity = self.discriminator(real_data)
            fake_validity = self.discriminator(fake_samples.detach())
            
            # Compute gradient penalty
            gradient_penalty = compute_gradient_penalty(
                self.discriminator, real_data, fake_samples, self.device
            )
            
            # Wasserstein loss for discriminator
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + self.lambda_gp * gradient_penalty
            
            d_loss.backward()
            self.discriminator_optimizer.step()
            
        
        # Train Generator
        self.generator_optimizer.zero_grad()
        
        # Generate new masks and apply them
        masks = self.mask_generator(batch_size, self.device)
        fake_samples = x * masks
        
        # Try to fool discriminator
        fake_validity = self.discriminator(fake_samples)
        
        # Wasserstein loss for generator (maximize discriminator prediction)
        g_loss = -torch.mean(fake_validity)
        
        g_loss.backward()
        self.generator_optimizer.step()
        
        # Return both losses for monitoring
        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item()
        }

    def aug_feature_wgan(self, input_feat, drop_percent=0.1, epoch=0):
        """Generate augmented feature using WGAN-GP with progressive training"""
        batch_size = input_feat.size(0)
        
        with torch.no_grad():
            # Generate binary mask
            masks = self.mask_generator(batch_size, self.device)
            
            # Progressive training - gradually decrease dropout 
            effective_dropout = max(0.05, drop_percent - (epoch * 0.01))
            dropout_mask = (torch.rand_like(masks) > effective_dropout).float()
            masks = masks * dropout_mask
            
            # Apply mask to input
            aug_input_feat = input_feat * masks
        
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

        x = self.conv(x)

        if training:
            x_aug = self.aug_feature_wgan(x)
        else:
            x_aug = x

        loss_cl = 0

        if self.use_inter_graph:
            enc_inter = self.inter_module(x)  # >> (b, n, k)
            
            if training:
                enc_inter_aug = self.inter_module(x_aug)  # >> (b, n, k)
                loss_inter_in = local_infoNCE(enc_inter, enc_inter_aug, k=16)*0.5 + global_infoNCE(enc_inter, enc_inter_aug)
                loss_cl += loss_inter_in
                enc_inter_aug = self.proj_head_inter(x_aug + enc_inter_aug)  # >> (b, n, k)

            enc_inter = self.proj_head_inter(x + enc_inter)  # >> (b, n, k)
        
        if self.use_intra_graph:
            enc_intra, queries_list, keys_list = self.intra_module(enc_inter)
            if training:
                enc_intra_aug, queries_list_aug, keys_list_aug = self.intra_module(enc_inter_aug)

                loss_intra_in = self.loss_cl_s_change(enc_intra, enc_intra_aug)
                loss_cl += loss_intra_in

                # 方案一
                for i in range(len(queries_list)):
                    queries_list[i] = queries_list[i] + queries_list_aug[i]
                    keys_list[i] = keys_list[i] + keys_list_aug[i]
                # # 方案二
                # for i in range(len(queries_list)):
                #     loss_cl += self.loss_cl(queries_list[i], queries_list_aug[i])
                #     loss_cl += self.loss_cl(keys_list[i], keys_list_aug[i])

        if self.use_contrastive:
            loss_cl += self.loss_cl(enc_intra, enc_inter)
        out = enc_intra
        return out, loss_cl, queries_list, keys_list


