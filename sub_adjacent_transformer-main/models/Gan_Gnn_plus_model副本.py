import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch.nn.functional as F
from models.transformer import EncoderLayer
from models.modules import ConvLayer, SmoothLayer
from models.graph import GraphEmbedding # , DynamicGraphEmbedding
from models.gnn_plus import create_gnn_plus_complete_model
from models.AnomalyGraph import DynamicGraphEmbedding
from model.AnomalyTransformer import AnomalyTransformer
from models.contrastive_loss import local_infoNCE, global_infoNCE
from models.PHilayer import PHilayer



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

        # Add noise and ensure mask diversity
        mask = mask + 0.1 * torch.randn_like(mask)
        mask = torch.clamp(mask, 0.1, 0.9)  # Prevent extreme values
        
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
        #     'num_layers': 1,
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
        self.recon_loss_weight = 0.5  # Weight for reconstruction loss in generator

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
            intra_features, kl_loss, _ = self.philayer(intra_features)  # Apply PHI layer for feature transformation
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
                
                masks = self.mask_generator(batch_size, self.device)
                fake_data = real_data * masks
                fake_features = self.extract_features_for_recon(fake_data)
            
            real_validity = self.discriminator(real_features)
            fake_validity = self.discriminator(fake_features)
            
            gradient_penalty = compute_gradient_penalty(
                self.discriminator, real_features, fake_features, self.device
            )
            
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + self.lambda_gp * gradient_penalty
            d_loss.backward()
            self.discriminator_optimizer.step()
        
        # Train generator with both adversarial and reconstruction loss
        self.generator_optimizer.zero_grad()
        
        masks = self.mask_generator(batch_size, self.device)
        fake_data_for_g = real_data * masks
        
        # Extract features with gradients enabled for generator
        fake_features_for_g = self.extract_features_for_recon(fake_data_for_g)
        
        # Adversarial loss
        fake_validity = self.discriminator(fake_features_for_g)
        adversarial_loss = -torch.mean(fake_validity)
        
        # Reconstruction loss
        # real_features_for_recon = self.extract_features_for_recon(real_data)
        recon_loss = F.mse_loss(fake_features_for_g, real_data.detach())
        
        # Combined generator loss (adversarial loss as primary, reconstruction as auxiliary)
        g_loss = adversarial_loss + self.recon_loss_weight * recon_loss
        
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
            masks = self.mask_generator(batch_size, self.device)
            
            # Multiple augmentation strategies
            # if torch.rand(1) < 0.5:
            if True:
                # Multiplicative masking
                aug_input_feat = input_feat * masks
            else:
                # Additive noise
                noise = 0.1 * torch.randn_like(input_feat)
                aug_input_feat = input_feat + noise * masks
        
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
                enc_intra_aug, _, _ = self.philayer(enc_intra_aug)  # Apply PHI layer for feature transformation
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