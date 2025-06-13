import torch
import torch.nn as nn
import torch.autograd as autograd

# 梯度惩罚函数
# def compute_gradient_penalty(D, real_samples, fake_samples):
#     """计算梯度惩罚"""
#     alpha = torch.rand(real_samples.size(0), 1, 1)
#     alpha = alpha.expand(real_samples.size())
#     alpha = alpha.to(real_samples.device)
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#     d_interpolates = D(interpolates)
#     fake = torch.ones(d_interpolates.size(), requires_grad=False).to(real_samples.device)
#     gradients = autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty


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
    def __init__(self, n_features, window_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features * window_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
    
# 判别器和生成器的训练步骤
def train_wgan_gp(discriminator, generator, optimizer_D, optimizer_G, real_data, device, lambda_gp=10):
    # 训练判别器
    discriminator.train()
    generator.train()
    optimizer_D.zero_grad()

    # 生成假数据
    fake_data, _ = generator(real_data)

    # 计算判别器对真实数据和假数据的输出
    real_validity = discriminator(real_data)
    fake_validity = discriminator(fake_data)

    # 计算梯度惩罚
    gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data)

    # 计算判别器的损失
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

    # 反向传播和更新判别器的参数
    d_loss.backward()
    optimizer_D.step()

    # 训练生成器
    optimizer_G.zero_grad()

    # 生成假数据
    fake_data, _ = generator(real_data)

    # 计算判别器对假数据的输出
    fake_validity = discriminator(fake_data)

    # 计算生成器的损失
    g_loss = -torch.mean(fake_validity)

    # 反向传播和更新生成器的参数
    g_loss.backward()
    optimizer_G.step()

    return d_loss.item(), g_loss.item()


def update_wgan(discriminator, mask_generator, discriminator_optimizer, generator_optimizer, x, device='cuda:0', lambda_gp=10):
        """Train WGAN-GP for one step""" 
        batch_size = x.size(0)
        real_data = x.detach()
        
        # Generate masks and apply them to create augmentations
        for _ in range(5):
            discriminator_optimizer.zero_grad()
            
            # Generate masks
            masks = mask_generator(batch_size, device)
            
            # Apply masks to input data to create fake samples
            fake_samples = x * masks
            
            # Discriminator predictions
            real_validity = discriminator(real_data)
            fake_validity = discriminator(fake_samples.detach())
            
            # Compute gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_data, fake_samples, device
            )
            
            # Wasserstein loss for discriminator
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + lambda_gp * gradient_penalty
            
            d_loss.backward()
            discriminator_optimizer.step()
            
            # Clip weights for Lipschitz constraint (alternative to gradient penalty)
            # Comment this out if using gradient penalty
            # for p in self.discriminator.parameters():
            #     p.data.clamp_(-0.01, 0.01)
        
        # Train Generator
        generator_optimizer.zero_grad()
        
        # Generate new masks and apply them
        masks = mask_generator(batch_size, device)
        fake_samples = x * masks
        
        # Try to fool discriminator
        fake_validity = discriminator(fake_samples)
        
        # Wasserstein loss for generator (maximize discriminator prediction)
        g_loss = -torch.mean(fake_validity)
        
        g_loss.backward()
        generator_optimizer.step()
        
        # Return both losses for monitoring
        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item()
        }

def aug_feature_wgan(mask_generator, input_feat, drop_percent=0.1, epoch=0, device='cuda:0'):
    """Generate augmented feature using WGAN-GP with progressive training"""
    batch_size = input_feat.size(0)
    
    with torch.no_grad():
        # Generate binary mask
        masks = mask_generator(batch_size, device)
        
        # Progressive training - gradually decrease dropout 
        effective_dropout = max(0.05, drop_percent - (epoch * 0.01))
        dropout_mask = (torch.rand_like(masks) > effective_dropout).float()
        masks = masks * dropout_mask
        
        # Apply mask to input
        aug_input_feat = input_feat * masks
    
    return aug_input_feat