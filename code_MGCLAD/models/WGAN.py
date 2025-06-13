import torch
import torch.nn as nn
import torch.autograd as autograd

# 梯度惩罚函数
def compute_gradient_penalty(D, real_samples, fake_samples):
    """计算梯度惩罚"""
    alpha = torch.rand(real_samples.size(0), 1, 1)
    alpha = alpha.expand(real_samples.size())
    alpha = alpha.to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False).to(real_samples.device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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