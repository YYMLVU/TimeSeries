import torch
import torch.nn as nn

class PHilayer(nn.Module):
    """
    PHi层：对隐藏状态施加信息瓶颈并预测未来隐藏状态
    :param hidden_dim: 输入隐藏状态的维度（如LSTM的隐藏层维度）
    :param output_dim: 潜在变量z的维度
    """
    def __init__(self, hidden_dim, output_dim):
        super(PHilayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 编码器：将输入的隐藏状态h映射为高斯分布的均值和对数方差
        self.encoder_mean = nn.Linear(hidden_dim, output_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, output_dim)

        # 解码器：将潜在变量z映射回压缩后的隐藏状态
        self.decoder = nn.Linear(output_dim, hidden_dim)

        # 自回归先验 p_chi: 基于历史z预测当前z（使用Transformer层实现因果序列建模）
        self.prior = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=4,
                dim_feedforward=256,
                batch_first=True  # 批次维度在前
            ),
            num_layers=1
        )
        
        # 先验网络的输出层：将Transformer的输出映射为高斯分布的均值和方差
        self.prior_mean = nn.Linear(output_dim, output_dim)
        self.prior_logvar = nn.Linear(output_dim, output_dim)
        
    def reparameterize(self, mean, logvar):
        """
        重参数化技巧：从编码器输出的高斯分布中采样潜在变量z
        :return: 采样得到的潜在变量z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(self, h, z_prev=None):
        """
        前向传播：处理隐藏状态h，输出重构的h'和PHi损失
        :param h: 输入隐藏状态，形状为 (batch_size, seq_len, hidden_dim)
        :param z_prev: 历史潜在变量（用于自回归先验），形状为 (batch_size, seq_len-1, latent_dim)
        :return: h_recon: 重构的隐藏状态； phi_loss: PHi损失（KL散度）
        """
        batchsize, seq_len, _ = h.shape

        # 1. 编码：计算后验分布 q_psi(z|h) 知道当前的隐藏状态h(结果)，来计算潜在变量z的分布(因)
        # 计算均值和对数方差
        mean = self.encoder_mean(h)
        logvar = self.encoder_logvar(h)
        z = self.reparameterize(mean, logvar)  # 采样潜在变量z，形状为 (batch_size, seq_len, output_dim) 
        
        # 2. 自回归先验：计算 p_chi(z_t | z_1,...,z_{t-1}) 通过之前所有的潜在变量z{1:t-1}(历史规律),来推测当前的潜在变量z的分布(因)
        if z_prev is not None:
            # 如果提供了历史z，使用它作为先验的输入
            prior_input = z_prev
        else:
            # 否则使用当前z作为输入，但需要创建因果掩码防止未来信息泄漏
            prior_input = z
        
        # 创建因果掩码，确保每个时间步只能看到之前的信息
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(h.device)
        
        # 通过Transformer获得先验特征
        prior_features = self.prior(prior_input, mask=causal_mask)
        
        # 计算先验分布的均值和方差
        prior_mean = self.prior_mean(prior_features)
        prior_logvar = self.prior_logvar(prior_features)
        
        # 对于第一个时间步，先验应该是标准正态分布 N(0,1)
        # 这样可以避免第一个时间步没有历史信息的问题
        prior_mean[:, 0, :] = 0.0
        prior_logvar[:, 0, :] = 0.0


        # 3. 计算PHi损失：KL散度 D_KL(q_psi(z|h) || p_chi(z_t | z_1,...,z_{t-1}))
        # KL(N(mean_q, var_q) || N(mean_p, var_p)) = 0.5 * (var_q / var_p + (mean_q - mean_p)^2 / var_p - 1 + log(var_p / var_q))
        kl = 0.5 * (torch.exp(logvar) / torch.exp(prior_logvar) + 
                 (mean - prior_mean) ** 2 / torch.exp(prior_logvar) - 1 + 
                 prior_logvar - logvar).sum(dim=-1).mean()

        # 4. 解码：从z重构隐藏状态h'
        h_recon = self.decoder(z)  # 重构的隐藏状态，形状为 (batch_size, seq_len, hidden_dim)
        
        return h_recon, kl, z # 返回重构的隐藏状态、PHi损失和潜在变量z