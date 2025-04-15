from models.graph import GraphEmbedding
from transformer_models.AnomalyTransformer import AnomalyTransformer
import torch
import torch.nn as nn
import json

print('CUDA版本:',torch.version.cuda)
print('Pytorch版本:',torch.__version__)
print('显卡是否可用:','可用' if(torch.cuda.is_available()) else '不可用')
print('显卡数量:',torch.cuda.device_count())
print('是否支持BF16数字格式:','支持' if (torch.cuda.is_bf16_supported()) else '不支持')
print('当前显卡型号:',torch.cuda.get_device_name())
print('当前显卡的CUDA算力:',torch.cuda.get_device_capability())
print('当前显卡的总显存:',torch.cuda.get_device_properties(0).total_memory/1024/1024/1024,'GB')
print('是否支持TensorCore:','支持' if (torch.cuda.get_device_properties(0).major >= 7) else '不支持')
print('当前显卡的显存使用率:',torch.cuda.memory_allocated(0)/torch.cuda.get_device_properties(0).total_memory*100,'%')

# x = torch.randn(512, 32, 17).to("cuda:0") # [batch_size, seq_len, num_features]
# # x_aug = x.permute(0, 2, 1)  # [batch_size, num_features, seq_len]
# gate_layer_intra = nn.Sequential(
#             nn.Linear(32*2, 32),
#             nn.Sigmoid()
# ).to("cuda:0")
# x = x.permute(0, 2, 1)  # [batch_size, num_features, seq_len]
# x = gate_layer_intra(torch.cat([x, x], dim=-1))
# print('x:', x.shape)
# # # model = GraphEmbedding(num_nodes=17, seq_len=32, num_levels=1, device="cuda:0").to("cuda:0")
# # model = AnomalyTransformer(win_size=32, enc_in=x.shape[-1], c_out=x.shape[-1],e_layers=3).to("cuda:0")
# # outputs, _, _ = model(x)
# # print(outputs.shape)

# x = x.mean(dim=(1,2)).view(-1, 1)
# net = nn.Sequential(
#             nn.Linear(1, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         ).to("cuda:0")
# print(net(x).shape)
# tp = 27559
# tn = 672189
# fp = 1909
# fn = 1399
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1 = 2 * (precision * recall) / (precision + recall)
# print(f1)


