import torch
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from transformers import AutoModelForCausalLM
import random
import copy
import os
import numpy as np
from torch.nn.functional import interpolate
import tsaug

model = AutoModelForCausalLM.from_pretrained("./weight", trust_remote_code=True).to("cuda:0")
# print(model)
# TimerForPrediction(
#   (model): TimerModel(
#     (embed_layer): TimerPatchEmbedding(
#       (emb): Linear(in_features=96, out_features=1024, bias=False)
#     )
#     (layers): ModuleList(
#       (0-7): 8 x TimerDecoderLayer(
#         (self_attn): TimerAttention(
#           (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
#           (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
#           (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
#           (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
#           (rotary_emb): TimeMoeRotaryEmbedding()
#         )
#         (ffn_layer): TimerMLP(
#           (gate_proj): Linear(in_features=1024, out_features=2048, bias=False)
#           (up_proj): Linear(in_features=1024, out_features=2048, bias=False)
#           (down_proj): Linear(in_features=2048, out_features=1024, bias=False)
#           (act_fn): SiLU()
#         )
#         (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#         (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#       )
#     )
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#   )
#   (lm_heads): ModuleList(
#     (0): Linear(in_features=1024, out_features=96, bias=False)
#   )
#   (loss_function): MSELoss()
# )


# 将单序列处理模型转换为多序列处理模型
def MultiSeqModel(model, inputs):
    # inputs: [batch_size, seq_len, num_features]
    for i in range(inputs.shape[-1]):
        # 转为二维[batch_size, seq_len]
        input = inputs[:, :, i].to("cuda:0")
        output = model.generate(input, max_new_tokens=1)
        # 拼接output，得到三维的outputs
        outputs = output.unsqueeze(2) if i == 0 else torch.cat((outputs, output.unsqueeze(2)), dim=-1)
    return outputs

# 修改MultiSeqModel函数，在每次生成时，给input增加一点扰动
def aug_feature_change(model, inputs):
    # inputs: [batch_size, seq_len, num_features]
    # model = AutoModelForCausalLM.from_pretrained("./weight", trust_remote_code=True).to("cuda:0")
    for i in range(inputs.shape[-1]):
        # 转为二维[batch_size, seq_len]
        input = inputs[:, :, i].to("cuda:0")
        # 给input增加一点扰动
        input = input + torch.randn_like(input) * 0.01
        output = model.generate(input, max_new_tokens=input.shape[1])
        # 拼接output，得到三维的outputs
        outputs = output.unsqueeze(2) if i == 0 else torch.cat((outputs, output.unsqueeze(2)), dim=-1)
    return outputs

def picture(inputs, original, outputs, lookback_length, prediction_length):
    # plot the prediction
    plt.figure(figsize=(12, 4))
    plt.plot(range(lookback_length, lookback_length + prediction_length), original, color="limegreen", label="Groundtruth")
    plt.plot(range(lookback_length, lookback_length + prediction_length), outputs[0], color="tomato", label="Prediction")
    plt.plot(inputs, color="royalblue", label="Lookback")
    plt.legend()
    plt.grid()
    plt.savefig('./TimeSeries.png')

def jitter(x, sigma=0.3):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + torch.normal(mean=0., std=sigma, size=x.shape).cuda()

def permutation(x, max_segments=5, seg_mode="equal"):
    x = x.cpu().numpy()
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret.astype(np.float32)

def additive_bias_augmentation(time_series, bias_scale=0.2, n_v=1, p=0.5):
    """
    加性偏置负增强实现
    :param time_series: 输入时间序列 (s, d)
    :param bias_scale: 偏置强度（文章SWaT数据集使用b=0.2）
    :param n_v: 选择的变量数量（文章SWaT使用n_v=1）
    :param p: 每个变量中选择点的比例（文章SWaT使用p=0.5）
    :return: 增强后的时间序列
    """
    s, d = time_series.shape
    augmented = time_series.copy()
    
    # 随机选择n_v个变量（第8页算法1，行2）
    selected_vars = np.random.choice(d, size=n_v, replace=False)
    
    for var in selected_vars:
        # 随机选择p比例的点（行5）
        selected_points = np.random.choice(s, size=int(s * p), replace=False)
        
        # 对选中的点应用加性偏置（行7，公式5）
        augmented[selected_points, var] += bias_scale
    
    return augmented

# W_t_augmented = additive_bias_augmentation(W_t, bias_scale=0.2, n_v=1, p=0.5)


# # 数据增强扰动函数
# def data_augmentation(inputs):
#     return jitter(torch.tensor(permutation(inputs)).float().to("cuda:0"))

def data_augmentation(inputs, dataset):
    # inputs.shape: [len, num_features]
    window_size = 4
    dic_b = {'SWAT':0.2, 'WADI':0.1, 'MSL':0.2, 'SMAP':0.2}
    b = dic_b[dataset]
    p = 0.5
    n_v = 1
    # if isinstance(inputs, np.ndarray):
    #     inputs = torch.from_numpy(inputs).float()
    # 将inputs按照窗口大小分割
    index = 0
    aug_inputs = []
    while index + window_size <= inputs.shape[0]:
        input = inputs[index:index+window_size, :]
        # print(type(input))
        aug_input = additive_bias_augmentation(input, bias_scale=b, n_v=n_v, p=p)
        # print(aug_input.shape)
        aug_inputs.append(torch.from_numpy(aug_input).float())
        index += window_size
    else:
        input = inputs[index:, :]
        aug_input = additive_bias_augmentation(input, bias_scale=b, n_v=n_v, p=p)
        aug_inputs.append(torch.from_numpy(aug_input).float())
    aug = torch.cat(aug_inputs, dim=0)
    return aug



datasets = ['MSL', 'SMAP', 'SWAT', 'WADI']
for dataset in datasets:
    print(f"Processing {dataset} dataset...")
    if dataset in ['MSL', 'SMAP']:
        dataset_lower = 'data'
    else:
        dataset_lower = dataset.lower()
    data_path = f'./code_MGCLAD/datasets/{dataset_lower}/processed/{dataset}_train.pkl'
    # 读取train.pkl文件
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(data.shape)
    data = data_augmentation(data, dataset)
    # print(data.shape[-1])
    all_aug_data = []
    for i in range(data.shape[-1]):
        win_size = 1440
        pre_len = 64
        current_pos = 0
        stride = pre_len
        data_aug = []
        # input = torch.tensor(data[:, i]).float().to('cuda:0')
        input = data[:, i].to('cuda:0')
        # print(input.shape)
        data_aug.append(input[:win_size].unsqueeze(0).cpu()) # 最开始的窗口直接填充
        while current_pos + win_size <= len(input):
            inputs = input[current_pos:current_pos+win_size].unsqueeze(0) # shape:[1, win_size]
            # inputs = data_augmentation(inputs, dataset)
            outputs = model.generate(inputs, max_new_tokens=pre_len)[:,-pre_len:].cpu()
            # print(outputs)
            data_aug.append(outputs)
            current_pos += stride
            # 定期清理显存
            if current_pos % (10 * stride) == 0:
                torch.cuda.empty_cache()
        aug = torch.concat(data_aug, dim=1).squeeze(0)
        if len(aug) > len(input):
            length = len(input)
            aug = aug[:length]
            # print(aug.shape)
        all_aug_data.append(aug)  # 移除批次维度并保存
    aug_data = torch.stack(all_aug_data).permute(1,0)
    print(aug_data.shape)
    save_path = f'./code_MGCLAD/datasets/{dataset_lower}/processed/{dataset}_train_aug_neg.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(aug_data, f)
    print(f"Augmented data saved to {save_path}")
    # picture(input[:win_size].cpu(), input[win_size:win_size+pre_len].cpu(), output.cpu(), win_size, pre_len)
    
    