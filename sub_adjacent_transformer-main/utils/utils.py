import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def set_random_seed(seed=42):
    """
    设置所有相关的随机数种子以确保结果可复现
    
    Args:
        seed (int): 随机数种子，默认为42
    """
    # Python内置随机数
    random.seed(seed)
    
    # NumPy随机数
    np.random.seed(seed)
    
    # PyTorch随机数
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.
    
    # 设置环境变量确保其他库的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' # 设置 CUBLAS 工作空间配置
    torch.use_deterministic_algorithms(True)  # 使用确定性算法
    
    print(f"Random seed set to {seed}")

def get_random_state():
    """获取当前随机数状态，用于调试"""
    return {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'torch_cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }