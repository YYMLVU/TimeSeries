# 将pkl转换为npy
import os
import numpy as np
import pickle

pkl_path = './code_MGCLAD/datasets/swat/processed'
npy_path = './sub_adjacent_transformer-main/dataset/SWAT'

os.makedirs(npy_path, exist_ok=True)

for file in os.listdir(pkl_path):
    if file.endswith('.pkl'):
        with open(os.path.join(pkl_path, file), 'rb') as f:
            data = pickle.load(f)
            np.save(os.path.join(npy_path, file.replace('.pkl', '.npy')), data)
            print(f'{file} converted to npy')

print('All done!')