# 自动根据要求，执行python任务：执行多个参数不同的train_cgraph_trans.py任务

import os
# 通过接受参数来判断有多少个任务
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# 1. 读取参数
parser = argparse.ArgumentParser()
# 输入任务列表
parser.add_argument('--mission', type=str, nargs='+', help='mission list')
# parser.add_argument('--init_lr', type=float, nargs='+', help='init_lr list')
parser.add_argument('--epochs', type=int, nargs='+', help='epochs list')
parser.add_argument('--lookback', type=int, nargs='+', help='lookback list')
parser.add_argument('--is_adjust', type=str2bool, default=True)
parser.add_argument("--comment", type=str, default="") # comment for the tensorboard log
args = parser.parse_args()
mission_name = args.mission
# init_lr = args.init_lr
epochs = args.epochs
lookback = args.lookback
# print('mission_name:', mission_name)
# for name in mission_name:
#     print(name)
length = len(mission_name)
for i in range(length):
    name = mission_name[i]
    epoch = epochs[i]
    look = lookback[i]
    if args.comment != "":
        if name == 'WADI':
        # 2. 执行任务
            os.system('python ./code_MGCLAD/train_cgraph_trans.py --dataset WADI --bs 256 --init_lr 1e-3 --epochs 30 --lookback 16 --is_adjust {} --comment {}'.format(args.is_adjust, args.comment))
        elif name == 'SWAT':
            os.system('python ./code_MGCLAD/train_cgraph_trans.py --dataset SWAT --init_lr 1e-3 --epochs 30 --lookback 32 --is_adjust {} --comment {}'.format(args.is_adjust, args.comment))
        else:
            os.system('python ./code_MGCLAD/train_cgraph_trans.py --dataset {} --init_lr 1e-4 --epochs {} --lookback {} --is_adjust {} --comment {}'.format(name, epoch, look, args.is_adjust, args.comment))
    else:
        if name == 'WADI':
        # 2. 执行任务
            os.system('python ./code_MGCLAD/train_cgraph_trans.py --dataset WADI --bs 256 --init_lr 1e-3 --epochs 30 --lookback 16 --is_adjust {}'.format(args.is_adjust))
        elif name == 'SWAT':
            os.system('python ./code_MGCLAD/train_cgraph_trans.py --dataset SWAT --init_lr 1e-3 --epochs 30 --lookback 32 --is_adjust {}'.format(args.is_adjust))
        else:
            os.system('python ./code_MGCLAD/train_cgraph_trans.py --dataset {} --init_lr 1e-4 --epochs {} --lookback {} --is_adjust {}'.format(name, epoch, look, args.is_adjust))