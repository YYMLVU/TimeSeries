import random
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch.nn.modules.module import Module
import numpy as np
import tsaug
import torch
import time
from torch.nn.functional import interpolate

def totensor(x):
    return torch.from_numpy(x).type(torch.FloatTensor).cuda()

class cutout():
    def __init__(self, perc=0.1) -> None:
        self.perc = perc
    def __call__(self,ts):
        seq_len = ts.shape[1]
        new_ts = ts.clone()
        win_len = int(self.perc * seq_len)
        start = np.random.randint(0, seq_len - win_len - 1)
        end = start + win_len
        start = max(0, start)
        end = min(end, seq_len)
        new_ts[:, start:end, :] = 0.0
        return new_ts

class jitter():
    def __init__(self, sigma=0.3) -> None:
        self.sigma = sigma
    def __call__(self,x):
        return x + torch.normal(mean=0., std=self.sigma, size=x.shape).cuda()

class scaling():
    def __init__(self, sigma=0.5) -> None:
        self.sigma = sigma
    def __call__(self,x):
        factor = torch.normal(mean=1., std=self.sigma, size=(x.shape[0], x.shape[2])).cuda()
        res = torch.multiply(x, torch.unsqueeze(factor, 1))
        return res

class time_warp():
    def __init__(self, n_speed_change=100, max_speed_ratio=10) -> None:
        self.transform = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

    def __call__(self, x_torch):
        x = x_torch.cpu().detach().numpy()
        x_tran =  self.transform.augment(x)
        return totensor(x_tran.astype(np.float32))

class magnitude_warp():

    def __init__(self, n_speed_change:int =100, max_speed_ratio=10) -> None:
        self.transform = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

    def __call__(self, x_torch):
        x = x_torch.cpu().detach().numpy()
        x_t = np.transpose(x, (0, 2, 1))
        x_tran =  self.transform.augment(x_t).transpose((0,2,1))
        return totensor(x_tran.astype(np.float32))


class window_slice():
    def __init__(self, reduce_ratio=0.5,diff_len=True) -> None:
        self.reduce_ratio = reduce_ratio
        self.diff_len = diff_len
    def __call__(self,x):

        # begin = time.time()
        x = torch.transpose(x,2,1)

        target_len = np.ceil(self.reduce_ratio * x.shape[2]).astype(int)
        if target_len >= x.shape[2]:
            return x
        if self.diff_len:
            starts = np.random.randint(low=0, high=x.shape[2] - target_len, size=(x.shape[0])).astype(int)
            ends = (target_len + starts).astype(int)
            croped_x =  torch.stack([x[i, :, starts[i]:ends[i]] for i in range(x.shape[0])],0)

        else:
            start = np.random.randint(low=0, high=x.shape[2] - target_len)
            end  = target_len+start
            croped_x = x[:, :, start:end]

        ret = interpolate(croped_x, x.shape[2], mode='linear',align_corners=False)
        ret = torch.transpose(ret,2,1)
        # end = time.time()
        # old_window_slice()(x)
        # end2 = time.time()
        # print(end-begin,end2-end)
        return ret


class window_warp():
    def __init__(self, window_ratio=0.3, scales=[0.5, 2.]) -> None:
        self.window_ratio = window_ratio
        self.scales = scales

    def __call__(self,x_torch):

        begin = time.time()
        B,T,D = x_torch.size()
        x = torch.transpose(x_torch,2,1)
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        warp_scales = np.random.choice(self.scales, B)
        warp_size = np.ceil(self.window_ratio * T).astype(int)
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(low=1, high=T - warp_size - 1, size=(B)).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        rets = []

        for i  in range(x.shape[0]):
            window_seg = torch.unsqueeze(x[i,:,window_starts[i]:window_ends[i]],0)
            window_seg_inter = interpolate(window_seg,int(warp_size * warp_scales[i]),mode='linear',align_corners=False)[0]
            start_seg = x[i,:,:window_starts[i]]
            end_seg = x[i,:,window_ends[i]:]
            ret_i = torch.cat([start_seg,window_seg_inter,end_seg],-1)
            ret_i_inter = interpolate(torch.unsqueeze(ret_i,0),T,mode='linear',align_corners=False)
            rets.append(ret_i_inter)

        ret = torch.cat(rets,0)
        ret = torch.transpose(ret,2,1)
        # end = time.time()
        # old_window_warp()(x_torch)
        # end2 = time.time()
        # print(end-begin,end2-end)
        return ret

class subsequence():
    def __init__(self) -> None:
        pass
    def __call__(self,x):
        ts = x
        seq_len = ts.shape[1]
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2, high=ts_l + 1)
        new_ts = ts.clone()
        start = np.random.randint(ts_l - crop_l + 1)
        end = start + crop_l
        start = max(0, start)
        end = min(end, seq_len)
        new_ts[:, :start, :] = 0.0
        new_ts[:, end:, :] = 0.0
        return new_ts

class AUGs(Module):
    def __init__(self,caugs,p=0.2):
        super(AUGs,self).__init__()

        self.augs = caugs
        self.p = p
    def forward(self,x_torch):
        x = x_torch.clone()
        for a in self.augs:
            if random.random()<self.p:
                x = a(x)
        return x.clone(),x_torch.clone()

class RandomAUGs(Module):
    def __init__(self,caugs,p=0.2):
        super(RandomAUGs,self).__init__()

        self.augs = caugs
        self.p = p
    def forward(self,x_torch):
        x = x_torch.clone()

        if random.random()<self.p:
            x =random.choice(self.augs)(x)
        return x.clone(),x_torch.clone()

class AutoAUG(Module):
    def __init__(self, aug_p1=0.2, aug_p2 = 0.0, used_augs=None, device=None, dtype=None) -> None:
        super(AutoAUG,self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        all_augs = [subsequence(),cutout(), jitter(), scaling(), time_warp(), window_slice(), window_warp()]

        if used_augs is not None:
            self.augs = []
            for i in range(len(used_augs)):
                if used_augs[i]:
                    self.augs.append(all_augs[i])
        else:
            self.augs = all_augs
        self.weight = Parameter(torch.empty((2,len(self.augs)), **factory_kwargs))
        self.reset_parameters()
        self.aug_p1 = aug_p1
        self.aug_p2 = aug_p2

    def get_sampling(self,temperature=1.0, bias=0.0):

        if self.training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(self.weight.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.cuda()
            gate_inputs = (gate_inputs + self.weight) / temperature
            # para = torch.sigmoid(gate_inputs)
            para = torch.softmax(gate_inputs,-1)
            return para
        else:
            return torch.softmax(self.weight,-1)


    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.01)
    def forward(self, x):
        if self.aug_p1 ==0 and self.aug_p2==0:
            return x.clone(), x.clone()
        para = self.get_sampling()

        if random.random()>self.aug_p1 and self.training:
            aug1 = x.clone()
        else:
            xs1_list = []
            for aug in self.augs:
                xs1_list.append(aug(x))
            xs1 = torch.stack(xs1_list, 0)
            xs1_flattern = torch.reshape(xs1, (xs1.shape[0], xs1.shape[1] * xs1.shape[2] * xs1.shape[3]))
            aug1 = torch.reshape(torch.unsqueeze(para[0], -1) * xs1_flattern, xs1.shape)
            aug1 = torch.sum(aug1,0)

        aug2 = x.clone()

        return aug1,aug2


if __name__ == '__main__':
    x = torch.randn((3, 32, 4)).cuda()
    # x = torch.randn((32, 128, 9))
    # augs = [cutout(), jitter(), scaling(), time_warp(), window_slice(), window_warp()]
    # aug = AUGs(augs)
    # aug(x)
    # print(x.shape)
    autoaug = AutoAUG(aug_p1=1.0,aug_p2=0.0,device='cuda')
    for i in range(100):
        out1, _ = autoaug(x)
        if out1.shape != x.shape or torch.isnan(out1).any():
            print(out1.shape)
    # with open('./output.txt', 'a') as f:
    #     f.write('\noutput1 : {}\n'.format(out1))
    #     f.write('output2 : {}\n'.format(out2))