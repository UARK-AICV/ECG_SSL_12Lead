from copy import deepcopy
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

import sys 
current_path = os.getcwd()
sys.path.append(current_path)

from models.signal_model import signal_model_byol
from utils.contrastive_dataloader import ECG_contrastive_dataset
from utils.eval_tools import load_weights
from utils.optimizers import LARS
from utils.eval_tools import load_weights
from utils.tools import weights_init_xavier, set_requires_grad

ctx = "cuda:0" if torch.cuda.is_available() else 'cpu'

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

def regression_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)

def run():
    root_folder = './data_folder'
    data_folder = os.path.join(root_folder,'data_summary_without_preprocessing')
    # equivalent_classes = [['CRBBB', 'RBBB'], ['PAC', 'SVPB'], ['PVC', 'VPB']]
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    no_channels = 12 
    signal_size = 250
    train_stride = signal_size
    train_chunk_length = 0


    transforms = ["TimeOut_difflead","GaussianNoise"]

    batch_size = 1024
    learning_rate = 1e-3
    no_epoches = 400

    get_mean = np.load(os.path.join(data_folder,"mean.npy"))
    get_std = np.load(os.path.join(data_folder,"std.npy"))

    t_params = {"gaussian_scale":[0.005,0.025], "global_crop_scale": [0.5, 1.0], "local_crop_scale": [0.1, 0.5],
                "output_size": 250, "warps": 3, "radius": 10, "shift_range":[0.2,0.5],
                "epsilon": 10, "magnitude_range": [0.5, 2], "downsample_ratio": 0.2, "to_crop_ratio_range": [0.2, 0.4],
                "bw_cmax":0.1, "em_cmax":0.5, "pl_cmax":0.2, "bs_cmax":1, "stats_mean":get_mean,"stats_std":get_std}


    train_dataset = ECG_contrastive_dataset(summary_folder=data_folder, signal_size=signal_size, stride=train_stride,
                            chunk_length=train_chunk_length, transforms=transforms,t_params=t_params,
                            equivalent_classes=equivalent_classes, sample_items_per_record=1,random_crop=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4,batch_size=batch_size,drop_last=True)

    no_classes = 24
    online_network = signal_model_byol(no_classes)
    target_network = deepcopy(online_network)
    online_network.to(ctx)
    target_network.to(ctx)

    set_requires_grad(target_network,False)
    
    # optimizer = torch.optim.Adam(list(online_network.parameters()) + list(target_network.parameters()),lr=learning_rate)
    optimizer = torch.optim.Adam(online_network.parameters(),lr=learning_rate)
    # optimizer = LARS(online_network.parameters(),lr=0.1,weight_decay=0.0048)
    scheduler_steplr = CosineAnnealingLR(optimizer, no_epoches, eta_min=1e-4, last_epoch=-1)


    optimizer.zero_grad()
    optimizer.step()

    lowest_train_loss = 2
    for epoch in range(1,no_epoches+1):
        print('===================Epoch [{}/{}]'.format(epoch,no_epoches))
        print('Current learning rate: ',optimizer.param_groups[0]['lr'])
        scheduler_steplr.step()
        online_network.train()
        train_loss = 0
        train_acc = 0

        for batch_idx, sample in enumerate(tqdm(train_dataloader)):
            data_i = sample['sig_i'].to(ctx).float()
            data_j = sample['sig_j'].to(ctx).float()

            # features, projector, predictor, output
            h1a,z1a,t1a,_ = online_network(data_i)
            h1b,z1b,t1b,_ = online_network(data_j)

            with torch.no_grad():
                h2a,z2a,t2a,_ = target_network(data_j)
                h2b,z2b,t2b,_ = target_network(data_j)
            
            # image 1 to image 2 loss
            loss = regression_loss(t1a, z2b)
            loss += regression_loss(t1b,z2a)
            total_loss = loss.mean()
            # image 2 to image 1 loss
            
            train_loss += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            t_d = 0.9 
            # t_d = 0.996
            for param_q, param_k in zip(online_network.parameters(), target_network.parameters()):
                param_k.data = param_k.data * t_d + param_q.data * (1. - t_d)

        whole_train_loss = train_loss / (batch_idx + 1)
        print(f'Train Loss: {whole_train_loss}')
        if whole_train_loss < lowest_train_loss:
            lowest_train_loss = whole_train_loss
            torch.save(online_network.state_dict(), f'./checkpoints/BYOL_signal.pth')


if __name__ == "__main__":
    run()