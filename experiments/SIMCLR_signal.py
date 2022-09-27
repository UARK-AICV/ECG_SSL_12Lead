import os
import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import math 

import sys 
current_path = os.getcwd()
sys.path.append(current_path)

from models.signal_model import signal_model_simclr
from utils.contrastive_dataloader import ECG_contrastive_dataset
from utils.tools import weights_init_xavier

ctx = "cuda:0" if torch.cuda.is_available() else 'cpu'
eps = 1e-7

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)

class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        
        out_1_dist = out_1
        out_2_dist = out_2
        
        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss


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
    no_epoches = 1000

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
    model = signal_model_simclr(no_classes)
    projection_head = Projection(model.num_ftrs, hidden_dim=512, output_dim=128)

    model.apply(weights_init_xavier)
    projection_head.apply(weights_init_xavier)
    model.to(ctx)
    projection_head.to(ctx)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler_steplr = CosineAnnealingLR(optimizer, no_epoches, eta_min=1e-4, last_epoch=-1)

    optimizer.zero_grad()
    optimizer.step()

    lowest_train_loss = 10
    for epoch in range(1,no_epoches+1):
        print('===================Epoch [{}/{}]'.format(epoch,no_epoches))
        print('Current learning rate: ',optimizer.param_groups[0]['lr'])
        scheduler_steplr.step()
        model.train()
        train_loss = 0

        for batch_idx, sample in enumerate(tqdm(train_dataloader)):
            data_i = sample['sig_i'].to(ctx).float()
            data_j = sample['sig_j'].to(ctx).float()

            h1 = model(data_i)[0]
            h2 = model(data_j)[0]

            # PROJECT
            # img -> E -> h -> || -> z
            # (b, 2048, 2, 2) -> (b, 128)
            z1 = projection_head(h1.squeeze())
            z2 = projection_head(h2.squeeze())

            loss = nt_xent_loss(z1,z2,temperature=0.1)
            
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        whole_train_loss = train_loss / (batch_idx + 1)
        print(f'Train Loss: {whole_train_loss}')
        if whole_train_loss < lowest_train_loss:
            lowest_train_loss = whole_train_loss
            torch.save(model.state_dict(), f'./checkpoints/SIMCLR_signal.pth')


if __name__ == "__main__":
    run()