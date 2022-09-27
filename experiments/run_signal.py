import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import argparse

import sys 
current_path = os.getcwd()
sys.path.append(current_path)

from models.signal_model import signal_model
from utils.base_dataloader import ECG_dataset_base
from utils.eval_tools import compute_auc, load_weights
from utils.tools import weights_init_xavier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int, default=128)
    parser.add_argument("--lr_rate",type=float, default=5e-3)
    parser.add_argument("--num_epoches",type=int, default=100)
    parser.add_argument('--fold', nargs='+', type=int, default=[11])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--save_folder", type=str, default=None)
    return parser.parse_args()

def run():
    args = parse_args()
    
    ctx = "cuda:"+args.gpu if torch.cuda.is_available() else 'cpu'

    root_folder = './data_folder'
    data_folder = os.path.join(root_folder,'data_summary_without_preprocessing')

    # equivalent_classes = [['CRBBB', 'RBBB'], ['PAC', 'SVPB'], ['PVC', 'VPB']]
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    weights_file = './data_folder/evaluation-2020-master/weights.csv'
    classes, weights = load_weights(weights_file, equivalent_classes)

    no_channels = 12 
    signal_size = 250
    train_stride = signal_size
    train_chunk_length = 0
    # train_stride = signal_size//2
    # train_chunk_length = signal_size
    val_stride = signal_size//2 # overlap sample signal
    val_chunk_length = signal_size

    transforms = True
    batch_size = args.batch_size
    learning_rate = args.lr_rate
    no_epoches = args.num_epoches

    list_folds = args.fold
    if 11 in list_folds:
        fold_range = np.arange(10)
    else:
        fold_range = list_folds
    # run 10 fold cross validation
    for no_fold in fold_range:
        print('### FOLD-FOLD-FOLD-FOLD-FOLD ###')
        print(f'Starting fold {no_fold} ...')
        print('### FOLD-FOLD-FOLD-FOLD-FOLD ###')
      
        train_dataset = ECG_dataset_base(summary_folder=data_folder,classes=classes, signal_size=signal_size, stride=train_stride,
                            chunk_length=train_chunk_length, transforms=transforms, stft_inc=False, meta_inc=False, t_or_v='train',
                            equivalent_classes=equivalent_classes, sample_items_per_record=5, preload=False,random_crop=True,val_fold=no_fold)
        train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4,batch_size=batch_size)

        val_dataset = ECG_dataset_base(summary_folder=data_folder, classes=classes,signal_size=signal_size, stride=val_stride,
                            chunk_length=val_chunk_length, transforms=transforms, stft_inc=False, meta_inc=False, t_or_v='val',
                            equivalent_classes=equivalent_classes, sample_items_per_record=1, preload=True,random_crop=False,val_fold=no_fold)
        val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=4,batch_size=batch_size)

        no_classes = train_dataset.get_num_classes()
        model = signal_model(no_classes)

        # use the pretrain models from self-supervised learning 
        if args.finetune is not None:
            state_dict = torch.load(args.finetune,map_location=ctx)
            model.load_state_dict(state_dict,strict=False)
        else:
            model.apply(weights_init_xavier)
        model.to(ctx)

        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        scheduler_steplr = CosineAnnealingLR(optimizer, no_epoches, eta_min=1e-5, last_epoch=-1)
        optimizer.zero_grad()
        optimizer.step()

        best_auroc = 0
        best_auprc = 0
        for epoch in range(1,no_epoches+1):
            print('===================Epoch [{}/{}]'.format(epoch,no_epoches))
            print('Current learning rate: ',optimizer.param_groups[0]['lr'])
            scheduler_steplr.step()
            model.train()
            train_loss = 0
            train_pred = []
            train_gt = []

            for batch_idx, sample in enumerate(tqdm(train_dataloader)):
                signal = sample['sig'].to(ctx).float()
                signal = signal.view(-1,no_channels,signal_size)
                label = sample['lbl'].to(ctx).float()
                label = label.view(-1,no_classes)
                
                pred = model(signal)
                result = torch.sigmoid(pred)

                loss = criterion(pred,label)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_pred.append(result.detach().cpu().numpy())
                train_gt.append(label.detach().cpu().numpy())

            train_pred = np.concatenate(train_pred,axis=0)
            train_gt = np.concatenate(train_gt,axis=0)

            print(f'Train Loss: {train_loss / (batch_idx + 1)}')

            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_pred = []
                val_gt = []
                val_name = []

                for batch_idx, sample in enumerate(val_dataloader):
                    signal = sample['sig'].to(ctx).float()
                    label = sample['lbl'].to(ctx).float()
                    name = sample['idx']

                    pred = model(signal)
                    result = torch.sigmoid(pred)

                    loss = criterion(pred,label)
                    val_loss += loss.item()

                    val_pred.append(result.detach().cpu().numpy())
                    val_gt.append(label.detach().cpu().numpy())
                    val_name.append(name)

                val_pred = np.concatenate(val_pred,axis=0)
                val_gt = np.concatenate(val_gt,axis=0)
                val_name = np.concatenate(val_name,axis=0)

                df_pred = pd.DataFrame(data=val_pred)
                df_gt = pd.DataFrame(data=val_gt)
                df_name = pd.DataFrame(data=val_name)
                df_concat = pd.concat([df_name,df_gt,df_pred],axis=1,ignore_index=True)
                df_concat_group = df_concat.groupby([0]).mean()
                val_gt_after = df_concat_group[df_concat_group.columns[np.arange(0,24)]].to_numpy()
                val_pred_after = df_concat_group[df_concat_group.columns[np.arange(24,48)]].to_numpy()
                
                print('########  VALIDATION  ########')
                print(f'-----> Val Loss: {val_loss / (batch_idx + 1)}')
                auroc, auprc = compute_auc(val_gt_after,val_pred_after.astype(np.float64))
                print(f'-----> AU_ROC: {auroc}, AUPRC: {auprc}')

                if auroc > best_auroc:
                    best_auroc = auroc
                    if args.finetune is not None:
                        torch.save(model.state_dict(), f'./checkpoints/{args.save_folder}/{args.model_type}_fold{no_fold}_bestROC_finetune.pth')
                    else:
                        torch.save(model.state_dict(), f'./checkpoints/{args.save_folder}/{args.model_type}_fold{no_fold}_bestROC.pth')
                if auprc > best_auprc:
                    best_auprc = auprc
                    if args.finetune is not None:
                        torch.save(model.state_dict(), f'./checkpoints/{args.save_folder}/{args.model_type}_fold{no_fold}_bestPRC_finetune.pth')
                    else:
                        torch.save(model.state_dict(), f'./checkpoints/{args.save_folder}/{args.model_type}_fold{no_fold}_bestPRC.pth')

if __name__ == "__main__":
    run()