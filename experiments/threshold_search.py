from calendar import leapdays
from re import X
from signal import signal
import time
import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from tqdm import TqdmSynchronisationWarning
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import argparse

import sys 
current_path = os.getcwd()
sys.path.append(current_path)

from models.signal_model import signal_model
from models.spectrogram_model import spectrogram_model
from models.ensemble_model import ensemble_model
from utils.base_dataloader import ECG_dataset_base

from utils.eval_tools import compute_accuracy, compute_f_measure_mod, compute_beta_measures
from utils.eval_tools import compute_auc, load_weights, compute_challenge_metric
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score, confusion_matrix, fbeta_score
from sklearn.metrics import precision_recall_curve, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="signal")
    parser.add_argument("--best_type", type=str, default="ROC")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--gating",action="store_true")
    parser.add_argument("--weight_folder",type=str,default="")
    return parser.parse_args()

def run():
    args = parse_args()
    ctx = "cuda:"+args.gpu if torch.cuda.is_available() else 'cpu'

    root_folder = './data_folder'
    data_folder = os.path.join(root_folder,'data_summary_without_preprocessing')
    normal_class = '426783006'
    # equivalent_classes = [['CRBBB', 'RBBB'], ['PAC', 'SVPB'], ['PVC', 'VPB']]
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    weights_file = './data_folder/evaluation-2020-master/weights.csv'
    classes, weights = load_weights(weights_file, equivalent_classes)

    no_channels = 12 
    signal_size = 250
    val_stride = signal_size//2 # overlap sample signal
    val_chunk_length = signal_size
    weight_folder = args.weight_folder
    batch_size = 128

    transforms = True

    # run 10 fold cross validation
    for no_fold in range(10):
        print('### FOLD-FOLD-FOLD-FOLD-FOLD ###')
        print(f'Starting fold {no_fold} ...')
        print('### FOLD-FOLD-FOLD-FOLD-FOLD ###')
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #----------SIGNAL--------------SIGNAL-------------SIGNAL-------------------SIGNAL---------------------SIGNAL-----------------SIGNAL------------#
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        if args.model_type == 'signal':
            val_dataset = ECG_dataset_base(summary_folder=data_folder, classes=classes,signal_size=signal_size, stride=val_stride,
                                        chunk_length=val_chunk_length, transforms=transforms, stft_inc=False, meta_inc=False, t_or_v='val',
                                        equivalent_classes=equivalent_classes, sample_items_per_record=1, preload=False,random_crop=False,val_fold=no_fold)
            val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=4,batch_size=batch_size)

            no_classes = val_dataset.get_num_classes()
            model = signal_model(no_classes)
            weight_name = os.path.join(weight_folder,args.model_type+"_fold"+str(no_fold)+"_best"+args.best_type+"_finetune.pth")
            print(weight_name)
            if not os.path.exists(weight_name):
                print("Cannot find this weight file")
                continue
            state_dict = torch.load(weight_name,map_location=ctx)
            model.load_state_dict(state_dict,strict=True)
            model.to(ctx)

            model.eval()
            with torch.no_grad():
                val_pred = []
                val_gt = []
                val_name = []

                for batch_idx, sample in enumerate(val_dataloader):
                    signal = sample['sig'].to(ctx).float()
                    label = sample['lbl'].to(ctx).float()
                    name = sample['idx']

                    pred = model(signal)
                    result = torch.sigmoid(pred)

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

            auroc, auprc = compute_auc(val_gt_after,val_pred_after.astype(np.float64))
            print(f'-----> AU_ROC: {auroc}, AUPRC: {auprc}')
            
            thres_range = np.linspace(start=0.1,stop=0.5,num=5, dtype=np.float16,endpoint=True)
            stack_thres_class = []
            for cl in range(no_classes):
                copy_pred_data = np.copy(val_pred_after)
                mod_col = copy_pred_data[:,cl]
                del_col_data = np.delete(copy_pred_data,cl,1)
                
                del_col_data = (del_col_data>=0.1)
                stack_chmetric = []
                for thres in thres_range:
                    copy_mod_col = np.copy(mod_col)
                    copy_del_col_data = np.copy(del_col_data)
                    copy_mod_col = (copy_mod_col>=thres)
                    copy_del_col_data = np.insert(copy_del_col_data,cl,copy_mod_col,1)

                    stack_chmetric.append(compute_challenge_metric(weights,val_gt_after.astype(np.bool),copy_del_col_data.astype(np.bool),classes,normal_class))
                    del copy_mod_col
                    del copy_del_col_data

                best_chmetric_pos = np.argmax(stack_chmetric)
                stack_thres_class.append(thres_range[best_chmetric_pos])
                del copy_pred_data
                del del_col_data
            # print(stack_thres_class)
            for idx in range(no_classes):
                val_pred_after[:,idx] = (val_pred_after[:,idx]>=stack_thres_class[idx])
            chmetric =compute_challenge_metric(weights,val_gt_after.astype(np.bool),val_pred_after.astype(np.bool),classes,normal_class)
            print(f'-----> Challenge metric: {chmetric}')
            acc = compute_accuracy(val_gt_after.astype(np.bool),val_pred_after.astype(np.bool))
            print(f'-----> Accuracy: {acc}')
            f1 = compute_f_measure_mod(val_gt_after.astype(np.bool),val_pred_after.astype(np.bool))
            print(f'-----> F1 macro score: {f1}')
            f2, b2 = compute_beta_measures(val_gt_after.astype(np.bool),val_pred_after.astype(np.bool),2)
            print(f'-----> F2 score: {f2}')
            print(f'-----> G2 score: {b2}')
            print(auroc,"\t",auprc,"\t",acc,"\t",f1,"\t",f2,"\t",b2,"\t",chmetric)
        
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #----------SPECTROGRAM------------SPECTROGRAM---------SPECTROGRAM-------------SPECTROGRAM---------------SPECTROGRAM----------SPECTROGRAM-------#
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        elif args.model_type == 'spectrogram':
            val_dataset = ECG_dataset_base(summary_folder=data_folder, classes=classes,signal_size=signal_size, stride=val_stride,
                                        chunk_length=val_chunk_length, transforms=transforms, stft_inc=True, meta_inc=False, t_or_v='val',
                                        equivalent_classes=equivalent_classes, sample_items_per_record=1, preload=False,random_crop=False,val_fold=no_fold)
            val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=4,batch_size=batch_size)
            
            no_classes = val_dataset.get_num_classes()
            model = spectrogram_model(no_classes)
            weight_name = os.path.join(weight_folder,args.model_type+"_fold"+str(no_fold)+"_best"+args.best_type+"_finetune.pth")
            print(weight_name)
            if not os.path.exists(weight_name):
                print("Cannot find this weight file")
                continue
            state_dict = torch.load(weight_name,map_location=ctx)
            model.load_state_dict(state_dict,strict=True)
            model.to(ctx)

            model.eval()
            with torch.no_grad():
                val_pred = []
                val_gt = []
                val_name = []

                for batch_idx, sample in enumerate(val_dataloader):
                    signal = sample['stft'].to(ctx).float()
                    label = sample['lbl'].to(ctx).float()
                    name = sample['idx']

                    pred = model(signal)
                    result = torch.sigmoid(pred)

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

            auroc, auprc = compute_auc(val_gt_after,val_pred_after.astype(np.float64))
            print(f'-----> AU_ROC: {auroc}, AUPRC: {auprc}')
            
            thres_range = np.linspace(start=0.1,stop=0.5,num=5, dtype=np.float16,endpoint=True)
            stack_thres_class = []
            for cl in range(no_classes):
                copy_pred_data = np.copy(val_pred_after)
                mod_col = copy_pred_data[:,cl]
                del_col_data = np.delete(copy_pred_data,cl,1)
                
                del_col_data = (del_col_data>=0.1)
                stack_chmetric = []
                for thres in thres_range:
                    copy_mod_col = np.copy(mod_col)
                    copy_del_col_data = np.copy(del_col_data)
                    copy_mod_col = (copy_mod_col>=thres)
                    copy_del_col_data = np.insert(copy_del_col_data,cl,copy_mod_col,1)

                    stack_chmetric.append(compute_challenge_metric(weights,val_gt_after.astype(np.bool),copy_del_col_data.astype(np.bool),classes,normal_class))
                    del copy_mod_col
                    del copy_del_col_data

                best_chmetric_pos = np.argmax(stack_chmetric)
                stack_thres_class.append(thres_range[best_chmetric_pos])
                del copy_pred_data
                del del_col_data
            # print(stack_thres_class)
            for idx in range(no_classes):
                val_pred_after[:,idx] = (val_pred_after[:,idx]>=stack_thres_class[idx])
            chmetric =compute_challenge_metric(weights,val_gt_after.astype(np.bool),val_pred_after.astype(np.bool),classes,normal_class)
            print(f'-----> Challenge metric: {chmetric}')
            acc = compute_accuracy(val_gt_after.astype(np.bool),val_pred_after.astype(np.bool))
            print(f'-----> Accuracy: {acc}')
            f1 = compute_f_measure_mod(val_gt_after.astype(np.bool),val_pred_after.astype(np.bool))
            print(f'-----> F1 macro score: {f1}')
            f2, b2 = compute_beta_measures(val_gt_after.astype(np.bool),val_pred_after.astype(np.bool),2)
            print(f'-----> F2 score: {f2}')
            print(f'-----> G2 score: {b2}')
            print(auroc,"\t",auprc,"\t",acc,"\t",f1,"\t",f2,"\t",b2,"\t",chmetric)

        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #----------ENSEMBLE-----------ENSEMBLE-------ENSEMBLE-----------ENSEMBLE--------------ENSEMBLE---------ENSEMBLE---------ENSEMBLE---------------#
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------------------------------------#
        elif args.model_type == 'ensemble':
            val_dataset = ECG_dataset_base(summary_folder=data_folder, classes=classes,signal_size=signal_size, stride=val_stride,
                                        chunk_length=val_chunk_length, transforms=transforms, stft_inc=True, meta_inc=False, t_or_v='val',
                                        equivalent_classes=equivalent_classes, sample_items_per_record=1, preload=False,random_crop=False,val_fold=no_fold)
            val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=4,batch_size=batch_size)

            no_classes = val_dataset.get_num_classes()
            model = ensemble_model(no_classes,gate=args.gating)
            if args.gating:
                weight_name = os.path.join(weight_folder,args.model_type+"_withgating_fold"+str(no_fold)+"_best"+args.best_type+".pth")
            else:
                weight_name = os.path.join(weight_folder,args.model_type+"_wthoutgating_fold"+str(no_fold)+"_best"+args.best_type+".pth")
            print(weight_name)
            if not os.path.exists(weight_name):
                print("Cannot find this weight file")
                continue
            state_dict = torch.load(weight_name,map_location=ctx)
            model.load_state_dict(state_dict,strict=True)
            model.to(ctx)

            model.eval()
            with torch.no_grad():
                val_pred = []
                val_gt = []
                val_name = []

                for batch_idx, sample in enumerate(val_dataloader):
                    signal = sample['sig'].to(ctx).float()
                    stft = sample['stft'].to(ctx).float()
                    label = sample['lbl'].to(ctx).float()
                    name = sample['idx']
                    
                    pred = model(signal,stft)
                    result = torch.sigmoid(pred)

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
                
            auroc, auprc = compute_auc(val_gt_after,val_pred_after.astype(np.float64))
            print(f'-----> AU_ROC: {auroc}, AUPRC: {auprc}')
            thres_range = np.linspace(start=0.1,stop=0.5,num=5, dtype=np.float16,endpoint=True)
            stack_thres_class = []
            for cl in range(no_classes):
                copy_pred_data = np.copy(val_pred_after)
                mod_col = copy_pred_data[:,cl]
                del_col_data = np.delete(copy_pred_data,cl,1)
                
                del_col_data = (del_col_data>=0.1)
                stack_chmetric = []
                for thres in thres_range:
                    copy_mod_col = np.copy(mod_col)
                    copy_del_col_data = np.copy(del_col_data)
                    copy_mod_col = (copy_mod_col>=thres)
                    copy_del_col_data = np.insert(copy_del_col_data,cl,copy_mod_col,1)

                    stack_chmetric.append(compute_challenge_metric(weights,val_gt_after.astype(np.bool),copy_del_col_data.astype(np.bool),classes,normal_class))
                    del copy_mod_col
                    del copy_del_col_data

                best_chmetric_pos = np.argmax(stack_chmetric)
                stack_thres_class.append(thres_range[best_chmetric_pos])
                del copy_pred_data
                del del_col_data
            
            for idx in range(no_classes):
                val_pred_after[:,idx] = (val_pred_after[:,idx]>=stack_thres_class[idx])
            chmetric =compute_challenge_metric(weights,val_gt_after.astype(np.bool),val_pred_after.astype(np.bool),classes,normal_class)
            print(f'-----> Challenge metric: {chmetric}')
            acc = compute_accuracy(val_gt_after.astype(np.bool),val_pred_after.astype(np.bool))
            print(f'-----> Accuracy: {acc}')
            f1 = compute_f_measure_mod(val_gt_after.astype(np.bool),val_pred_after.astype(np.bool))
            print(f'-----> F1 macro score: {f1}')
            f2, b2 = compute_beta_measures(val_gt_after.astype(np.bool),val_pred_after.astype(np.bool),2)
            print(f'-----> F2 score: {f2}')
            print(f'-----> G2 score: {b2}')
            print(auroc,"\t",auprc,"\t",acc,"\t",f1,"\t",f2,"\t",b2,"\t",chmetric)


if __name__ == "__main__":
    run()