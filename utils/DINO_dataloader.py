import numpy as np 
import torch
import torch.utils.data
import os 
import pickle
import os 
import random
from scipy.signal import stft

from .timeseries_transformations import TimeWarp, TimeOut, ToTensor, TBaselineShift, TGaussianBlur1d
from .timeseries_transformations import TGaussianNoise, TRandomResizedCrop, TChannelResize, TNegation, TDynamicTimeWarp
from .timeseries_transformations import TDownSample, TTimeOut, TBaselineWander, TPowerlineNoise, TEMNoise
from .timeseries_transformations import TTimeOut_difflead, TChannelReduction, TTimeShift, TNormalize, Transpose

class Normalize(object):
    """Normalize using given stats.
    """
    def __init__(self, stats_mean, stats_std, input=True, channels=[]):
        self.stats_mean=stats_mean.astype(np.float32) if stats_mean is not None else None
        self.stats_std=stats_std.astype(np.float32)+1e-8 if stats_std is not None else None
        self.input = input
        if(len(channels)>0):
            for i in range(len(stats_mean)):
                if(not(i in channels)):
                    self.stats_mean[:,i]=0
                    self.stats_std[:,i]=1

    def __call__(self, sample):
        datax, labelx = sample
        data = datax if self.input else labelx
        #assuming channel last
        if(self.stats_mean is not None):
            data = data - self.stats_mean
        if(self.stats_std is not None):
            data = data/self.stats_std

        if(self.input):
            return (data, labelx)
        else:
            return (datax, data)

def replace_labels(x, stay_idx, remove_idx):
    res = []
    for y in x:
        if y == remove_idx:
            res.append(stay_idx)
        else:
            res.append(y)
    return res

def keep_one_random_class(x):
    res = np.random.choice(x,1)[0]
    return res 

def transformations_from_strings_DINO(transformations, t_params, transform_type = 'global'):
    if transformations is None:
        return [ToTensor()]

    def str_to_trafo(trafo):
        if trafo == "RandomResizedCrop":
            return TRandomResizedCrop(crop_ratio_range=t_params["rr_crop_ratio_range"], output_size=t_params["output_size"])
        elif trafo == "ChannelResize":
            return TChannelResize(magnitude_range=t_params["magnitude_range"])
        elif trafo == "Negation":
            return TNegation()
        elif trafo == "DynamicTimeWarp":
            return TDynamicTimeWarp(warps=t_params["warps"], radius=t_params["radius"])
        elif trafo == "DownSample":
            return TDownSample(downsample_ratio=t_params["downsample_ratio"])
        elif trafo == "TimeWarp":
            return TimeWarp(epsilon=t_params["epsilon"])
        elif trafo == "TimeOut":
            return TTimeOut(crop_ratio_range=t_params["to_crop_ratio_range"])
        elif trafo == "TimeOut_difflead":
            return TTimeOut_difflead(crop_ratio_range=t_params["to_crop_ratio_range"])
        elif trafo == "TimeShift":
            return TTimeShift(shift_range=t_params["shift_range"])
        elif trafo == "ChannelReduction":
            return TChannelReduction()
        elif trafo == "GaussianNoise":
            return TGaussianNoise(scale=t_params["gaussian_scale"])
        elif trafo == "BaselineWander":
            return TBaselineWander(Cmax=t_params["bw_cmax"])
        elif trafo == "PowerlineNoise":
            return TPowerlineNoise(Cmax=t_params["pl_cmax"])
        elif trafo == "EMNoise":
            return TEMNoise(Cmax=t_params["em_cmax"])
        elif trafo == "BaselineShift":
            return TBaselineShift(Cmax=t_params["bs_cmax"])
        elif trafo == "GaussianBlur":
            return TGaussianBlur1d()
        elif trafo == "Normalize":
            return TNormalize(stats_mean=t_params["stats_mean"],stats_std=t_params["stats_std"])
        else:
            raise Exception(str(trafo) + " is not a valid transformation")

    if transform_type == 'global':
        trafo_list = [ToTensor(transpose_data=False)] + [TRandomResizedCrop(crop_ratio_range=t_params["global_crop_scale"], output_size=250)] + \
        [str_to_trafo(trafo) for trafo in transformations] + [Normalize(stats_mean=t_params["stats_mean"],stats_std=t_params["stats_std"])] + [Transpose()] 
    elif transform_type == 'local':
        trafo_list = [ToTensor(transpose_data=False)] + [TRandomResizedCrop(crop_ratio_range=t_params["local_crop_scale"], output_size=100)] + \
        [str_to_trafo(trafo) for trafo in transformations] + [Normalize(stats_mean=t_params["stats_mean"],stats_std=t_params["stats_std"])] + [Transpose()] 
    
    return trafo_list

class ECG_dataset_DINO_signal(torch.utils.data.Dataset):
    def __init__(self, summary_folder, signal_size, stride, chunk_length, transforms, t_params,
                 equivalent_classes, sample_items_per_record=1, random_crop=True):
        
        self.folder = summary_folder
        self.signal_size = signal_size
        self.transforms_global = transformations_from_strings_DINO(transforms, t_params, 'global')
        self.transforms_local = transformations_from_strings_DINO(transforms, t_params, 'local')
        # number of small samples we want to take out of the big signal data
        self.sample_items_per_record = sample_items_per_record
        # from the large signal data, we randomly choose where we acquire the sample data
        self.random_crop = random_crop

        # Loading data info 
        self.df = pickle.load(open(os.path.join(self.folder,"df_memmap.pkl"), "rb"))
        self.lbl_itos =  np.load(os.path.join(self.folder,"lbl_itos.npy"))
        self.mean = np.load(os.path.join(self.folder,"mean.npy"))
        self.std = np.load(os.path.join(self.folder,"std.npy"))        
        
        stack_remove_idx = []
        # Grouping the equivalent classes, remove the correspond classes
        if len(equivalent_classes)!=0:
            for i in range(len(equivalent_classes)):
                stay_class, remove_class = equivalent_classes[i]
                if stay_class not in self.lbl_itos or remove_class not in self.lbl_itos:
                    print(f'{stay_class},{remove_class}: one of those is not in the dictionary')
                else:
                    stay_idx = np.where(self.lbl_itos==stay_class)[0][0]
                    remove_idx = np.where(self.lbl_itos==remove_class)[0][0]
                    self.df['label'] = self.df['label'].apply(lambda x: replace_labels(x,stay_idx,remove_idx))
                    stack_remove_idx.append(remove_idx)
                    
        
        self.df['label'] = self.df['label'].apply(lambda x: keep_one_random_class(x))
        self.lbl_itos = np.delete(self.lbl_itos,stack_remove_idx)
       
      
        self.timeseries_df_data = np.array(self.df['data'])
        if(self.timeseries_df_data.dtype not in [np.int16, np.int32, np.int64]):
            self.timeseries_df_data = np.array(self.df["data"].astype(str)).astype(np.string_)

        #stack arrays/lists for proper batching
        if(isinstance(self.df['data'].iloc[0],list) or isinstance(self.df['label'].iloc[0],np.ndarray)):
            self.timeseries_df_label = np.stack(self.df['label'])
        else: # single integers/floats
            self.timeseries_df_label = np.array(self.df['label'])
        #everything else cannot be batched anyway mp.Manager().list(self.timeseries_df_label)
        if(self.timeseries_df_label.dtype not in [np.int16, np.int32, np.int64, np.float32, np.float64]): 
        #     assert(annotation and memmap_filename is None and npy_data is None)
            self.timeseries_df_label = np.array(self.df['label'].apply(lambda x:str(x))).astype(np.string_)
    
        # load meta data for memmap npy
        memmap_meta = np.load(os.path.join(self.folder,"memmap_meta.npz"), allow_pickle=True)
        self.memmap_start = memmap_meta["start"]
        self.memmap_shape = memmap_meta["shape"]
        self.memmap_length = memmap_meta["length"]
        self.memmap_file_idx = memmap_meta["file_idx"]
        self.memmap_dtype = np.dtype(str(memmap_meta["dtype"]))
        self.memmap_filenames = np.array(memmap_meta["filenames"]).astype(np.string_)#save as byte to avoid issue with mp

        # load data from memamp file 
        self.memmap_signaldata = np.memmap(os.path.join(self.folder,"memmap.npy"),self.memmap_dtype, mode="r", shape=tuple(self.memmap_shape[0]))

        # get the position of the signal inside the stack memmap signal data
        self.df_idx_mapping = []
        self.start_idx_mapping = []
        self.end_idx_mapping = []
        start_idx = 0
        min_chunk_length = signal_size
        
        for df_idx,(id,row) in enumerate(self.df.iterrows()):
            data_length = self.memmap_length[row["data"]]

            if(chunk_length == 0): # do not split into chunks
                idx_start = [start_idx]
                idx_end = [data_length]
            else:
                idx_start = list(range(start_idx,data_length,chunk_length if stride is None else stride))
                idx_end = [min(l+chunk_length, data_length) for l in idx_start]
            
            #remove final chunk(s) if too short
            for i in range(len(idx_start)):
                if(idx_end[i]-idx_start[i]< min_chunk_length):
                    del idx_start[i:]
                    del idx_end[i:]
                    break
            #append to lists
            copies = 0
            for _ in range(copies+1):
                for i_s,i_e in zip(idx_start,idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(i_s)
                    self.end_idx_mapping.append(i_e)
        
        #convert to np.array to avoid mp issues with python lists
        self.df_idx_mapping = np.array(self.df_idx_mapping)
        self.start_idx_mapping = np.array(self.start_idx_mapping)
        self.end_idx_mapping = np.array(self.end_idx_mapping)
    
    def __len__(self):
        return len(self.df_idx_mapping)

    @property
    def is_empty(self):
        return len(self.df_idx_mapping)==0

    def __getitem__(self, idx):
        lst_crops = []
        lst_lbl = []
        lst_patient = []
        for _ in range(self.sample_items_per_record):
            #determine crop idxs
            timesteps= self.get_sample_length(idx)

            if(self.random_crop): #random crop
                if(timesteps==self.signal_size):
                    start_idx_rel = 0
                else:
                    # get random start of the crop inside the big signal
                    start_idx_rel = random.randint(0, timesteps - self.signal_size -1)#np.random.randint(0, timesteps - self.output_size)
            else:
                # if not random, this may be for valid and the timesteps is probably equal to the signal_size
                start_idx_rel =  (timesteps - self.signal_size)//2 
            if(self.sample_items_per_record==1):
                crops, label, patient = self.get_signal_sample(idx,start_idx_rel)
                return {'crops':crops,'lbl':label,'idx':patient}
            else:
                crops, label, patient = self.get_signal_sample(idx,start_idx_rel)
                lst_crops.append(crops)
                lst_patient.append(patient)
                lst_lbl.append(label)
        lst_crops = torch.stack(lst_crops)
        lst_lbl = torch.from_numpy(np.stack(lst_lbl))

        return {'crops':lst_crops,'lbl':lst_lbl,'idx':lst_patient}

    def get_signal_sample(self, idx,start_idx_rel):
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]
        # determine crop idxs
        timesteps= end_idx - start_idx
        assert(timesteps>=self.signal_size)
        start_idx_crop = start_idx + start_idx_rel
        end_idx_crop = start_idx_crop+self.signal_size
        
        memmap_idx = self.timeseries_df_data[df_idx] 
        idx_offset = self.memmap_start[memmap_idx]

        signal_data = np.copy(self.memmap_signaldata[idx_offset + start_idx_crop: idx_offset + end_idx_crop])

        label = self.timeseries_df_label[df_idx]
        sample = (signal_data,label)

        crops = []
        for global_idx in range(2):
            copy_sample = (signal_data.copy(),label.copy())
            for trans in self.transforms_global:
                copy_sample = trans(copy_sample)
            aug_obj = copy_sample[0]
            crops.append(aug_obj)
        
        for local_idx in range(8):
            copy_sample = (signal_data.copy(),label.copy())
            for trans in self.transforms_local:
                copy_sample = trans(copy_sample)
            aug_obj = copy_sample[0]
            crops.append(aug_obj)

        return crops, label, df_idx

    def get_id_mapping(self):
        return self.df_idx_mapping

    def get_sample_id(self,idx):
        return self.df_idx_mapping[idx]

    def get_sample_length(self,idx):
        return self.end_idx_mapping[idx]-self.start_idx_mapping[idx]

    def get_sample_start(self,idx):
        return self.start_idx_mapping[idx]


######------------------------------------------------------------------------------------------------------------######
######------------------------------------------------------------------------------------------------------------######
######------------------------------------------------------------------------------------------------------------######
######------------------------------------------------------------------------------------------------------------######
######------------------------------------------------------------------------------------------------------------######
######------------------------------------------------------------------------------------------------------------######
######------------------------------------------------------------------------------------------------------------######
######------------------------------------------------------------------------------------------------------------######
######------------------------------------------------------------------------------------------------------------######



def spec_time_cutout(arr):
    spec_crop_ratio_range=[0.1, 0.35]
    new_arr = np.copy(arr)
    channels, freq , timesteps = new_arr.shape
    crop_ratios = [random.uniform(*spec_crop_ratio_range) for i in range(channels)]
    crop_timesteps = [int(cr*timesteps) for cr in crop_ratios]
    start_idxes = [random.randint(0, timesteps - ct-1) for ct in crop_timesteps]
    for ch in range(channels):
        new_arr[ch,:,start_idxes[ch]:start_idxes[ch]+crop_timesteps[ch]] = 0
    
    return new_arr

def spec_freq_cutout(arr):
    new_arr = np.copy(arr)
    channels, freq, timesteps = new_arr.shape
    crop_freq = [random.randint(0, freq-1) for i in range(channels)]
    
    for ch in range(channels):
        new_arr[ch,crop_freq[ch],:] = 0
    
    return new_arr

def spec_time_shift(arr):
    shift_range = [0.1,0.5]
    channels, freq , timesteps = arr.shape
    shift_ratios = [random.uniform(*shift_range) for i in range(channels)]
    shift_timesteps =  [int(sh*timesteps) for sh in shift_ratios]
    zero_arr = np.zeros((channels, freq, timesteps))
    for ch in range(channels):
        zero_arr[ch,:,0:shift_timesteps[ch]] = arr[ch,:,timesteps - shift_timesteps[ch]-1:timesteps-1]
        zero_arr[ch,:,shift_timesteps[ch]:timesteps] = arr[ch,:,0:timesteps - shift_timesteps[ch]]

    return zero_arr

def random_crop(arr,scale):
    new_arr = np.copy(arr)
    channels, freq, timesteps = new_arr.shape

    crop_ratio = random.uniform(*scale)
    crop_range = int(crop_ratio*timesteps) 
    start_idx = random.randint(0, timesteps - crop_range-1)
    cropped_arr = new_arr[:,:,start_idx:start_idx+crop_range]
    
    return cropped_arr


def crop_padding(arr,ori_arr):
    channels, freq, timesteps = ori_arr.shape
    _, _, timesteps_r = arr.shape

    left_padding = (timesteps - timesteps_r)//2

    padded_arr = np.zeros((channels, freq, timesteps))
    for ch in range(channels):
        padded_arr[ch][:,left_padding:left_padding+timesteps_r] = arr[ch]

    return padded_arr


class ECG_dataset_DINO_spectrogram(torch.utils.data.Dataset):
    def __init__(self, summary_folder, signal_size, stride, chunk_length,
                 equivalent_classes, sample_items_per_record=1, random_crop=True):
        
        self.folder = summary_folder
        self.signal_size = signal_size
        # number of small samples we want to take out of the big signal data
        self.sample_items_per_record = sample_items_per_record
        # from the large signal data, we randomly choose where we acquire the sample data
        self.random_crop = random_crop

        # Loading data info 
        self.df = pickle.load(open(os.path.join(self.folder,"df_memmap.pkl"), "rb"))
        self.lbl_itos =  np.load(os.path.join(self.folder,"lbl_itos.npy"))
        self.mean = np.load(os.path.join(self.folder,"mean.npy"))
        self.std = np.load(os.path.join(self.folder,"std.npy"))        

        self.mean_stft = [0.01696577, 0.018700505, 0.01614143, 0.016069638, 0.013717703, 0.015169965, 0.021021612,\
                             0.031449795, 0.032050893, 0.031188736, 0.028222399, 0.024229238]
        self.std_stft = [0.05418394, 0.05418394, 0.05418394, 0.05418394, 0.05418394, 0.05418394, 0.05418394,\
                             0.05418394, 0.05418394, 0.05418394, 0.05418394, 0.05418394]
        
        stack_remove_idx = []
        # Grouping the equivalent classes, remove the correspond classes
        if len(equivalent_classes)!=0:
            for i in range(len(equivalent_classes)):
                stay_class, remove_class = equivalent_classes[i]
                if stay_class not in self.lbl_itos or remove_class not in self.lbl_itos:
                    print(f'{stay_class},{remove_class}: one of those is not in the dictionary')
                else:
                    stay_idx = np.where(self.lbl_itos==stay_class)[0][0]
                    remove_idx = np.where(self.lbl_itos==remove_class)[0][0]
                    self.df['label'] = self.df['label'].apply(lambda x: replace_labels(x,stay_idx,remove_idx))
                    stack_remove_idx.append(remove_idx)
                    
        
        self.df['label'] = self.df['label'].apply(lambda x: keep_one_random_class(x))
        self.lbl_itos = np.delete(self.lbl_itos,stack_remove_idx)
       
      
        self.timeseries_df_data = np.array(self.df['data'])
        if(self.timeseries_df_data.dtype not in [np.int16, np.int32, np.int64]):
            self.timeseries_df_data = np.array(self.df["data"].astype(str)).astype(np.string_)

        #stack arrays/lists for proper batching
        if(isinstance(self.df['data'].iloc[0],list) or isinstance(self.df['label'].iloc[0],np.ndarray)):
            self.timeseries_df_label = np.stack(self.df['label'])
        else: # single integers/floats
            self.timeseries_df_label = np.array(self.df['label'])
        #everything else cannot be batched anyway mp.Manager().list(self.timeseries_df_label)
        if(self.timeseries_df_label.dtype not in [np.int16, np.int32, np.int64, np.float32, np.float64]): 
        #     assert(annotation and memmap_filename is None and npy_data is None)
            self.timeseries_df_label = np.array(self.df['label'].apply(lambda x:str(x))).astype(np.string_)
    
        # load meta data for memmap npy
        memmap_meta = np.load(os.path.join(self.folder,"memmap_meta.npz"), allow_pickle=True)
        self.memmap_start = memmap_meta["start"]
        self.memmap_shape = memmap_meta["shape"]
        self.memmap_length = memmap_meta["length"]
        self.memmap_file_idx = memmap_meta["file_idx"]
        self.memmap_dtype = np.dtype(str(memmap_meta["dtype"]))
        self.memmap_filenames = np.array(memmap_meta["filenames"]).astype(np.string_)#save as byte to avoid issue with mp

        # load data from memamp file 
        self.memmap_signaldata = np.memmap(os.path.join(self.folder,"memmap.npy"),self.memmap_dtype, mode="r", shape=tuple(self.memmap_shape[0]))

        # get the position of the signal inside the stack memmap signal data
        self.df_idx_mapping = []
        self.start_idx_mapping = []
        self.end_idx_mapping = []
        start_idx = 0
        min_chunk_length = signal_size
        
        for df_idx,(id,row) in enumerate(self.df.iterrows()):
            data_length = self.memmap_length[row["data"]]

            if(chunk_length == 0): # do not split into chunks
                idx_start = [start_idx]
                idx_end = [data_length]
            else:
                idx_start = list(range(start_idx,data_length,chunk_length if stride is None else stride))
                idx_end = [min(l+chunk_length, data_length) for l in idx_start]
            
            #remove final chunk(s) if too short
            for i in range(len(idx_start)):
                if(idx_end[i]-idx_start[i]< min_chunk_length):
                    del idx_start[i:]
                    del idx_end[i:]
                    break
            #append to lists
            copies = 0
            for _ in range(copies+1):
                for i_s,i_e in zip(idx_start,idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(i_s)
                    self.end_idx_mapping.append(i_e)
        
        #convert to np.array to avoid mp issues with python lists
        self.df_idx_mapping = np.array(self.df_idx_mapping)
        self.start_idx_mapping = np.array(self.start_idx_mapping)
        self.end_idx_mapping = np.array(self.end_idx_mapping)
    
    def __len__(self):
        return len(self.df_idx_mapping)

    @property
    def is_empty(self):
        return len(self.df_idx_mapping)==0

    def __getitem__(self, idx):
        lst_crops = []
        lst_lbl = []
        lst_patient = []
        for _ in range(self.sample_items_per_record):
            #determine crop idxs
            timesteps= self.get_sample_length(idx)

            if(self.random_crop): #random crop
                if(timesteps==self.signal_size):
                    start_idx_rel = 0
                else:
                    # get random start of the crop inside the big signal
                    start_idx_rel = random.randint(0, timesteps - self.signal_size -1)#np.random.randint(0, timesteps - self.output_size)
            else:
                # if not random, this may be for valid and the timesteps is probably equal to the signal_size
                start_idx_rel =  (timesteps - self.signal_size)//2 
            if(self.sample_items_per_record==1):
                crops, label, patient = self.get_signal_sample(idx,start_idx_rel)
                return {'crops':crops,'lbl':label,'idx':patient}
            else:
                crops, label, patient = self.get_signal_sample(idx,start_idx_rel)
                lst_crops.append(crops)
                lst_patient.append(patient)
                lst_lbl.append(label)
        lst_crops = torch.stack(lst_crops)
        lst_lbl = torch.from_numpy(np.stack(lst_lbl))

        return {'crops':lst_crops,'lbl':lst_lbl,'idx':lst_patient}

    def get_signal_sample(self, idx,start_idx_rel):
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]
        # determine crop idxs
        timesteps= end_idx - start_idx
        assert(timesteps>=self.signal_size)
        start_idx_crop = start_idx + start_idx_rel
        end_idx_crop = start_idx_crop+self.signal_size
        
        memmap_idx = self.timeseries_df_data[df_idx] 
        idx_offset = self.memmap_start[memmap_idx]

        signal_data = np.copy(self.memmap_signaldata[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
        label = self.timeseries_df_label[df_idx]
  
        # Short Time Fast Fourier Transform for 12 channels of ECG signal
        f,t, Zxx = stft(signal_data.transpose(1,0),fs=100, window='hann',nperseg=25)
        stft_data = np.abs(Zxx)
        # normalize / must have channel last 
        stft_data = stft_data.transpose(1,2,0)
        stft_data = stft_data - self.mean_stft
        stft_data = stft_data/self.std_stft
        stft_data = stft_data.transpose(2,0,1)
        # stft_data = torch.from_numpy(stft_data)
       
        global_crops_scale = [0.6,1.]
        local_crops_scale = [0.3,0.6]

        crops = []
        for global_idx in range(2):
            crop_arr = random_crop(stft_data,global_crops_scale)

            aug_arr = spec_freq_cutout(crop_arr)
            aug_arr = spec_time_cutout(crop_arr)
            # aug_arr = spec_time_shift(crop_arr)

            pad_arr = crop_padding(aug_arr,stft_data)
            pad_arr = torch.from_numpy(pad_arr)
            crops.append(pad_arr) 

        for local_idx in range(8):
            crop_arr = random_crop(stft_data,local_crops_scale)

            aug_arr = spec_freq_cutout(crop_arr)
            aug_arr = spec_time_cutout(crop_arr)
            # aug_arr = spec_time_shift(crop_arr)

            pad_arr = crop_padding(aug_arr,stft_data)
            pad_arr = torch.from_numpy(pad_arr)
            crops.append(pad_arr) 

        return crops, label, df_idx

    def get_id_mapping(self):
        return self.df_idx_mapping

    def get_sample_id(self,idx):
        return self.df_idx_mapping[idx]

    def get_sample_length(self,idx):
        return self.end_idx_mapping[idx]-self.start_idx_mapping[idx]

    def get_sample_start(self,idx):
        return self.start_idx_mapping[idx]

