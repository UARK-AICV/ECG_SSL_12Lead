from signal import signal
import numpy as np 
import torch
import torch.utils.data
from torchvision import transforms
import os 
import pickle
import os 
from tqdm import tqdm
import random
from scipy.signal import stft

from .timeseries_transformations import ToTensor

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

def multihot_encode(x, new_num_classes, convert_dict):
    converted_arr = []
    for y in x:
        converted_arr.append(convert_dict[y])

    res = np.zeros(new_num_classes, dtype=np.float32)
    for cnv in converted_arr:
        res[cnv] = 1
    
    return res

class ECG_dataset_base(torch.utils.data.Dataset):
    def __init__(self, summary_folder, classes, signal_size, stride, chunk_length, transforms, stft_inc, meta_inc, 
                t_or_v, equivalent_classes, sample_items_per_record=1, preload=False,random_crop=True, val_fold=0):
        
        self.folder = summary_folder
        self.signal_size = signal_size
        self.transforms = transforms
        self.t_or_v = t_or_v
        self.stft_inc = stft_inc 
        self.meta_inc = meta_inc
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

        # Choos the strat fold data
        if t_or_v == 'train':
            self.df = self.df[self.df.strat_fold!=val_fold]
        elif t_or_v == 'val':
            self.df = self.df[self.df.strat_fold==val_fold]
        
        
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
                    
        # put the order of lbl_itos following the order of Challenge matrix
        convert = [np.where(self.lbl_itos==classes[idx])[0][0] for idx in range(len(classes))]
        # from lbl_itos to classes
        convert_dict = dict(zip(convert,np.arange(len(classes))))
        
        self.df['label'] = self.df['label'].apply(lambda x: multihot_encode(x,len(classes),convert_dict))
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
        
        if (isinstance(self.transforms,list) or isinstance(self.transforms,np.ndarray)):
            print("Warning: the use of list as arguments for transforms is dicouraged")
    
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
        self.meta_idx = []
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
            
            if row['age'] != row['age']:
                meta_age = 0.
            elif row['age'] == -1.:
                meta_age = 0.
            else:
                meta_age = row['age']/104.0

            if row['sex'] != row['sex']:
                meta_sex = 'nan'
            else:
                meta_sex = row['sex']
            
            #append to lists
            copies = 0
            for _ in range(copies+1):
                for i_s,i_e in zip(idx_start,idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(i_s)
                    self.end_idx_mapping.append(i_e)
                    self.meta_idx.append([meta_age,meta_sex])
        
        #convert to np.array to avoid mp issues with python lists
        self.df_idx_mapping = np.array(self.df_idx_mapping)
        self.start_idx_mapping = np.array(self.start_idx_mapping)
        self.end_idx_mapping = np.array(self.end_idx_mapping)
        self.meta_idx = np.array(self.meta_idx)

        self.preload = preload
        if preload==True:
            self.preload_signal_data = []
            self.preload_stft_data = []
            
            print("Preloading data ...")

            if(chunk_length==self.signal_size):
                start_idx_rel = 0
            else:
                # get random start of the crop inside the big signal
                start_idx_rel = random.randint(0, chunk_length - self.signal_size -1)

            for idx in tqdm(range(len(self.df_idx_mapping))):
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
                self.preload_signal_data.append(signal_data)

                # Short Time Fast Fourier Transform for 12 channels of ECG signal
                f,t, Zxx = stft(signal_data.transpose(1,0),fs=100, window='hann',nperseg=25)
                stft_data = np.abs(Zxx)
                # normalize / must have channel last 
                stft_data = stft_data.transpose(1,2,0)
                stft_data = stft_data - self.mean_stft
                stft_data = stft_data/self.std_stft
                stft_data = stft_data.transpose(2,0,1)
                stft_data = torch.from_numpy(stft_data)
                self.preload_stft_data.append(stft_data)

            print("Finsh loading data !!!")
    
    def __len__(self):
        return len(self.df_idx_mapping)

    @property
    def is_empty(self):
        return len(self.df_idx_mapping)==0

    def __getitem__(self, idx):
        if self.preload:
            signal_data = self.preload_signal_data[idx]
            stft_data = self.preload_stft_data[idx]
            df_idx = self.df_idx_mapping[idx]
            meta_idx = self.meta_idx[idx]
            label = self.timeseries_df_label[df_idx]

            # Meta data
            meta_vector = np.zeros((5),dtype=np.float)
            if meta_idx[0] == 0.:    # check is NaN
                meta_vector[1] = 1.0
            else:
                meta_vector[0] = meta_idx[0]
                    
            if meta_idx[1] == 'nan':    # check is NaN
                meta_vector[4] = 1.0
            elif meta_idx[1] == 'male':
                meta_vector[2] = 1.0
            else:
                meta_vector[3] = 1.0
            meta_vector = torch.from_numpy(meta_vector)

            sample = (signal_data,label)

            get_transform = transforms.Compose([Normalize(self.mean, self.std),ToTensor()])
            if self.transforms == True:
                sample = get_transform(sample)
            return {'sig':sample[0],'stft':stft_data,'meta':meta_vector,'lbl':sample[1],'idx':df_idx}
        else:
            lst_sig_data = []
            lst_stft_data = []
            lst_meta_data = []
            lst_label = []
            lst_patient = []
            for _ in range(self.sample_items_per_record):
                # determine crop idxs
                timesteps= self.get_sample_length(idx)

                if(self.random_crop): # random crop
                    if(timesteps==self.signal_size):
                        start_idx_rel = 0
                    else:
                        # get random start of the crop inside the big signal
                        start_idx_rel = random.randint(0, timesteps - self.signal_size -1)#np.random.randint(0, timesteps - self.output_size)
                else:
                    # if not random, this may be for valid and the timesteps is probably equal to the signal_size
                    start_idx_rel =  (timesteps - self.signal_size)//2 
                if(self.sample_items_per_record==1):
                    sig_data, label, stft_data, meta_data, patient = self.get_signal_sample(idx,start_idx_rel)
                    if stft_data is None:
                        stft_data = 0
                    return {'sig':sig_data,'stft':stft_data,'meta':meta_data,'lbl':label,'idx':patient}
                else:
                    sig_data, label, stft_data, meta_data, patient = self.get_signal_sample(idx,start_idx_rel)
                    lst_sig_data.append(sig_data)
                    if stft_data is not None:
                        lst_stft_data.append(stft_data)
                    lst_label.append(label)
                    lst_patient.append(patient)
                    lst_meta_data.append(meta_data)
            lst_sig_data = torch.stack(lst_sig_data)
            lst_meta_data = torch.stack(lst_meta_data)
            if len(lst_stft_data) > 0:
                lst_stft_data = torch.stack(lst_stft_data)
            lst_label = torch.stack(lst_label)

            return {'sig':lst_sig_data,'stft':lst_stft_data,'meta':lst_meta_data,'lbl':lst_label,'idx':lst_patient}

    def get_signal_sample(self, idx,start_idx_rel):
        # low-level function that actually fetches the data
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]
        meta_idx = self.meta_idx[idx]
        # determine crop idxs
        timesteps= end_idx - start_idx
        assert(timesteps>=self.signal_size)
        start_idx_crop = start_idx + start_idx_rel
        end_idx_crop = start_idx_crop+self.signal_size

        # grab the actual index (Note the df to create the ds might be a subset of the original df used to create the memmap)
        memmap_idx = self.timeseries_df_data[df_idx] 
        idx_offset = self.memmap_start[memmap_idx]

        signal_data = np.copy(self.memmap_signaldata[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
        
        # Meta data
        meta_vector = np.zeros((5),dtype=np.float)
        if meta_idx[0] == 0.:    # check is NaN
            meta_vector[1] = 1.0
        else:
            meta_vector[0] = meta_idx[0]
                
        if meta_idx[1] == 'nan':    # check is NaN
            meta_vector[4] = 1.0
        elif meta_idx[1] == 'male':
            meta_vector[2] = 1.0
        else:
            meta_vector[3] = 1.0
        meta_vector = torch.from_numpy(meta_vector)

        # Short Time Fourier Transform
        if self.stft_inc:
            # Short Time Fast Fourier Transform for 12 channels of ECG signal
            f,t, Zxx = stft(signal_data.transpose(1,0),fs=100, window='hann',nperseg=25)
            stft_data = np.abs(Zxx)
            # normalize / must have channel last 
            stft_data = stft_data.transpose(1,2,0)
            stft_data = stft_data - self.mean_stft
            stft_data = stft_data/self.std_stft
            stft_data = stft_data.transpose(2,0,1)
            stft_data = torch.from_numpy(stft_data)
        else:
            stft_data = None

        label = self.timeseries_df_label[df_idx]
        sample = (signal_data,label)

        get_transform = transforms.Compose([Normalize(self.mean, self.std),ToTensor()])
        if self.transforms == True:
            sample = get_transform(sample)

        return sample[0], sample[1], stft_data, meta_vector, df_idx

    def get_id_mapping(self):
        return self.df_idx_mapping

    def get_sample_id(self,idx):
        return self.df_idx_mapping[idx]

    def get_sample_length(self,idx):
        return self.end_idx_mapping[idx]-self.start_idx_mapping[idx]

    def get_sample_start(self,idx):
        return self.start_idx_mapping[idx]

    def get_num_classes(self):
        return len(self.lbl_itos)
