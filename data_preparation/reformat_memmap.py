import numpy as np 
import os 
import pandas as pd 
import pickle
from tqdm import tqdm

def npys_to_memmap(npys, target_filename, max_len=0, delete_npys=True):
    memmap = None
    start = []#start_idx in current memmap file
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]

    for idx,npy in tqdm(list(enumerate(npys))):
        data = np.load(npy, allow_pickle=True)
        if(memmap is None or (max_len>0 and start[-1]+length[-1]>max_len)):
            filenames.append(target_filename)

            if(memmap is not None):#an existing memmap exceeded max_len
                shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
                del memmap
            #create new memmap
            start.append(0)
            length.append(data.shape[0])
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
        else:
            #append to existing memmap
            start.append(start[-1]+length[-1])
            length.append(data.shape[0])
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple([start[-1]+length[-1]]+[l for l in data.shape[1:]]))

        #store mapping memmap_id to memmap_file_id
        file_idx.append(len(filenames)-1)
        #insert the actual data
        memmap[start[-1]:start[-1]+length[-1]]=data[:]
        memmap.flush()
        if(delete_npys is True):
            npy.unlink()
    del memmap

    #append final shape if necessary
    if(len(shape)<len(filenames)):
        shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
    #convert everything to relative paths
    filenames= [f.split('/')[-1] for f in filenames]
    #save metadata
    last_name = target_filename.split('/')[-1].split('.')[0]
    parent_name = target_filename.split(last_name)[0]
    np.savez(parent_name+last_name+"_meta.npz",start=start,length=length,shape=shape,file_idx=file_idx,dtype=data.dtype,filenames=filenames)

if __name__ == "__main__":
    data_type = "without_preprocessing"
    # data_type = "with_preprocessing"

    save_folder = "./data/extracted_data_" + data_type
    save_summary = "./data/data_summary_"  + data_type


    df = pickle.load(open(os.path.join(save_summary,"df.pkl"), "rb"))
    lbl_itos = np.load(os.path.join(save_summary,"lbl_itos.npy"))
    mean = np.load(os.path.join(save_summary,"mean.npy"))
    std = np.load(os.path.join(save_summary,"std.npy"))

    npys_data = []
    npys_label = []

    for id,row in df.iterrows():
        npys_data.append(os.path.join(save_folder,row["data"]))

    npys_to_memmap(npys_data, os.path.join(save_summary,"memmap.npy"), max_len=0, delete_npys=False)

    df_mapped = df.copy()
    df_mapped["data_original"]=df_mapped.data
    df_mapped["data"]=np.arange(len(df_mapped))

    df_mapped.to_pickle(os.path.join(save_summary,"df_memmap.pkl"))