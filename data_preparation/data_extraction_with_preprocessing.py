import os 
import glob
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import wfdb
from scipy.signal import resample
import pywt
from stratify import stratify

save_folder = "./data_folder/extracted_data_with_preprocessing"
save_summary =  "./data_folder/data_summary_with_preprocessing"
raw_data_cinc = "./data_folder/datasets"
dataset_names = ["ICBEB2018","ICBEB2018_2","INCART","PTB","PTB-XL","Georgia"]
mapping_scored_path = "./data_folder/evaluation-2020-master/dx_mapping_scored.csv"    # 27 main labels
target_fs = 100
strat_folds = 10
channels = 12

mapping_scored_df = pd.read_csv(mapping_scored_path)
dx_mapping_snomed_abbrev = {a:b for [a,b] in list(mapping_scored_df.apply(lambda row: [row["SNOMED CT Code"],row["Abbreviation"]],axis=1))}
list_label_available = np.array(mapping_scored_df["SNOMED CT Code"])

CPSC_files = glob.glob(os.path.join(raw_data_cinc,dataset_names[0],'**/*.hea'))
print('No files in CPSC:', len(CPSC_files))
CPSC_extra_files = glob.glob(os.path.join(raw_data_cinc,dataset_names[1],'**/*.hea'))
print('No files in CPSC-Extra:', len(CPSC_extra_files))
SPeter_files = glob.glob(os.path.join(raw_data_cinc,dataset_names[2],'**/*.hea'))
print('No files in StPetersburg:', len(SPeter_files))
PTB_files = glob.glob(os.path.join(raw_data_cinc,dataset_names[3],'**/*.hea'))
print('No files in PTB:', len(PTB_files))
PTBXL_files = glob.glob(os.path.join(raw_data_cinc,dataset_names[4],'**/*.hea'))
print('No files in PTB-XL:', len(PTBXL_files))
Georgia_files = glob.glob(os.path.join(raw_data_cinc,dataset_names[5],'**/*.hea'))
print('No files in Georgia:', len(Georgia_files))

all_files = CPSC_files + CPSC_extra_files + SPeter_files + PTB_files + PTBXL_files + Georgia_files
print('Total no files:',len(all_files))
# (7500, 12)
# {'fs': 500, 'sig_len': 7500, 'n_sig': 12, 'base_date': None, 'base_time': datetime.time(0, 0, 12), 
# 'units': ['mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV'], 
# 'sig_name': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], 
# 'comments': ['Age: 74', 'Sex: Male', 'Dx: 59118001', 'Rx: Unknown', 'Hx: Unknown', 'Sx: Unknown']}

# CSPC fs 500 level 3 stride 50
# CSPC extra fs 500 level 3 stride 50
# StPeter fs 257 level 2 stride 30
# PTB fs 1000 level 2 stride 100
# PTB XL fs 500 level 3 stride 50
# Grorgia fs 500 level 3 stride 50

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w    
    

skip_files = 0
metadata = []
for idx, hea_file in enumerate(tqdm(all_files)):
    file_name = hea_file.split("/")[-1].split(".hea")[0]
    data_folder = hea_file.split("/")[-3]
    sigbufs, header = wfdb.rdsamp(str(hea_file)[:-4])

    if(np.any(np.isnan(sigbufs))):
        print("Warning:",str(hea_file),"is corrupt. Skipping.")
        continue

    labels=[]
    age=np.nan
    sex="nan"
    for l in header["comments"]:
        arrs = l.strip().split(' ')
        if l.startswith('Dx:'):
            for x in arrs[1].split(','):
                if int(x) in list_label_available:
                    labels.append(x)
        elif l.startswith('Age:'):
            try:
                age = int(arrs[1])
            except:
                age= np.nan
        elif l.startswith('Sex:'):
            sex = arrs[1].strip().lower()
            if(sex=="m"):
                sex="male"
            elif(sex=="f"):
                sex="female"
    
    if len(labels) == 0:
        skip_files += 1 
        continue

    if data_folder == "ICBEB2018":
        level = 3
        stride = 50
    elif data_folder == "ICBEB2018_2":
        level = 3
        stride = 50
    elif data_folder == "INCART":
        level = 3
        stride = 30
    elif data_folder == "PTB":
        level = 3
        stride = 100
    elif data_folder == "PTB-XL":
        level = 3
        stride = 50
    elif data_folder == "Georgia":
        level = 3
        stride = 50
    
    
    # DENOISE
    # Create wavelet object and define parameters
    w = pywt.Wavelet('db4')
    maxlev = pywt.dwt_max_level(len(sigbufs[:,0]), w.dec_len)
    denoised_data = np.zeros((len(sigbufs), channels), dtype=np.float32)
    # Decompose into wavelet components, to the level selected:
    for cha in range(channels):
        coeffs = pywt.wavedec(sigbufs[:,cha], 'db4', mode='periodic',level=maxlev)
        
        sigma = (1/0.6745) * madev(coeffs[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(sigbufs[:,cha])))
        
        coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])
        
        datarec = pywt.waverec(coeffs, 'db4')
        if len(datarec) < len(sigbufs):
            datarec = np.pad(datarec,len(sigbufs)-len(datarec),'edge')
            denoised_data[:,cha] = datarec
        elif len(datarec) > len(sigbufs):
            denoised_data[:,cha] = datarec[0:len(sigbufs)]
        else:
            denoised_data[:,cha] = datarec
    
    # BASELINE WANDER REMOVAL
    baseline_removal_data = np.zeros((len(sigbufs), channels), dtype=np.float32)
    for cha in range(channels):
        avg_output = moving_average(denoised_data[:,cha],stride)
        avg_pad = np.pad(avg_output,(0,len(sigbufs[:,cha])-len(avg_output)),'edge')
        baseline_removal_data[:,cha] = denoised_data[:,cha]- avg_pad


    ori_fs = header['fs']
    factor = target_fs/ori_fs
    timesteps_new = int(len(sigbufs)*factor)
    data = np.zeros((timesteps_new, channels), dtype=np.float32)
    for i in range(channels):
        data[:,i] = resample(baseline_removal_data[:,0],timesteps_new)
    
    np.save(os.path.join(save_folder,file_name+".npy"),data)

    metadata.append({"data":file_name+".npy","label":labels,"sex":sex,"age":age,"dataset":data_folder})

df =pd.DataFrame(metadata)
lbl_itos = np.unique([item for sublist in list(df.label) for item in sublist])
lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
df["label"] = df["label"].apply(lambda x: [lbl_stoi[y] for y in x])

df["strat_fold"]=-1
for ds in np.unique(df["dataset"]):
    print("Creating CV folds:",ds)
    dfx = df[df.dataset==ds]
    idxs = np.array(dfx.index.values)
    lbl_itosx = np.unique([item for sublist in list(dfx.label) for item in sublist])
    stratified_ids = stratify(list(dfx["label"]), lbl_itosx, [1./strat_folds]*strat_folds)

    for i,split in enumerate(stratified_ids):
        df.loc[idxs[split],"strat_fold"]=i

print("Add Mean Column")
df["data_mean"]=df["data"].apply(lambda x: np.mean(np.load(x if save_folder is None else os.path.join(save_folder,x), allow_pickle=True),axis=0))
print("Add Std Column")
df["data_std"]=df["data"].apply(lambda x: np.std(np.load(x if data_folder is None else os.path.join(save_folder,x), allow_pickle=True),axis=0))
print("Add Length Column")
df["data_length"]=df["data"].apply(lambda x: len(np.load(x if data_folder is None else os.path.join(save_folder,x), allow_pickle=True)))

#save means and stds
df_mean = df["data_mean"].mean()
df_std = df["data_std"].mean()

# save dataset
df.to_pickle(os.path.join(save_summary,'df.pkl'),protocol=4)
np.save(os.path.join(save_summary,"lbl_itos.npy"),lbl_itos)
np.save(os.path.join(save_summary,"mean.npy"),df_mean)
np.save(os.path.join(save_summary,"std.npy"),df_std)

# file1 = 'df.pkl'
# file2 = 'lbl_itos.npy'
# file3 = 'memmap.npy'
# file4 = 'memmap_meta.npz'
# file5 = 'df_memmap.pkl'
# file6 = 'mean.npy'
# file7 = 'std.npy'

