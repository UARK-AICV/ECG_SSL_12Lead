Multimodality Multi-Lead ECG Arrhythmia Classification using Self-Supervised Learning
Paper link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9926925

1. Download datasets from the PhysioNet 2020 Competition. Put in the folder ./data_folder/datasets and extract all of them .
https://physionetchallenges.github.io/2020/

2. Preparing the data 
python data_preparation/data_extraction_without_preprocessing.py
python data_preparation/reformat_memmap.py

3. Training base models 
python experiments/run_signal.py --batch_size 128 --lr_rate 5e-3 --num_epoches 100 --gpu 0 --save_folder ./checkpoints/base_signal
python experiments/run_spectrogram.py --batch_size 256 --lr_rate 5e-3 --num_epoches 200 --gpu 0 --save_folder ./checkpoints/base_spectrogram
(without gating fusion)
python experiments/run_ensembled.py --batch_size 128 --lr_rate 5e-3 --num_epoches 100 --gpu 0 --save_folder ./checkpoints/base_ensemble_wogating
(with gating fusion)
python experiments/run_ensembled.py --batch_size 128 --lr_rate 5e-3 --num_epoches 100 --gpu 0 --gating --save_folder ./checkpoints/base_ensemble_wgating

4. Self-supervised learning for pretrained models
(SimCLR)
python experiments/SIMCLR_signal.py
(BYOL)
python experiments/BYOL_signal.py
(DINO)
python experiments/DINO_signal.py
python experiments/DINO_spectrogram.py

5. Finetuning the main model based on the self-supervised pretrained models
(SimCLR)
python experiments/SIMCLR_signal_finetune.py
(BYOL)
python experiments/BYOL_signal_finetune.py
(DINO)
python experiments/run_signal.py --batch_size 128 --lr_rate 5e-3 --num_epoches 100 --gpu 0 --finetune ./checkpoints/DINO_signal_student.pth --save_folder ./checkpoints/finetune_signal
python experiments/run_spectrogram.py --batch_size 256 --lr_rate 5e-3 --num_epoches 200 --gpu 0 --finetune ./checkpoints/DINO_spectrogram_student.pth --save_folder ./checkpoints/finetune_spectrogram
(without gating fusion)
python experiments/run_ensembled.py --batch_size 128 --lr_rate 5e-3 --num_epoches 100 --gpu 0 --finetune ./checkpoints --save_folder ./checkpoints/finetune_ensemble_wogating
(with gating fusion)
python experiments/run_ensembled.py --batch_size 128 --lr_rate 5e-3 --num_epoches 100 --gpu 0 --finetune ./checkpoints --gating --save_folder ./checkpoints/finetune_ensemble_wgating

6. Searching the thresholds of classes for best Challenge score
python experiments/threshold_search.py --model_type signal --best-type PRC --gpu 0 --weight_folder ./checkpoints/base_signal
