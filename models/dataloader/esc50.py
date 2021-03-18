######################################
## 数据文件夹下ESC50文件夹的名字. ##
######################################
ESC50_DATA_PATH_NAME = 'esc50'
######################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import os
import os.path as osp
import numpy as np
import imageio
import random
import collections
import csv
import librosa

import transform.wav as wav_transforms

from models.sampler import ShotTaskSamplerForList


class ESC50(Dataset):
    def __init__(self, stype, args):
        # 50 类, 每类一共40个.
        if stype not in ('train', 'val', 'test'):
            raise(ValueError, 'stype must be one of (train, val, test)')
        self.sampler_handle = ShotTaskSamplerForList
        self.data_path = args.data_path
        self.stype = stype

        self.wave_path = osp.join(self.data_path, f'{ESC50_DATA_PATH_NAME}/audio')

        train_folds = ['2', '3', '4', '5']
        val_folds = ['1']
        test_folds = ['1']
        cur_folds = eval(f'{stype}_folds')

        wav_file_list = os.listdir(self.wave_path)
        wav_file_list.sort()
        self.file_names = [i for i in wav_file_list if i.split('-')[0] in cur_folds]
        self.label = [int(i.split('.')[0].split('-')[-1]) for i in self.file_names]
        self.num_classes = len(set(self.label))

        class TransformConfig():
            def __init__(self):
                self.freq_masks_width = 32
                self.freq_masks = 2
                self.time_masks_width = 32
                self.time_masks = 1
        my_config = TransformConfig()

        if stype == 'train':
            self.wave_transforms = transforms.Compose([
                wav_transforms.ToTensor1D(),
                wav_transforms.RandomScale(max_scale = 1.25),
                wav_transforms.RandomPadding(out_len = 220500),
                wav_transforms.RandomCrop(out_len = 220500)
            ])

            self.spec_transforms = transforms.Compose([
                transforms.ToTensor(),
                wav_transforms.FrequencyMask(max_width = my_config.freq_masks_width, numbers = my_config.freq_masks), 
                wav_transforms.TimeMask(max_width = my_config.time_masks_width, numbers = my_config.time_masks)
            ])

        else:
            self.wave_transforms = transforms.Compose([
                wav_transforms.ToTensor1D(),
                wav_transforms.RandomPadding(out_len = 220500),
                wav_transforms.RandomCrop(out_len = 220500)
            ])

            self.spec_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        wave, rate = librosa.load(self.wave_path + self.file_names[index], sr=44100)
        class_id = self.label[index]
        
        if wave.ndim == 1:
            wave = wave[:, np.newaxis]
		
	    # normalizing waves to [-1, 1]
        if np.abs(wave.max()) > 1.0:
            wave = wav_transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
        wave = wave.T * 32768.0
        
        # Remove silent sections
        start = wave.nonzero()[1].min()
        end = wave.nonzero()[1].max()
        wave = wave[:, start: end + 1]  
        
        wave_copy = np.copy(wave)
        wave_copy = self.wave_transforms(wave_copy)
        wave_copy.squeeze_(0)
        
        s = librosa.feature.melspectrogram(wave_copy.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512) 
        log_s = librosa.power_to_db(s, ref=np.max)
        
	    # masking the spectrograms
        log_s = self.spec_transforms(log_s)

        # creating 3 channels by copying log_s1 3 times 
        spec = torch.cat((log_s, log_s, log_s), dim=0)

        return spec, class_id