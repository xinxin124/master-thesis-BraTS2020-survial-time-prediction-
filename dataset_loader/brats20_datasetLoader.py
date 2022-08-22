#!/usr/bin/env python

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
import nibabel as nib
from nilearn import image as nimg
import os
from glob import glob
import csv
from datetime import datetime
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt

from numpy import expand_dims
from sklearn.model_selection import train_test_split


class BraTS20Dataset(Dataset):

    def __init__(self,  root_dir, survival_info, multimod_list = ["flair", "t1", "t1ce", "t2"], transform=None,
                 get_sample_id=False):
        # data loading
        self.root_dir = root_dir
        self.patients_info = survival_info #survial_info.csv
        self.multimod_list = multimod_list
        self.sur_info_path = self.root_dir + self.patients_info
        key_value = np.genfromtxt(self.sur_info_path, dtype="str", delimiter=",", skip_header=1)
        self.sample_id_list= [v  for v, _, _, _ in key_value]  # string type
        self.get_sample_id = get_sample_id
        self.transform = transform


    def __len__(self):
        # len(dataset) get length of the dataset, The size of our dataset
        return len(self.sample_id_list)

    def __getitem__(self, index):
        # dataset[0]
        sample_id = self.sample_id_list[index]
        sample_ar = self.select_multipl_types_nifti(self.root_dir,sample_id, self.multimod_list) #np arra
        #sample_tensor = torch.from_numpy(sample_ar) # in tensor
        data = expand_dims(sample_ar, axis=0) # add channel [1, 620,240,240]

        age, survival_days = self.get_info_of_samples(self.root_dir,self.patients_info)[sample_id]
        survival_days = torch.from_numpy(np.array(survival_days, dtype=np.float32))
        data = torch.from_numpy(data) #[1, 1, 620, 240, 240]
        data_pre = torch.permute(data, (1, 0, 2, 3)) # [620, 1, 240, 240]

        if self.transform:
            data_pre = self.transform(data_pre)

        if self.get_sample_id:
            return data_pre, survival_days, sample_id
        return data_pre, survival_days

    def get_nifti_file_for_filetype(self, root_path, sample_id, file_type):
        file_path = os.listdir(root_path)
        if sample_id in file_path:
            file_dir = os.path.join(root_path, sample_id)
            file_list = os.listdir(file_dir)
            file_type_name = sample_id + "_" + file_type + ".nii.gz"
            if file_type_name in file_list:
                file_path = os.path.join(file_dir, file_type_name)
                file = nib.load(file_path)
                nifti_img = nimg.load_img(file)
                nifti_array = np.array(nifti_img.dataobj).astype(np.double)
                return nifti_array


    def select_multipl_types_nifti(self, root_path, sample_id, file_types):
        multimodel_nifti_stack = []
        for type in file_types:
            nifti_np = self.get_nifti_file_for_filetype(root_path, sample_id, type)
            nifti_np = np.moveaxis(nifti_np, -1, 0)
            multimodel_nifti_stack.append(nifti_np)

        multimodel_nifti_array = np.stack(multimodel_nifti_stack, axis=0).astype(np.float64) 
        new_mul = multimodel_nifti_array.reshape(155*len(self.multimod_list), 240, 240)
        return new_mul

    def get_labels_of_samples(self, root_path, samples_info): 
        train_labels = root_path + samples_info
        key_value = np.genfromtxt(train_labels, dtype="str", delimiter=",", skip_header=1)
        patient_labels = {k: self.hgg_lgg_to_int(v) for v, k in key_value}
        return patient_labels


    def get_info_of_samples(self, root_path, samples_info): 
        train_labels = root_path + samples_info
        key_value = np.genfromtxt(train_labels, dtype="str", delimiter=",", skip_header=1)
        patient_age_and_sur_days = {v: [k, f] for v, k, f, _ in key_value} #string type
        for key in patient_age_and_sur_days.keys():
            patient_age_and_sur_days[key] = [float(x) for x in patient_age_and_sur_days[key]]
        return patient_age_and_sur_days




