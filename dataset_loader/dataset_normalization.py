
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
import nibabel as nib
from nilearn import image as nimg
import os
from glob import glob
import csv

from tqdm import tqdm
import matplotlib.pyplot as plt



def calculate_mean_and_std_of_samples(dataloader, type="flair"):
    # place holders
    psum = torch.tensor([0.0])
    psum_sq = torch.tensor([0.0])
    file_type = type

    for inputs, _ in tqdm(dataloader):
        if file_type == "flair":
            data = inputs[:, :150, :240, :240]
        elif file_type == "t1":
            data = inputs[:, 155:310, :240, :240]
        elif file_type == "t1ce":
            data = inputs[:, 310:465, :240, :240]
        elif file_type == "t2":
            data  = inputs[:, 465:620, :240, :240]

        psum  += data.sum()
        psum_sq += (data ** 2).sum()

    #pixel count
    count = 235 * 620 * 240 * 240

    #mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    return total_mean, total_std