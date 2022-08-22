#!/usr/bin/env python 

import os 
import numpy as np 
from PIL import Image 
from matplotlib import pyplot as plt
import nibabel as nib
from nilearn import image as nimg
import cv2
import PIL
from PIL import Image

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import pandas as pd

#count pixel of different labels of brain tumor area 
def get_segmentation_area(seg_path):
    seg_np = np.array(nimg.load_img(seg_path).dataobj)
    count = 0
    count2 = 0
    count4 = 0
    for i in seg_np.flatten():
        if i == 1:
            count += 1
        if i == 2:
            count2 += 1
        if i == 4:
            count4 += 1
    # print("label 1- NCR/NET : ", count)
    # print("label 2- ED: ", count2)
    # print("label 4- ET: ", count4)
    return count, count2, count4

#make brain tumor area to be binary
def binary_segmentaion(seg_path):
    seg_np = np.array(nimg.load_img(seg_path).dataobj)
    seg_cp = np.full((seg_np.shape[0], seg_np.shape[1],seg_np.shape[2] ), 0)
    for x in range(seg_np.shape[0]):
        for y in range(seg_np.shape[1]):
            for i in range(seg_np.shape[2]):
                    j = seg_np[x,y,i]
                    if j == 1 or j == 2 or j == 4 :
                        seg_cp[x,y,i] = 1
    return seg_cp

#counting  percent of different labels of tumor area
def count_seg_area_of_slices(index, seg_path):
    seg_np = np.array(nimg.load_img(seg_path).dataobj)
    count = 0
    count2 = 0
    count4 = 0
    for i in index:
        seg_1s = seg_np[:, :, i]
        for j in seg_1s.flatten():
            if j == 1:
                count += 1
            if j == 2:
                count2 += 1
            if j == 4:
                count4 += 1
    label_1, label_2, label_4 = get_segmentation_area(seg_path)
    total = label_1 + label_2 + label_4
    pre_label_1 = count / total
    pre_label_2 = count2 / total
    pre_label_4 = count4 / total
    return round(pre_label_1,2), round(pre_label_2,2), round(pre_label_4,2)

#heatmaps for segmentation ground truth
def segmentation_weights(seg_path, plot=False):
    seg_np = np.array(nimg.load_img(seg_path).dataobj)
    slices = seg_np.shape[2]
    weights_list = [ ]
    for i in range(slices):
        sum = 0
        for x in seg_np[:,:,i].flatten():
            if x == 1 or x == 2 or x == 4:
                sum += 1
        weights_list.append(sum)
    weights_per = np.array(weights_list) / np.sum(np.array(weights_list))
    weights = np.reshape(weights_per, (155,1))
    if plot:
        ax1 = sns.heatmap(weights, annot=False)
        plt.title("weights of segmentation")
        plt.savefig("seg_weights.png")
        plt.show()
    save_weights_list = list(weights_per)
    return save_weights_list

#comparing the original slices with brain tumor overlayed slices 
def plot_results(index, nifti_path, seg_path, save_file):
    flair_np = np.array(nimg.load_img(nifti_path).dataobj)
    flair_1s = flair_np[:, :, index].astype(np.uint8)

    seg_np = np.array(nimg.load_img(seg_path).dataobj)
    seg_1s = seg_np[:, :, index].astype(np.uint8)

    seg_binary = binary_segmentaion(seg_path)
    seg_bi = seg_binary[:, :, index].astype(np.uint8)
    ret, seg_gray = cv2.threshold(seg_1s, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=seg_gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # copy original slice
    overlay_img = flair_1s.copy()
    cv2.drawContours(image=overlay_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(flair_1s, 'gray')
    plt.title("original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(overlay_img, ) # cmap= 'plasma'
    plt.title("segmentation overlay")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(seg_bi, 'gray', interpolation='nearest')
    plt.title("segmentation")
    plt.axis("off")
    plt.savefig(save_file)
    plt.show()

#generate heatmaps with attention-weights 
def get_heatmap(weights_file, save_name, file_type, normalization = False, figsize=(10,10)):
    weights_f = pd.read_csv(weights_file)
    wf_2 = weights_f.apply(lambda x: x / x.max())
    plt.figure(figsize=figsize)
    if normalization == True:
        sns.heatmap(wf_2, cmap='viridis')
    else:
        sns.heatmap(weights_f, cmap='viridis')
    plt.ylabel('slice index')
    plt.title(file_type)
    plt.savefig(save_name)
    plt.show()

#select the slices with attentioin-weights over threshold
def get_index_and_weights_of_slices(csv_file, sample_id):
    weights_f = pd.read_csv(csv_file)
    df = weights_f.reset_index(level=0)
    df_id_idx = df[['index', sample_id]]
    threshold = weights_f[sample_id].mean()  #average value of weights
    df_threshold = df_id_idx.loc[df_id_idx[sample_id] > threshold]
    index = list(df_threshold['index'])
    weights = list(df_threshold[sample_id])
    print(index)
    print(weights)
    return {'index': index, 'weights': weights }

#get the ground truth of segmentation of brain tumor area
def get_segmentation_ground_truth(weights_file, root_path, save_file_name, save_heatmap_name):
    df = pd.read_csv(weights_file)
    sample_ids = df.columns
    new_df = pd.DataFrame()
    for sample_id in sample_ids:
        file_dir = os.path.join(root_path, sample_id)
        seg_sample_id = sample_id + "_seg.nii.gz"
        file_path = os.path.join(file_dir, seg_sample_id)
        seg_weights_list = segmentation_weights(file_path)
        new_df[sample_id] = seg_weights_list
    new_df.to_csv(save_file_name, index=False)
    get_heatmap(save_file_name, save_name=save_heatmap_name,
                file_type='segmentation ground truth', normalization = True)


#get pearson correlation coefficient 
def pearson_cor_coe_os(os_file, survival_file, save_pc_os):
    f_os = pd.read_csv(os_file)
    f_sur = pd.read_csv(survival_file)
    df = pd.merge(f_os, f_sur, on='Brats20ID')
    correlation = df.corr()
    os_cor = correlation.loc['Survival_days', 'Predicted_survival_time']
    sns.lmplot(x="Survival_days", y="Predicted_survival_time", data=df)
    #plt.title('pearson correlation coefficient')
    plt.savefig(save_pc_os+'.png')
    plt.show()

    #accuracy
    df.to_csv(save_pc_os+".csv", index=False)

    return os_cor

def get_file(root_path,sample_id, file_type='flair'):
    file_dir = os.path.join(root_path, sample_id)
    seg_sample_id = sample_id + "_"+ file_type + ".nii.gz"
    file_path = os.path.join(file_dir, seg_sample_id)
    return file_path

#accuracy calculation
def os_accuracy(os_file, survival_file):
    f_os = pd.read_csv(os_file)
    f_sur = pd.read_csv(survival_file)
    df = pd.merge(f_os, f_sur, on='Brats20ID')
    targets = list(df['Survival_days'])
    preds = list(df['Predicted_survival_time'])
    if len(targets) == len(preds):
        ls, ms, ss = 0, 0, 0
        for i in range(len(targets)):
          if preds[i] >= 450 and targets[i] > 450:
              ls += 1
          elif preds[i] <=300 and targets[i] <=300:
              ss += 1
          elif preds[i] > 300 and preds[i] < 450 and targets[i] > 300 and targets[i] < 450:
              ms += 1
        num_right = ls + ss + ms
        accuracy = round(num_right/len(targets), 3)
    return accuracy


#plot residuals 
def plot_residual(os_file, survival_file):
  f_os = pd.read_csv(os_file)
  f_sur = pd.read_csv(survival_file)
  df = pd.merge(f_os, f_sur, on='Brats20ID')
  data = df 
  sns.residplot( y ='Predicted_survival_time', x ='Survival_days', data=data)
  plt.ylabel('Residual')
  plt.savefig('residual_t1ce_r18milpe.png')
  plt.show()



def get_segmentation_overlay_comparing(weights_file, sample_id, file_type="flair"):
    root_path = '/dhc/home/huang.xin/BraTS2020/train_20/'
    nifti_path = get_file(root_path, sample_id, file_type=file_type)
    seg_path = get_file(root_path, sample_id, file_type="seg")
    idx_and_weights_dic = get_index_and_weights_of_slices(weights_file, sample_id)
    indexs = idx_and_weights_dic['index']
    save_file = sample_id + '.png'
    new_dir_path = os.path.join('/dhc/home/huang.xin/my_project/mulMod3D/w_heatmap/png_heatmap/', sample_id)
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    for index, weight in zip(indexs, idx_and_weights_dic['weights']):
        for i in range(len(idx_and_weights_dic)):
            file_name = str(index) + '_' + save_file
            save_name = os.path.join(new_dir_path, file_name)
            plot_results(index, nifti_path, seg_path, save_name)

    
    









