import tianmoucv
from tianmoucv.isp import SD2XY
from tianmoucv.data import TianmoucDataReader

import numpy as np
import torch
import os
import cv2
import tqdm

# 用于处理差分数据的函数
def process_diff(numpy_diff, TH=0):
    if isinstance(numpy_diff, torch.Tensor):
        numpy_diff = numpy_diff.numpy()
    high_contrast = 1
    if not high_contrast:
        TH_1 = TH
        numpy_diff_255 = numpy_diff * 255
    else:
        TH_1 = TH * 4
        numpy_diff_255 = numpy_diff * 800
        numpy_diff_255 = np.where(numpy_diff > 254,  254,  numpy_diff_255)
        numpy_diff_255 = np.where(numpy_diff < -254, -254, numpy_diff_255)
        
    positive_mask  = numpy_diff_255 > TH_1
    negative_mask  = numpy_diff_255 < -TH_1
    plt_image      = np.zeros((numpy_diff_255.shape[0], numpy_diff_255.shape[1], 3))
    
    green_channel = numpy_diff_255[positive_mask]
    zero_channel  = np.zeros_like(numpy_diff_255[positive_mask])
    plt_image[positive_mask] = np.dstack((zero_channel, green_channel, zero_channel))

    red_channel   = -numpy_diff_255[negative_mask]
    zero_channel  = np.zeros_like(numpy_diff_255[negative_mask])
    plt_image[negative_mask] = np.dstack((red_channel, zero_channel, zero_channel))
    
    plt_image = 255 - plt_image
    plt_image = plt_image / 255
    return plt_image 

dataPath = ['/media/booster/a80040b9-afda-472d-b2e9-3f6e46d7f85b/ZKC_exp/9']
key_list = ['cam']

for key in key_list:
    dataset = TianmoucDataReader(dataPath, N=1, matchkey=key, camera_idx=0)
    save_dir_key = "raw_data_0"
    sub_directory = os.path.join(dataPath[0], key)
    save_dir = os.path.join(sub_directory, save_dir_key)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir + ' has created')

    for index_COP in tqdm.tqdm(range(len(dataset))):
        if 0 <= index_COP < len(dataset):
            sample = dataset[index_COP]
            rgb = sample['F0_without_isp']
            rgb = rgb[:, ::-1, :]
            rgb_out1 = np.zeros_like(rgb)
            np.copyto(rgb_out1, rgb)
            COP_name = "COP_{}.bmp".format(index_COP)
            COP_path = os.path.join(save_dir, COP_name)
            cv2.imwrite(COP_path, rgb_out1 * 255) 

            rawdiff = sample['rawDiff'] / 255.0
            tsdiff = torch.Tensor(sample['tsdiff'])

            index_AOP_list = range(25)
            for index_AOP in index_AOP_list:
                TD = tsdiff[0, index_AOP, ...].numpy()
                TD = np.flip(TD, axis=1)
                TD_name = 'COP_{}_TD_{}.npy'.format(index_COP, index_AOP)
                TD_path = os.path.join(save_dir, TD_name)
                np.save(TD_path, TD)

                # SD = rawdiff[1:, index_AOP, ...].permute(1, 2, 0)
                # SDx, SDy = SD2XY(SD)

                # SD1 = np.repeat(np.repeat(SDx.numpy(), 2, axis=0), 2, axis=1)
                # SD2 = np.repeat(np.repeat(SDy.numpy(), 2, axis=0), 2, axis=1)
                # SD1 = np.flip(SD1, axis=1)
                # SD2 = np.flip(SD2, axis=1)
                # SD1_name = 'COP_{}_SD1_{}.npy'.format(index_COP, index_AOP)
                # SD2_name = 'COP_{}_SD2_{}.npy'.format(index_COP, index_AOP)
                # SD1_path = os.path.join(save_dir, SD1_name)
                # SD2_path = os.path.join(save_dir, SD2_name)
                # np.save(SD1_path, SD1)
                # np.save(SD2_path, SD2)
