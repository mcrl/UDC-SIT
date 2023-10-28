import os
import sys
import natsort
import glob
# from PIL import Image
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import rawpy as rp

# import torch.nn as nn
# import torch.nn.functional as F

'''
Pre-processing for taken images of input and groundtruth.
This function first crop the groundtruth images with the size (target_h, target_w).
Then it shifts input images so that the loss between input and groundtruth becomes minimum.
User should set the parameters below:
  - target_h: the height of post-processed image
  - target_w: the width of post-processed image
  - max_shift: how much pixels will be shifted and compare the loss between img_input_crop and img_gt_crop
After the alignment using this script, the alignment discrepancies are performed through crowdsourcing.
Finally, we measure the Percentage of Correct Keypoints (PCK)
'''

LAMBDA_1, LAMBDA_2, LAMBDA_3 = 1, 1, 1

def load_npy(dataset_list):
    dataset_list = natsort.natsorted(dataset_list)
    img_list = [None] * len(dataset_list)
    fname_list = [None] * len(dataset_list)
    for i in range(len(dataset_list)):
        fname_list[i] = dataset_list[i].split("/")[-1]
        img_list[i] = np.load(dataset_list[i])

    return img_list, fname_list


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def FFT_abs_L1(_input, gt):
    criterion = torch.nn.L1Loss()
    input_fft = abs( torch.fft.fft2(_input) )
    gt_fft = abs( torch.fft.fft2(gt) )
    loss_fft = criterion(input_fft, gt_fft)
    return loss_fft


def FFT_angle_L1(_input, gt):
    criterion = torch.nn.L1Loss()
    input_fft = torch.angle( torch.fft.fft2(_input))  
    gt_fft = torch.angle( torch.fft.fft2(gt) ) 
    loss_fft = criterion(input_fft, gt_fft)
    return loss_fft


def NPYtoPNG(res_dir, source_img_path, list_in, target_name):
    input_name = str(target_name)
    npy = list_in
    data  = rp.imread(source_img_path)
    
    GR = data.raw_image[0::2,0::2]
    R = data.raw_image[0::2,1::2]
    B = data.raw_image[1::2,0::2]
    GB = data.raw_image[1::2,1::2]
    GB[:,:] = 0
    B[:,:] = 0
    R[:,:] = 0
    GR[:,:] = 0
    for k in range(4):
        for i in range(npy.shape[1]): # 1280 
            for j in range(npy.shape[2]): # 1792
                if(k==0):
                    GR[i][j] = npy[k][i][j] 
                elif(k==1):
                        R[i][j] = npy[k][i][j]
                elif(k==2):
                    B[i][j] = npy[k][i][j]
                elif(k==3):    
                    GB[i][j] = npy[k][i][j]
    data.raw_image
    newData = data.postprocess()
    start = (0,464)
    end = (3584,3024)
    output = newData[start[0]:end[0], start[1]:end[1]]
    output = cv2.cvtColor( output,cv2.COLOR_RGB2BGR)
    
    output_name = res_dir + input_name + '.png'
    cv2.imwrite(output_name, output)


def pre_processing(list_input, list_gt, fname, res_dir, target_h, target_w, max_shift):
    
    for i in range(len(list_input)):
        img_input = list_input[i] / 1023.0    
        img_gt = list_gt[i] / 1023.0
        
        img_input = torch.from_numpy(img_input.astype(np.float32)).cuda()
        img_gt = torch.from_numpy(img_gt.astype(np.float32)).cuda()

        if LAMBDA_1 != 0:
            mse_ori = float(torch.nn.functional.mse_loss(img_gt, img_input))        
        if LAMBDA_2 != 0:
            fft_abs_ori = FFT_abs_L1(img_input, img_gt)
        if LAMBDA_3 != 0:
            fft_angle_ori = FFT_angle_L1(img_input, img_gt)
        loss_total_ori = mse_ori * LAMBDA_1 + fft_abs_ori * LAMBDA_2 + fft_angle_ori * LAMBDA_3
        
        orig_h, orig_w = img_gt.shape[1], img_gt.shape[2]
        st_h, st_w = int( (orig_h-target_h) / 2), int( (orig_w-target_w) / 2) 

        if(st_h < max_shift or st_w < max_shift):
            print("This script is to crop and align from larger than 1792 x 1280 x 4 (e.g., 2016 x 1512 x 4).")
            print("Check your image size, target_h, target_w, and max_shift.")
            exit(0)
    
        img_gt_crop = img_gt[:, st_h:st_h+target_h, st_w:st_w+target_w]
        img_gt_crop = img_gt_crop.cuda()
        loss_min, loss_total = 100000000, -1
        for j in range(max_shift*2):
            for k in range(max_shift*2):
                img_input_crop = img_input[:, st_h-max_shift+j : st_h-max_shift+j+target_h, st_w-max_shift+k : st_w-max_shift+k+target_w]
                img_input_crop = img_input_crop.cuda()
                fft_abs, fft_angle = 0, 0
                if LAMBDA_1 != 0:
                    mse = torch.nn.functional.mse_loss(img_gt_crop, img_input_crop)
                if LAMBDA_2 != 0:
                    fft_abs = FFT_abs_L1(img_input_crop, img_gt_crop)
                if LAMBDA_3 != 0:
                    fft_angle = FFT_angle_L1(img_input_crop, img_gt_crop)
                
                loss_total = mse * LAMBDA_1 + fft_abs * LAMBDA_2 + fft_angle * LAMBDA_3
    
                if loss_total < loss_min:
                    loss_min = loss_total
                    offset_h = j
                    offset_w = k
                    img_input_res = img_input_crop   

        print(f' Loss of %s: %0.4f ===> %0.4f, where offsets for h and w are  %d, %d, respectively.' 
           % (fname[i], loss_total_ori, loss_min, offset_h-max_shift, offset_w-max_shift) )

        img_input_save = (img_input_res.cpu().numpy().transpose(0,2,1).squeeze()*1023) 
        img_gt_save = (img_gt_crop.cpu().numpy().transpose(0,2,1).squeeze()*1023) 
        
        NPYtoPNG(res_dir[0], '../background.dng', img_input_save, fname[i].split('.')[0]) # input
        NPYtoPNG(res_dir[1], '../background.dng', img_gt_save, fname[i].split('.')[0]) # GT
        img_input_save = img_input_save.transpose(2,1,0)
        img_gt_save = img_gt_save.transpose(2,1,0)
        np.save(res_dir[0] + fname[i], img_input_save) # x_save.npy
        np.save(res_dir[1] + fname[i], img_gt_save) # x_save.npy
    
    print("\nDone.")


if __name__ == "__main__":

    target_h = int(sys.argv[1])
    target_w = int(sys.argv[2])
    max_shift = int(sys.argv[3]) # It is generally smaller than 25. If not, be more carefule when capturing images.

    dir_source = ["/data/s0/udc/dataset/UDC_SIT/npy/validation/input/","/data/s0/udc/dataset/UDC_SIT/npy/validation/GT/"]
    res_dir = ["./input_aligned/", "./GT_aligned/"]

    mkdir(res_dir[0])
    mkdir(res_dir[1])
    
    list_input = glob.glob(dir_source[0] + "*.npy")
    list_gt = glob.glob(dir_source[1] + "*.npy")
    print("input size = ", len(list_input), "gt_size = ",len(list_gt))
  
    list_input, fname_input = load_npy(list_input)
    list_gt, fname_gt = load_npy(list_gt)

    print('-----'*10)
    print('Target H = %d' % target_h)
    print('Target W = %d' % target_w)
    print('Max shift = %d' % max_shift)
    print('-----'*10)
    print("\nBegin pre-processing for UDC images manually taken.\n")
    
    pre_processing(list_input, list_gt, fname_gt, res_dir, target_h, target_w, max_shift)
    
