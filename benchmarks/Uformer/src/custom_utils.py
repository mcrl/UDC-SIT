import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as im
import random
import imageio
import rawpy as rp
import cv2

def NPYtoPNG_feng(npy,output_name):
    print(output_name)

    img = ((npy.squeeze()*255).astype(np.float32)).transpose(1,2,0)
    print(img.shape)
    img = cv2.cvtColor( img,cv2.COLOR_BGR2RGB) #for feng

    cv2.imwrite(output_name, img)

def save_4ch_npy_to_img(
    img_npy, img_path, dng_info, in_pxl=255.0, max_pxl=1023.0
):
    if dng_info is None:
        raise RuntimeError(
            "DNG information for saving 4 channeled npy file not provided"
        )
    # npy file in hwc manner to cwh mannerirestoredirestored
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
    img_npy = fn_tonumpy(img_npy)
    data = rp.imread(dng_info)
    npy = img_npy.transpose(0,1,3,2)/ in_pxl * max_pxl

    GR = data.raw_image[0::2, 0::2]
    R = data.raw_image[0::2, 1::2]
    B = data.raw_image[1::2, 0::2]
    GB = data.raw_image[1::2, 1::2]
    GB[:, :] = 0
    B[:, :] = 0
    R[:, :] = 0
    GR[:, :] = 0

    h = npy.shape[3:]
    w = npy.shape[2:]

    GR[:w, :h] = npy[0][:w][:h]
    R[:w, :h] = npy[1][:w][:h]
    B[:w, :h] = npy[2][:w][:h]
    GB[:w, :h] = npy[3][:w][:h]
    newData = data.postprocess()
    start = (0, 464)  # (448 , 0) # 1792 1280 ->   3584, 2560
    end = (3584, 3024)
    output = newData[start[0] : end[0], start[1] : end[1]]
    
    imageio.imsave(img_path, output)

def NPYtoPNG(source_img_path,npy,output_name,epoch,fnames):
   
    str_epoch = "Epoch_" + str(epoch)

    if not os.path.exists(os.path.join(output_name,str_epoch)):
        os.makedirs(os.path.join(output_name,str_epoch))
    
    output_name = output_name + str_epoch + '/' + fnames
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
 
    data  = rp.imread(source_img_path)
    npy = fn_tonumpy(npy)  
    npy = (np.squeeze(npy)*1023).astype(np.float32)

    GR = data.raw_image[0::2,0::2]
    R = data.raw_image[0::2,1::2]
    B = data.raw_image[1::2,0::2]
    GB = data.raw_image[1::2,1::2]
    GB[:,:] = 0
    B[:,:] = 0
    R[:,:] = 0
    GR[:,:] = 0

    npy = npy.transpose(0,2,1)
    
    for k in range(4):
        for i in range(npy.shape[1]):
            for j in range(npy.shape[2]):
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
    start = (0,464)# (448 , 0) # 1792 1280 ->   3584, 2560
    end =  (3584,3024)
    output = newData[start[0]:end[0], start[1]:end[1]]
    imageio.imsave(output_name, output)

def save_res_img2(npy_to_save, epoch, res_dir, fnames):
    array = np.reshape((npy_to_save), (800, 800))
    str_epoch = "Epoch_" + str(epoch)

    if not os.path.exists(os.path.join(res_dir,str_epoch)):
        os.makedirs(os.path.join(res_dir,str_epoch))        
    for i in range(npy_to_save.shape[0]):
        data = im.fromarray(array)
      
    data.save

    
def save_res_img(tensor_to_save, epoch, res_dir, fnames):
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    str_epoch = "Epoch_" + str(epoch)
    
    if not os.path.exists(os.path.join(res_dir,str_epoch)):
        os.makedirs(os.path.join(res_dir,str_epoch))        
    for i in range(tensor_to_save.shape[0]):
        npy = fn_tonumpy(tensor_to_save)

        img = (npy.squeeze()*255)#.astype(np.float32)

        fname = fnames[i].split('.')[0]

        imwrite(res_dir + str_epoch +'/' + fname + '.png',img)


