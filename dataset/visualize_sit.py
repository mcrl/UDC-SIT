import os
import glob
import numpy as np
import rawpy as rp
from PIL import Image
import imageio
import torch

dir_sit= './training/' # Modify training as validation or test if you need.
subdir = ['GT', 'input']
res_dir = './visualize_sit/' 


def load_npy(filepath):
    img = np.load(filepath)
    img = img/1023
        
    return img


def save_4ch_npy_png(tensor_to_save, res_dir, fname, save_type):
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,1)
    source_dir = './background.dng'
    data  = rp.imread(source_dir)
    if not os.path.exists(os.path.join(res_dir,save_type)):
        os.makedirs(os.path.join(res_dir,save_type))        
    npy = fn_tonumpy(tensor_to_save)  
    npy = (npy.squeeze()*1023)
    GR = data.raw_image[0::2,0::2]
    R = data.raw_image[0::2,1::2]
    B = data.raw_image[1::2,0::2]
    GB = data.raw_image[1::2,1::2]
    GB[:,:] = 0
    B[:,:] = 0
    R[:,:] = 0
    GR[:,:] = 0
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
    newData = data.postprocess()
    start = (0,464)
    end =  (3584,3024)
    newData = newData[start[0]:end[0], start[1]:end[1]]    
    fname = fname.split('/')[7].split('.')[0]
    if save_type is None:
        output_name = res_dir + save_type + '/' + fname + '.png'
    if save_type is not None:
        output_name = res_dir + sub + '/' + fname + '_' + save_type + '.png'
    imageio.imsave(output_name, newData)


for sub in subdir:
    load_dir = dir_sit + sub + '/'
    flist = glob.glob(load_dir + '*.npy')
    i = 0
    for fname in flist:
        i += 1
        print("Saving images......{}/{}".format(i, 2000))
        npy_to_save = torch.from_numpy(np.float32(load_npy(fname))).permute(2,0,1)
        npy_to_save = torch.clamp(npy_to_save,0,1)
        save_4ch_npy_png(npy_to_save, res_dir, fname, sub)

