import os
import sys
import custom_utils

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
print(opt)

import utils

from dataset.dataset_motiondeblur import *
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader
import glob
import random
import time
import numpy as np
import datetime
from pdb import set_trace as stx

import dataset
from utils.dataset_utils import to_small_batches, merge_patches
import image_utils

best_psnr, best_epoch, best_iter = 0, 0, 0

######### Logs dir ###########
log_dir = os.path.join(opt.save_dir, opt.dataset, opt.arch+opt.env)
logname_test = os.path.join(log_dir, datetime.datetime.now().isoformat() + '_test'+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
res_test_dir = os.path.join(log_dir, 'results/test/')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(res_test_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname_test,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### DataParallel ########### 
model_restoration = torch.nn.DataParallel (model_restoration) 
model_restoration.cuda()

path_chk_rest = os.path.join(model_dir, 'model_best.pth')
print("Loading chekpoint......", path_chk_rest)
utils.load_checkpoint(model_restoration,path_chk_rest) 

######### DataLoader ###########
print('===> Loading datasets')
img_options_test = {'patch_size':opt.val_ps}
test_dataset = get_validation_deblur_data(opt.test_dir,img_options_test, opt.dataset_name)
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size_val, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)
dim, opt.H, opt.W = test_dataset.return_size()
len_testset = test_dataset.__len__()
print("sizeof test set: ", len_testset)

torch.cuda.empty_cache()
#### Evaluation ####
psnr_test_iter, ssim_test_iter, psnr_test_rgb, ssim_test_rgb = [], [], [], []
with torch.no_grad():
    model_restoration.eval()
    for ii, data_test in enumerate((test_loader), 1):
        target, input_, target_fnames, restored_fnames = data_test[0], data_test[1].cuda(), data_test[2], data_test[3]
        num_img = target.size()[0]

        input_small = to_small_batches(input_)
        restored_small = model_restoration(input_small).cpu().detach()

        restored = merge_patches(opt, restored_small, input_.shape[0])
        restored = torch.clamp(restored,0,1)

        psnr_test_rgb = image_utils.each_PSNR(restored, target)
        ssim_test_rgb = image_utils.calc_metric(restored, target, size_average=False)
        psnr_test_iter.append(sum(psnr_test_rgb)/len(psnr_test_rgb))
        ssim_test_iter.append(sum(ssim_test_rgb)/len(ssim_test_rgb))
        
        source_img = '../../background.dng'
        for j in range(restored.shape[0]):
            fname = target_fnames[j] + ".png"
            custom_utils.NPYtoPNG(source_img,restored[j],res_test_dir,'test',fname)        

        for i in range(restored.shape[0]):
            restored_fname = restored_fnames[i]
            print("[ TEST %d/%d | %s] PSNR-test=%.4f, SSIM-test=%.4f, PSNR-avg=%.4f, SSIM-avg=%.4f" 
            % ((ii-1)*opt.batch_size_val+i+1, len_testset, restored_fname, 
            psnr_test_rgb[i], ssim_test_rgb[i], sum(psnr_test_iter)/len(psnr_test_iter), 
            sum(ssim_test_iter)/len(ssim_test_iter))) # avg PSNR of this iteration

        with open(logname_test,'a') as logF:
            for i in range(restored.shape[0]):
                restored_fname = restored_fnames[i].split('.')[0]
                logF.write("%s, %.4f, %.4f" % (restored_fname, psnr_test_rgb[i], ssim_test_rgb[i])+'\n')
    torch.cuda.empty_cache()

print("AVG of TEST PSNR: ", sum(psnr_test_iter)/len(psnr_test_iter))
print("AVG of TEST SSIM: ", sum(ssim_test_iter)/len(ssim_test_iter))
print("Total images tested: ", len_testset)

print("Now time is : ",datetime.datetime.now().isoformat())
