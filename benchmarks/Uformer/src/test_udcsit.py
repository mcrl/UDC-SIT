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

path_chk_rest = os.path.join(model_dir, 'model_best.pth') #opt.pretrain_weights 
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

with torch.no_grad():
    model_restoration.eval()
    psnr_test = []
    ssim_test = []
    for i, data_test in enumerate((test_loader), 1):
        target, input_, target_fnames, restored_fnames = data_test[0].cuda(), data_test[1].cuda(), data_test[2], data_test[3]
        num_img = target.size()[0]

        input_small = utils.dataset_utils.to_small_batches(input_)
        restored_small = model_restoration(input_small)

        restored = utils.dataset_utils.merge_patches(opt, restored_small, input_.shape[0])
        
        target_tensor_temp, restored_tensor_temp = torch.Tensor(target), torch.Tensor(restored)
        restored = torch.clamp(restored,0,1).cuda()
        restored_PSNR = round(utils.batch_PSNR(restored, target, True).item(),4)
        restored_SSIM = round(utils.batch_SSIM(restored, target, True).item(),4)

        source_img = '../../background.dng'
        
        for j in range(restored.shape[0]):
            restored_fnames_j = restored_fnames[j]
            output_name = restored_fnames_j + '.png'
            #custom_utils.NPYtoPNG(source_img,restored[j],res_test_dir,'test',output_name)
        
        restored = torch.clamp(restored,0,1)     
        psnr_test.append(utils.batch_PSNR(restored, target, True).item())
        ssim_test.append(utils.batch_SSIM(restored, target, True).item())

        psnr_iter = sum(psnr_test)/len(psnr_test)
        ssim_iter = sum(ssim_test)/len(ssim_test)
        
        print("Iter=%s | [PSNR=%.4f, SSIM=%.4f]" % (str(i), psnr_iter,ssim_iter))
        
    psnr_test = sum(psnr_test)/len(psnr_test)
    ssim_test = sum(ssim_test)/len(ssim_test)

    print("Inference finished.")
    print("[PSNR-test=%.4f, SSIM-test=%.4f]" % (psnr_test,ssim_test))

    with open(logname_test,'a') as logF:
        logF.write("PSNR-test=%.4f, PSNR-test=%.4f]" \
                     % (psnr_test,ssim_test)+'\n')
    torch.cuda.empty_cache()

print("Now time is : ",datetime.datetime.now().isoformat())
