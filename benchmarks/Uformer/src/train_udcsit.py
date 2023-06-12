import os
import sys
import custom_utils
import image_utils

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image motion deblurring')).parse_args()
print(opt)

import utils
from dataset.dataset_motiondeblur import *
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler



######### Logs dir ###########
log_dir = os.path.join(opt.save_dir,opt.dataset, opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
res_val_dir = os.path.join(log_dir, 'results/validation/')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(res_val_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ########### 
model_restoration = torch.nn.DataParallel (model_restoration) 
model_restoration.cuda() 
     

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ########### 
if opt.resume: 
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest) 

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')


######### Loss ###########
criterion_Charbonier = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt,opt.train_dir, img_options_train, opt.dataset_name)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=False, drop_last=False)

img_options_val = {'patch_size':opt.val_ps}
val_dataset = get_validation_deblur_data(opt.val_dir, img_options_val, opt.dataset_name)

val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size_val, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)
dim,opt.H, opt.W = train_dataset.return_size()
len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
best_ssim = 0

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    psnr_train_rgb = []
    for i, data in enumerate(tqdm(train_loader), 0): 
        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()
        row, col = data[2], data[3]

        restored = model_restoration(input_)

        loss_content = criterion_Charbonier(restored, target)
        loss = loss_content 

        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss +=loss.item()
        psnr_train_rgb.append(utils.batch_PSNR(restored, target, True).item())

    psnr_train_rgb = sum(psnr_train_rgb)/len(psnr_train_rgb) 

 
    with torch.no_grad():
        model_restoration.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):

            target, input_, target_fnames, restored_fnames = data_val[0].cuda(), data_val[1].cuda(), data_val[2], data_val[3]
            num_img = target.size()[0]

            input_small = utils.dataset_utils.to_small_batches(input_)
            restored_small = model_restoration(input_small)
            restored = utils.dataset_utils.merge_patches(opt, restored_small, input_.shape[0])
            restored = torch.clamp(restored,0,1).cuda()
            
            source_img = '../../background.dng'
            if ii < opt.save_img_iter:
                for j in range(restored.shape[0]):
                    restored_fnames_j = restored_fnames[j]
                    print(restored.shape[0], restored_fnames_j)
                    output_name = restored_fnames_j + '.png'
                    custom_utils.NPYtoPNG(source_img,restored[j],res_val_dir,epoch,output_name)

            psnr_val_rgb.append(utils.batch_PSNR(restored, target, True).item())
            ssim_val_rgb.append(utils.batch_SSIM(restored, target, True).item())

        psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
        ssim_val_rgb = sum(ssim_val_rgb)/len(ssim_val_rgb)
        if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
          
        print("[ Ep %d | PSNR-tr=%.4f | PSNR-val=%.4f |  | SSIM-val=%.4f] ----  [ best_Ep %d | Best_PSNR: %.4f | Best_SSIM: %.4f ]" % (epoch,psnr_train_rgb,psnr_val_rgb,ssim_val_rgb,best_epoch,best_psnr,best_ssim)) 
        with open(logname,'a') as logF:
            logF.write("[ Ep %d | PSNR-tr=%.4f | PSNR-val=%.4f  |  SSIM-val=%.4f] ----  [ best_Ep %d | Best_PSNR: %.4f | Best_SSIM: %.4f ]" \
                % (epoch,psnr_train_rgb,psnr_val_rgb,ssim_val_rgb,best_epoch,best_psnr,best_ssim) + '\n')
        model_restoration.train()
        torch.cuda.empty_cache()
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())
