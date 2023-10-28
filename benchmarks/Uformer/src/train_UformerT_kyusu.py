import os
import sys
import custom_utils
#import image_utils2

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
log_dir = os.path.join(opt.save_dir,'motiondeblur',opt.dataset, opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
res_train_dir_restored = os.path.join(log_dir, 'results/train/restored/')
res_train_dir_target = os.path.join(log_dir, 'results/train/target/')
res_train_dir_input = os.path.join(log_dir, 'results/train/input/')
res_val_dir = os.path.join(log_dir, 'results/validation/')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
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

    # for p in optimizer.param_groups: p['lr'] = lr 
    # warmup = False 
    # new_lr = lr 
    # print('------------------------------------------------------------------------------') 
    # print("==> Resuming Training with learning rate:",new_lr) 
    # print('------------------------------------------------------------------------------') 
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6) 

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
if opt.patch_size > opt.H or opt.patch_size > opt.W:
    print(opt.patch_size,opt.H,opt.patch_size,opt.W )
    raise ValueError('Patch size should be smaller than both H and W of input images.\n \
                    Check opt.patch_size, opt.output_size, and opt.ir_patch.')
len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
######### validation ###########
# with torch.no_grad():
#     model_restoration.eval()
#     psnr_dataset = []
#     psnr_model_init = []
#     for ii, data_val in enumerate((val_loader), 0):
#         target = data_val[0].cuda()
#         input_ = data_val[1].cuda()
#         with torch.cuda.amp.autocast():
#             restored = model_restoration(input_)
#             restored = torch.clamp(restored,0,1)  
#         psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
#         psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
#     psnr_dataset = sum(psnr_dataset)/len_valset
#     psnr_model_init = sum(psnr_model_init)/len_valset
#     print('Input & GT (PSNR) -->%.4f dB'%(psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB'%(psnr_model_init))

psnr_dataset = []
ssim_dataset = []

for ii, data_val in enumerate((val_loader), 0):
    target = data_val[0].cuda()
    input_ = data_val[1].cuda()
    nnn = data_val[2]
    # nnn = data_val[4] # for train
    psnr_dataset.append(utils.batch_PSNR(input_, target, True).item())
    ssim_dataset.append(utils.batch_SSIM(input_, target, True).item())#image_utils2.calc_metric(input_, target, True).item())
psnr_dataset = sum(psnr_dataset)/len_valset
ssim_dataset = sum(ssim_dataset)/len_valset

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
best_ssim = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)//4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    psnr_train_rgb, rmse_train_rgb = [], []
    for i, data in enumerate(tqdm(train_loader), 0): 
        # zero_grad
        optimizer.zero_grad()
        #if i > 1:
        #    break

        target = data[0].cuda()
        input_ = data[1].cuda()
        row, col = data[2], data[3]
       # with torch.cuda.amp.autocast():
            # print('input : ' , input_.shape)
        restored = model_restoration(input_)
        # print('restored : ', restored.shape)
        # print('target : ',target.shape)
        loss_content = criterion_Charbonier(restored, target)
        loss = loss_content# * opt.mse_scale + loss_fft_abs * opt.fft_scale_abs + loss_fft_angle * opt.fft_scale_angle
        #print("loss_content : " + str(loss_content) + "loss_fft_scale_abs : " + str(loss_fft_abs) +  "loss_fft_angle : " + str(loss_fft_angle))
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss +=loss.item()
        psnr_train_rgb.append(utils.batch_PSNR(restored, target, True).item())

    psnr_train_rgb = sum(psnr_train_rgb)/len(psnr_train_rgb) 

 
    with torch.no_grad():
        model_restoration.eval()
        psnr_val_rgb = []
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
           # if(ii > 1): break 
            target, input_, target_fnames, restored_fnames = data_val[0].cuda(), data_val[1].cuda(), data_val[2], data_val[3]
            num_img = target.size()[0]
            num_split_h, num_split_w = int((opt.H-opt.ir_patch*2) / opt.output_size), int((opt.W-opt.ir_patch*2) / opt.output_size)
            #out_img_h, out_img_w = opt.output_size * num_split_h, opt.output_size * num_split_w
         #   print(opt.output_size, num_img, num_split_h, num_split_w, opt.ir_patch)
            #input_small, getback_idx, shuffle_idx = utils.dataset_utils.to_small_batches(opt,target, input_)
            input_small = utils.dataset_utils.to_small_batches(opt, input_)
            #target_small = target_small[getback_idx,:,:,:]
            #target_small = target_small.cpu().detach().numpy()
          #  print(input_small.shape)
            restored_small = model_restoration(input_small)
            # print(restored_small.shape)
            #restored_small = restored_small[getback_idx,:,:]
            # print(restored_small.shape)
           # restored_small = restored_small.cpu().detach().numpy()
           # target = utils.dataset_utils.merge_patches(target_small, dim, opt.output_size, \
                                               # num_imgß, num_split_h, num_split_w,opt)
            #print(restored_fnames)
            restored = utils.dataset_utils.merge_patches(opt, restored_small, input_.shape[0])
            #real_fname = (str(restored_fnames).split('.')[0].split('\'')[1])
            source_img = '/home/n3/kyusu/Uformer_hyungyu/visualized/background.dng'
           # target_tensor_temp, restored_tensor_temp = torch.Tensor(target), torch.Tensor(restored)
            restored = torch.clamp(restored,0,1).cuda()
            
         #   resotred_FFT_ABS = round(utils.batch_FFT(restored_temp, target_tensor_temp, "FFT_abs",True).item(),4)
          #  resotred_FFT_ANGLE = round(utils.batch_FFT(restored_temp, target_tensor_temp, "FFT_angle",True).item(),4)
            resotred_PSNR = round(utils.batch_PSNR(restored, target, True).item(),4)
            resotred_SSIM = round(utils.batch_SSIM(restored, target, True).item(),4)
            # if(ii < 1):
            #     print("saving ..")
            #     source_img = '/home/n3/kyusu/Uformer_hyungyu/visualized/background.dng'
            #     output_name = '/home/n3/kyusu/Uformer_hyungyu/visualized/kyusu/' + opt.test_dir + '/'
            #     if not os.path.exists(output_name):
            #         os.makedirs(output_name)
            #     restored_fnames = restored_fnames[0]
            #     new_output_name = restored_fnames+'_'+str(resotred_PSNR) + '_' + str(resotred_SSIM) +'.png'
            #     custom_utils.NPYtoPNG(source_img,restored,output_name,epoch,new_output_name)
            

            if ii < 1: #opt.save_img_iter:
                for j in range(restored.shape[0]):
                    restored_fnames_j = restored_fnames[j]
                    print(restored.shape[0], restored_fnames_j)
                    output_name = restored_fnames_j + '.png'
                    custom_utils.NPYtoPNG(source_img,restored[j],res_val_dir,epoch,output_name)
              #  custom_utils.save_4ch_npy_to_img(restored, new_output_name,source_img, in_pxl=255.0, max_pxl=1023.0)
                #image_utils2.save_output(opt, source_img,restored, epoch, '/home/n4/hyungyu/Uformer/Uformer/visualized/UDC_validation_visualized_TEST/Test3/', restored_fnames, '.png')
           #  target, restored = torch.Tensor(target), torch.Tensor(restored)ls 

            #else: # Validate with original image
            #    with torch.cuda.amp.autocast():
            #        restored = model_restoration(input_, row, col)
            # target, restored = torch.Tensor(target), torch.Tensor(restored)
            # restored = torch.clamp(restored,0,1)
            psnr_val_rgb.append(utils.batch_PSNR(restored, target, True).item())
            ssim_val_rgb.append(utils.batch_SSIM(restored, target, True).item())#image_utils2.calc_metric(restored, target, True).item())
          #  fft_abs_val_rgb.append(utils.batch_FFT(restored, target,"FFT_abs", True).item())
         #   fft_angle_val_rgb.append(utils.batch_FFT(restored, target, "FFT_angle",True).item())
        #print("sum(psnr_val_rgb) : ",psnr_val_rgb," len(psnr_val_rgb) : ",len(psnr_val_rgb)) 
        psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
        ssim_val_rgb = sum(ssim_val_rgb)/len(ssim_val_rgb)
      #  fft_abs_val_rgb = sum(fft_abs_val_rgb)/len(fft_abs_val_rgb)
      #  fft_angle_val_rgb = sum(fft_angle_val_rgb)/len(fft_angle_val_rgb)
        if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
    #    if fft_abs_val_rgb > best_fft_abs:
       #     best_fft_abs = fft_abs_val_rgb
     #   if fft_angle_val_rgb > best_fft_angle:
           # best_fft_angle = fft_angle_val_rgb
          
        print("[ Ep %d | PSNR-tr=%.4f | PSNR-val=%.4f |  | SSIM-val=%.4f] ----  [ best_Ep %d | Best_PSNR: %.4f | Best_SSIM: %.4f ]" % (epoch,psnr_train_rgb,psnr_val_rgb,ssim_val_rgb,best_epoch,best_psnr,best_ssim)) # best_iter,
        with open(logname,'a') as logF:
            logF.write("[ Ep %d | PSNR-tr=%.4f | PSNR-val=%.4f  |  SSIM-val=%.4f] ----  [ best_Ep %d | Best_PSNR: %.4f | Best_SSIM: %.4f ]" \
                % (epoch,psnr_train_rgb,psnr_val_rgb,ssim_val_rgb,best_epoch,best_psnr,best_ssim) + '\n')# (epoch,psnr_train_rgb,psnr_val_rgb,best_epoch,best_psnr)+'\n') # best_iter,
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
