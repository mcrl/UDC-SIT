import torch
import os
import torch.nn as nn
import random
import torchvision
import numpy as np
### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor

### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

def to_small_batches(input_):

    num_img = input_.shape[0]

    num_patches_H, num_patches_W = 2, 1
    inputs = None

    for i in range(num_img):
        input_batch = input_[i,:,:,:]
        for j in range(num_patches_H):
            for k in range(num_patches_W):
                if j == 0:
                    input_piece = input_batch[:,0:1280,0:1280]
                else:
                    input_piece = input_batch[:,512:1792,0:1280]

                if i == 0 and j == 0 and k == 0:
                    inputs = input_piece.unsqueeze(dim=0)
                else:
                    inputs = torch.cat((inputs, input_piece.unsqueeze(dim=0)), dim=0)

    return inputs


def merge_patches(opt, patches, num_img):

    num_patches_H, num_patches_W = 2, 1
    
    image = torch.zeros((num_img, opt.dd_in, 1792, 1280), dtype=torch.float32)

    for i in range(num_img):
        for j in range(num_patches_H):
            for k in range(num_patches_W):
                if j == 0:
                    image[i,:,0:896,:] = patches[0,:,0:896,:]
                else:
                    image[i,:,896:1792,:] = patches[1,:,384:1280,:]
   
    center_crop = torchvision.transforms.CenterCrop((opt.H, opt.W))
    image = center_crop(image)

    return image
