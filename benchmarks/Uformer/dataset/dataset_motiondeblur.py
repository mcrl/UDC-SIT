import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, opt, rgb_dir,img_options=None, dataset_name=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'GT' 
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files ]
        
        self.img_options=img_options
        self.dataset_name = dataset_name

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size
        
    def return_size(self):
        # if self.data_format != 0:
        #     clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[0])))
        # else:
        clean = torch.from_numpy(np.float32(np.load(self.clean_filenames[0])))
        
        clean = clean.permute(2,0,1) # to CHW
        return clean.shape[0], clean.shape[1], clean.shape[2]

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(load_img(self.clean_filenames[tar_index],self.dataset_name))
        noisy = torch.from_numpy(load_img(self.noisy_filenames[tar_index],self.dataset_name))
        
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
       
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target

        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if W-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        # apply_trans = transforms_aug[random.getrandbits(3)]

        # clean = getattr(augment, apply_trans)(clean)
        # noisy = getattr(augment, apply_trans)(noisy)        
        
        return clean, noisy,r,c ,clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, dataset_name,target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        self.dataset_name = dataset_name

        gt_dir = 'GT'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        
        
        clean = torch.from_numpy(load_img(self.clean_filenames[tar_index]),self.dataset_name)
        noisy = torch.from_numpy(load_img(self.noisy_filenames[tar_index]),self.dataset_name)
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        
        return clean, noisy, clean_filename, noisy_filename
    def return_size(self):
        # if self.data_format != 0:
        #     clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[0])))
        # else:
        clean = torch.from_numpy(np.float32(np.load(self.clean_filenames[0])))
        
        clean = clean.permute(2,0,1) # to CHW
        return clean.shape[0], clean.shape[1], clean.shape[2]

class DataLoaderVal_deblur(Dataset):

    def __init__(self, rgb_dir, img_options=None, dataset_name=None, rgb_dir2=None):
        super(DataLoaderVal_deblur, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'GT')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files]
        self.tar_filenames = [os.path.join(rgb_dir, 'GT', x) for x in tar_files]
        self.img_options = img_options
        self.dataset_name = dataset_name
        self.tar_size       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size'] if img_options is not None else None

    def __len__(self):
        return self.tar_size
  
    def __getitem__(self, index):
        index_ = index % self.tar_size
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        inp_img =  torch.from_numpy(load_img(self.inp_filenames[index_],self.dataset_name))
        tar_img =  torch.from_numpy(load_img(self.tar_filenames[index_],self.dataset_name))

        inp_img = inp_img.permute(2,0,1)
        tar_img = tar_img.permute(2,0,1)
      
        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
      
        filename2 = os.path.splitext(os.path.split(tar_path)[-1])[0]
        return tar_img, inp_img, filename, filename2
    
    def return_size(self):
        # if self.data_format != 0:
        #     clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[0])))
        # else:
        clean = torch.from_numpy(np.float32(np.load(self.tar_filenames[0])))
        
        clean = clean.permute(2,0,1) # to CHW
        return clean.shape[0], clean.shape[1], clean.shape[2]
        
##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename


def get_training_data(opt,rgb_dir, img_options,dataset_name):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(opt,rgb_dir, img_options,dataset_name, None)


def get_validation_deblur_data(rgb_dir, img_options, dataset_name):
    print(rgb_dir)
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_deblur(rgb_dir, img_options, dataset_name, None)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)
