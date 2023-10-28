import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def _norm(x, norm=1024):
    return x / norm


def _tonemap(x, k=0.25):
    return x / (x + k)


def set_mapping(mode: str):
    if mode == "norm":
        return _norm
    elif mode == "tonemap":
        return _tonemap
    else:
        raise NotImplementedError("mode should be norm or tonemap")


class my_dataset(Dataset):
    def __init__(self, root_in, root_label, mapping="norm", crop_size=256):
        super(my_dataset, self).__init__()
        self.mapping = set_mapping(mapping)

        # in_imgs
        in_files = os.listdir(root_in)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        # gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]

        self.crop_size = crop_size

    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        in_img = self.mapping(np.load(in_img_path))  # Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = self.mapping(np.load(gt_img_path))  # Image.open(gt_img_path)

        data_IN, data_GT = self.train_transform(in_img, gt_img, self.crop_size)
        return data_IN, data_GT

    def train_transform(self, img, label, patch_size=256):
        ih, iw, _ = img.shape
        patch_size = patch_size
        ix = random.randrange(0, max(0, iw - patch_size))
        iy = random.randrange(0, max(0, ih - patch_size))
        img = img[iy : iy + patch_size, ix : ix + patch_size, :]
        label = label[iy : iy + patch_size, ix : ix + patch_size, :]

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(img)
        label = transform(label)

        return img, label

    def __len__(self):
        return len(self.imgs_in)


class my_dataset_eval(Dataset):
    def __init__(self, root_in, root_label, mapping="norm", transform=None):
        super(my_dataset_eval, self).__init__()
        self.mapping = self.mapping = set_mapping(mapping)
        # in_imgs
        in_files = os.listdir(root_in)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        # gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]

        self.transform = transform

    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        in_img = self.mapping(np.load(in_img_path))  # Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = self.mapping(np.load(gt_img_path))  # Image.open(gt_img_path)

        img_name = in_img_path.split("/")[-1]

        if self.transform:
            data_IN = self.transform(in_img)
            data_GT = self.transform(gt_img)
            print(data_IN.shape, data_GT.shape)
            exit(0)
        else:
            data_IN = np.asarray(in_img)
            data_IN = torch.from_numpy(data_IN)
            data_IN = data_IN.permute(2, 0, 1)
            data_GT = np.asarray(gt_img)
            data_GT = torch.from_numpy(data_GT)
            data_GT = data_GT.permute(2, 0, 1)
        return data_IN, data_GT, img_name

    def __len__(self):
        return len(self.imgs_in)
