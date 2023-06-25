import os
import random

import numpy as np
import imageio
import cv2
from PIL import Image

import torch
from torch import nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms

import albumentations as A
import matplotlib.pyplot as plt
from torch import einsum


class PolypDataset(data.Dataset):
    """
    Polyp Dataset for Orientation Classification (for training)
    """

    def __init__(self,
                 root_path='./Datasets/VANet_Dataset/TrainDataset/',
                 shape=(352, 352)):
        """
        params:
            root_path : saves the image dirs and corresponding label
            shape (h, w) : the size of image
        """

        super(PolypDataset, self).__init__()

        # Load Images and responding GTs
        self.img_path = []
        self.label = []
        self.mask_path = []

        # Parallel:0 Vertical:1
        dirs = ['Parallel', 'Vertical']
        mask_path = root_path + 'mask'

        for i, dir_name in enumerate(dirs):
            for name in os.listdir(root_path + dir_name):
                if name.split('.')[-1] not in ['jpg', 'JPG', 'png', 'PNG']:
                    continue
                self.img_path.append(os.path.join(root_path + dir_name, name))
                self.mask_path.append(os.path.join(mask_path, name))
                self.label.append(i)

        # The size of dataset
        self.size = len(self.img_path)

        # Data Augmentation
        self.transform = A.Compose([
            A.Resize(shape[0], shape[1], always_apply=True),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=45, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])

        # Numpy to Tensor
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = np.array(self.rgb_loader(self.img_path[index]))
        mask = np.array(self.gray_loader(self.mask_path[index]))
        label = np.array(self.label[index])
        transformed = self.transform(image=img, mask=mask)
        image = self.img_transform(transformed['image'])
        mask = self.gt_transform(transformed['mask'])
        label = torch.tensor(label, dtype=torch.float32)
        return image, label, mask

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class ValDataset(data.Dataset):
    """
    Polyp Dataset for Orientation Classification (for validating)
    """

    def __init__(self,
                 root_path='./Datasets/VANet_Dataset/TrainDataset/',
                 shape=(352, 352)):
        """
        params:
            root_path : saves the image dirs and corresponding label
            shape (h, w) : the size of image
        """

        super(ValDataset, self).__init__()

        # Load Images and responding GTs
        self.img_path = []
        self.label = []
        self.mask_path = []

        # Parallel:0 Vertical:1
        dirs = ['Parallel', 'Vertical']
        mask_path = root_path + 'mask'

        for i, dir_name in enumerate(dirs):
            for name in os.listdir(root_path + dir_name):
                if name.split('.')[-1] not in ['jpg', 'JPG', 'png', 'PNG']:
                    continue
                self.img_path.append(os.path.join(root_path + dir_name, name))
                self.mask_path.append(os.path.join(mask_path, name))
                self.label.append(i)

        # The size of dataset
        self.size = len(self.img_path)

        # Data Augmentation
        self.transform = A.Compose([
            A.Resize(shape[0], shape[1], always_apply=True),
        ])

        # Numpy to Tensor
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = np.array(self.rgb_loader(self.img_path[index]))
        mask = np.array(self.gray_loader(self.mask_path[index]))
        label = np.array(self.label[index])
        transformed = self.transform(image=img, mask=mask)
        image = self.img_transform(transformed['image'])
        mask = self.gt_transform(transformed['mask'])
        label = torch.tensor(label, dtype=torch.float32)
        return image, label, mask

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset(data.Dataset):
    """
    Polyp Dataset for Orientation Classification (for testing)
    """

    def __init__(self,
                 root_path='./Datasets/VANet_Dataset/TestDataset/ETIS-LaribPolypDB/',
                 shape=(352, 352)):
        """
        params:
            root_path : saves the image dirs and corresponding label
            shape (h, w) : the size of image
        """
        super(TestDataset, self).__init__()

        # Load Images and responding GTs
        self.img_path = []
        self.label = []
        self.mask_path = []

        # Parallel:0 Vertical:1
        dirs = ['Parallel', 'Vertical']
        mask_path = root_path + 'mask'

        for i, dir_name in enumerate(dirs):
            for name in os.listdir(root_path + dir_name):
                if name.split('.')[-1] not in ['jpg', 'JPG', 'png', 'PNG']:
                    continue
                self.img_path.append(os.path.join(root_path + dir_name, name))
                self.mask_path.append(os.path.join(mask_path, name))
                self.label.append(i)

        # The size of dataset
        self.size = len(self.img_path)

        # Data Augmentation
        self.transform = A.Compose([
            A.Resize(shape[0], shape[1], always_apply=True),
        ])

        # Numpy to Tensor
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = np.array(self.rgb_loader(self.img_path[index]))
        mask = np.array(self.gray_loader(self.mask_path[index]))
        label = np.array(self.label[index])
        transformed = self.transform(image=img, mask=mask)
        image = self.img_transform(transformed['image'])
        mask = self.gt_transform(transformed['mask'])
        label = torch.tensor(label, dtype=torch.float32)
        return image, label, mask, self.img_path[index]

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == "__main__":
    label_list = ['Parallel', 'Vertical']
    dataset = PolypDataset()
    image, label, mask = dataset[0]
    image = einsum('c h w -> h w c', image)
    mask = einsum('c h w -> h w c', mask)
    image = image.numpy()
    mask = mask.numpy()
    image = 255 * (image - image.min()) / (image.max() - image.min())
    plt.subplot(121)
    plt.imshow(np.uint8(image))
    plt.title(label_list[int(label.numpy())])
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(np.uint8(255 * mask), cmap='gray')
    plt.title(label_list[int(label.numpy())])
    plt.axis('off')
    plt.show()
