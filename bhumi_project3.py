# -*- coding: utf-8 -*-
"""HW3P2CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N4IYxE5L7NqPaTudOlSRhic7xKZ4yqCF
"""

import torchvision.models as models
inception = models.inception_v3(pretrained=True)

print(inception)

!cd Train\ data

!unzip "/content/drive/My Drive/COCO Dataset/train2017.zip"

import os
import numpy as np
import math
# from PIL import Image

import torch
import torchvision
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import matplotlib.image as mpimg
# from PIL import Image

from google.colab import files
files.upload()

!unzip "/content/imagefolder.zip"

from skimage import io, color
from skimage.transform import resize
import os

class MyData(Dataset):
    def __init__(self, root_dir):
        self.root_dir =root_dir
        self.files = [f for f in os.listdir(root_dir)]
        print(self.files[0])
    def __getitem__(self, index):
        self.rgb = io.imread(os.path.join(self.root_dir,self.files[index]))
        
        rgb1 = resize(self.rgb, (225, 225),anti_aliasing=True)
        rgb2 = resize(self.rgb, (300, 300),anti_aliasing=True)

        self.lab1 = color.rgb2lab(rgb1)

        l1 = self.lab1[:,:,0]
        l1 = np.stack((l1,)*3,axis = -1).transpose(2,0,1)
        l1 = torchvision.transforms.ToTensor()(l1)

        ab1 = self.lab1[:,:,1:3].transpose(2,0,1)
        ab1 = torchvision.transforms.ToTensor()(ab1)

        self.lab2 = color.rgb2lab(rgb2)
        l2 = self.lab2[:,:,0]
        l2 = np.stack((l2,)*3,axis = -1).transpose(2,0,1)
        l2 = torchvision.transforms.ToTensor()(l2)

        return l1, ab1, l2

    def len(self):
        return len(self.files)

    def show_rgb(self, index):
        print("RGB image size:", self.rgb.shape)
        io.imshow(self.rgb)

    def show_lab1(self, index):
        print("Lab image 1 size:", self.lab1.shape)
        io.imshow(self.lab1)

    def show_lab2(self, index):
        print("Lab image 2 size:", self.lab2.shape)
        io.imshow(self.lab2)

MyDataset = MyData("/content/imagefolder")

l1 ,ab, l2 = MyDataset[2]

MyDataset.show_rgb(2)

MyDataset.show_lab1(2)

MyDataset.show_lab2(2)

train_dataset = torchvision.datasets.ImageFolder(root='/content/Train data', 
                                               transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.Resize((224,224),interpolation=3),torchvision.transforms.ToTensor() ]))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8)

# print(train_dataset[1][0].shape)
print(len(train_dataset))
for i in range(1000):
    print("Training example ",str(i) ," size:",train_dataset[i][0].shape)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

inception.train()
inception.to(DEVICE)
for batch_num, (feats, labels) in enumerate(train_dataloader):
    feats, labels = feats.to(DEVICE), labels.to(DEVICE)
    outputs = inception(feats)
    print(outputs.shape)
    break

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
files.upload()

!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle

!chmod 600 /root/.kaggle/kaggle.json

!kaggle competitions download -c imagenet-object-localization-challenge

