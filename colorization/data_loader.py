

import torchvision.models as models

inception = models.inception_v3(pretrained=True)





import os
import numpy as np
import math
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
from PIL import Image

train_dataset = torchvision.datasets.ImageFolder(root='/content/Train data', 
                                               transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224),interpolation=3),torchvision.transforms.ToTensor() ]))
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




