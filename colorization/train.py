
#****************************************#
#***          Import Modules          ***#
#****************************************#
import os
import time
import numpy as np 
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, TensorDataset
from colorization import Colorization


#****************************************#
#***           Configuration          ***#
#****************************************#
class Configuration:
    model_file_name = 'checkpoint.pt'
    load_model_to_train = False
    load_model_to_test = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    point_batches = 100


#****************************************#
#***         Hyper Parameters         ***#
#****************************************#
class HyperParameters:
    epochs = 1
    batch_size = 32
    learning_rate = 0.001

config = Configuration()
hparams = HyperParameters()


#****************************************#
#***       Architecture Pipeline      ***#
#****************************************#
model = Colorization(256).to(config.device) 
inception_model = models.inception_v3(pretrained=True).to(config.device)
inception_model.eval()
loss_criterion = torch.nn.MSELoss(reduction='mean').to(config.device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=1e-6)


#****************************************#
#***  Training & Validation Pipeline  ***#
#****************************************#

train_dataset = torchvision.datasets.ImageFolder(root='data/train', 
                                               transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.Resize((224,224),interpolation=3),torchvision.transforms.ToTensor()]))
train_dataset = Dataset('data/train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=8)
validation_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=8)


for epoch in range(epochs):
    print('Starting epoch:',epoch)
    # Training step
    loop_start = time.time()
    for idx,(img_l,img_ab) in enumerate(train_dataloader):
        optimizer.zero_grad()
        img_embs = inception_model(img_l)
        output_ab = model(img_l,img_embs)
        
        loss = criterion(output_ab, img_ab)
        loss.backward()
        
        scheduler.step()
        optimizer.step()
        
        avg_loss += loss.item()

        if batch%config.point_batches==0: 
            loop_end = time.time()   
            print('Batch:' batch ,'of', batches, '| Processing time for',config.point_batches,'batches:',loop_end-loop_start)
            loop_start = time.time()


    for idx,(img_l,img_ab) in enumerate(validation_dataloader):
        model.eval()

    # Save the variables to disk
    checkpoint = {'model': model,'model_state_dict': model.state_dict(),\
                  'optimizer' : optimizer,'optimizer_state_dict' : optimizer.state_dict(),\
                  'train_loss':train_loss, 'train_acc':train_acc,
                  'val_loss':val_loss, 'val_acc':val_acc}
    torch.save(checkpoint, config.model_file_name)
    print("Model saved at:",os.getcwd(),'/',config.model_file_name)
    print('')
