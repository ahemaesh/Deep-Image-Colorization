
#****************************************#
#***          Import Modules          ***#
#****************************************#
import os
import time
import numpy as np 
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from network_definition import Colorization
from skimage import io, color
from skimage.transform import resize


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
    batch_size = 2
    learning_rate = 0.001
    num_workers = 4
    learning_rate_decay = 0.72

config = Configuration()
hparams = HyperParameters()
print('Device:',config.device)


#****************************************#
#***         Hyper Parameters         ***#
#****************************************#

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir =root_dir
        self.files = [f for f in os.listdir(root_dir)]
        print(self.files[0])

    def __len__(self):
        return 500#len(self.files)

    def __getitem__(self, index):
        self.rgb = io.imread(os.path.join(self.root_dir,self.files[index]))
        rgb_encoder = resize(self.rgb, (225, 225),anti_aliasing=True)
        rgb_inception = resize(self.rgb, (300, 300),anti_aliasing=True)

        self.lab_encoder = color.rgb2lab(rgb_encoder)
        l_encoder = self.lab_encoder[:,:,0]
        l_encoder = np.stack((l_encoder,)*3,axis = -1)
        l_encoder = torchvision.transforms.ToTensor()(l_encoder)
        ab_encoder = self.lab_encoder[:,:,1:3]
        ab_encoder = torchvision.transforms.ToTensor()(ab_encoder)

        self.lab_inception = color.rgb2lab(rgb_inception)
        l_inception = self.lab_inception[:,:,0]
        l_inception = np.stack((l_inception,)*3,axis = -1)
        l_inception = torchvision.transforms.ToTensor()(l_inception)

        return l_encoder.to(config.device), ab_encoder.to(config.device), l_inception.to(config.device)

    def show_rgb(self, index):
        self.__getitem__(index)
        print("RGB image size:", self.rgb.shape)
        io.imshow(self.rgb)

    def show_lab_encoder(self, index):
        self.__getitem__(index)
        print("Encoder Lab image size:", self.lab_encoder.shape)
        io.imshow(self.lab_encoder)

    def show_lab_inception(self, index):
        self.__getitem__(index)
        print("Inception Lab image size:", self.lab_inception.shape)
        io.imshow(self.lab_inception)


#****************************************#
#***       Architecture Pipeline      ***#
#****************************************#
# ipdb.set_trace()
model = Colorization(256).to(config.device) 
inception_model = models.inception_v3(pretrained=True).float().to(config.device)
inception_model = inception_model.float()
inception_model.eval()
loss_criterion = torch.nn.MSELoss(reduction='mean').to(config.device)
optimizer = torch.optim.Adam(model.parameters(),lr=hparams.learning_rate, weight_decay=1e-6)
milestone_list  = list(range(0,hparams.epochs,2))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_list, gamma=hparams.learning_rate_decay)

#****************************************#
#***  Training & Validation Pipeline  ***#
#****************************************#

train_dataset = CustomDataset('data/train')
validataion_dataset = CustomDataset('data/validation')
test_dataset = CustomDataset('data/test')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers)
validation_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers)


for epoch in range(hparams.epochs):
    print('Starting epoch:',epoch+1)

    # Training step
    loop_start = time.time()
    avg_loss = 0.0
    for idx,(img_l_encoder, img_ab_encoder, img_l_inception) in enumerate(train_dataloader):
        
        model.train()
        optimizer.zero_grad()
        img_embs = inception_model(img_l_inception.float())
        output_ab = model(img_l_encoder,img_embs)
        
        loss = criterion(output_ab, img_ab_encoder)
        loss.backward()
        
        scheduler.step()
        optimizer.step()
        
        avg_loss += loss.item()

        if idx%config.point_batches==0: 
            loop_end = time.time()   
            print('Batch:',idx, '| Processing time for',config.point_batches,':',loop_end-loop_start)
            loop_start = time.time()


    print('Training Loss:',avg_loss/len(train_dataloader*hparams.batch_size))

    #******     Validation-Accuracy     ******       
    avg_loss = 0.0
    loop_start = time.time()
    for idx,(img_l_encoder, img_ab_encoder, img_l_inception) in enumerate(validation_dataloader):
        model.eval()
        img_embs = inception_model(img_l_inception)
        output_ab = model(img_l_encoder,img_embs)
        loss = criterion(output_ab, img_ab_encoder)
        avg_loss += loss.item()

        if idx%config.point_batches==0: 
            loop_end = time.time()   
            print('Batch:', idx, '| Processing time for',config.point_batches,':',loop_end-loop_start)
            loop_start = time.time()

    print('Validation Loss:',avg_loss/len(validation_dataloader*hparams.batch_size))

    # Save the variables to disk
    checkpoint = {'model': model,'model_state_dict': model.state_dict(),\
                  'optimizer' : optimizer,'optimizer_state_dict' : optimizer.state_dict(),\
                  'train_loss':train_loss, 'train_acc':train_acc,
                  'val_loss':val_loss, 'val_acc':val_acc}
    torch.save(checkpoint, config.model_file_name)
    print("Model saved at:",os.getcwd(),'/',config.model_file_name)
