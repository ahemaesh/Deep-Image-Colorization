
# coding: utf-8

# # Deep Colorization
# ### Deep learning final project for conversion of gray scale images to rgb
# ### Contributors: Bhumi Bhanushali, Avinash Hemaeshwara Raju, Kathan Nilesh Mehta, Atulya Ravishankar

# ### Download Dataset

# In[1]:


# wget -N images.cocodataset.org/zips/train2017.zip
# wget -N images.cocodataset.org/zips/val2017.zip
# wget -N images.cocodataset.org/zips/test2017.zip
# pip3 install tensorboard
# tensorboard --logdir=runs


# ### Import Modules

# In[2]:


import os
import time
import numpy as np 
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from skimage import io, color
from skimage.transform import resize


# ### Configuration

# In[3]:


class Configuration:
    model_file_name = 'checkpoint.pt'
    load_model_to_train = False
    load_model_to_test = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    point_batches = 100


# ### Hyper Parameters

# In[6]:


class HyperParameters:
    epochs = 1
    batch_size = 32
    learning_rate = 0.001
    num_workers = 16
    learning_rate_decay = 0.5


# In[7]:


config = Configuration()
hparams = HyperParameters()
print('Device:',config.device)


# ### Custom Dataloader

# In[8]:


class CustomDataset(Dataset):
    def __init__(self, root_dir, process_type):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir)]
        self.process_type = process_type
        print('File[0]:',self.files[0],'| Total Files:', len(self.files), '| Process:',self.process_type,)

    def __len__(self):
        return 1024#len(self.files)

    def __getitem__(self, index):
        try:
            #*** Read the image from file ***
            self.rgb_img = io.imread(os.path.join(self.root_dir,self.files[index]),plugin='matplotlib') 
            
            #*** Resize the color image to pass to encoder ***
            rgb_encoder_img = resize(self.rgb_img, (224, 224))
            
            #*** Resize the color image to pass to decoder ***
            rgb_inception_img = resize(self.rgb_img, (300, 300))
            
            ''' Encoder Images '''
            #*** Convert the encoder color image to normalized lab space ***
            self.lab_encoder_img = color.rgb2lab(rgb_encoder_img) 
            
            #*** Splitting the lab images into l-channel, a-channel, b-channel ***
            l_encoder_img, a_encoder_img, b_encoder_img = self.lab_encoder_img[:,:,0],self.lab_encoder_img[:,:,1],self.lab_encoder_img[:,:,2]
            
            #*** Normalizing l-channel between [-1,1] ***
            l_encoder_img = (2*l_encoder_img/100.0)-1.0
            
            #*** Repeat the l-channel to 3 dimensions ***
            l_encoder_img = torchvision.transforms.ToTensor()(l_encoder_img)
            l_encoder_img = l_encoder_img.expand(3,-1,-1)
            
            #*** Normalize a and b channels and concatenate ***
            a_encoder_img = (a_encoder_img/128.0)
            b_encoder_img = (b_encoder_img/128.0)
            a_encoder_img = torch.stack([torch.Tensor(a_encoder_img)])
            b_encoder_img = torch.stack([torch.Tensor(b_encoder_img)])
            ab_encoder_img = torch.cat([a_encoder_img, b_encoder_img], dim=0)
            
            ''' Inception Images '''
            #*** Convert the inception color image to lab space ***
            self.lab_inception_img = color.rgb2lab(rgb_inception_img)
            
            #*** Extract the l-channel of inception lab image *** 
            l_inception_img = self.lab_inception_img[:,:,0]
            
            #*** Convert the inception l-image to torch Tensor and stack it in 3 channels ***
            l_inception_img = torchvision.transforms.ToTensor()(l_inception_img)
            l_inception_img = l_inception_img.expand(3,-1,-1)
            
            ''' return images to data-loader '''
            rgb_encoder_img = torchvision.transforms.ToTensor()(rgb_encoder_img)
            return l_encoder_img, ab_encoder_img, l_inception_img, rgb_encoder_img, self.files[index]
        
        except Exception as e:
            print('Exception at ',self.files[index], e)
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'

    def show_rgb(self, index):
        self.__getitem__(index)
        print("RGB image size:", self.rgb_img.shape)        
        plt.imshow(self.rgb_img)
        plt.show()

    def show_lab_encoder(self, index):
        self.__getitem__(index)
        print("Encoder Lab image size:", self.lab_encoder_img.shape)
        plt.imshow(self.lab_encoder_img)
        plt.show()

    def show_lab_inception(self, index):
        self.__getitem__(index)
        print("Inception Lab image size:", self.lab_inception_img.shape)
        plt.imshow(self.lab_inception_img)
        plt.show()
    
    def show_other_images(self, index):
        a,b,c,d,_ = self.__getitem__(index)
        print("Encoder l channel image size:",a.shape)
        plt.imshow((a.detach().numpy().transpose(1,2,0)))#+1)*50)
        plt.show()
        print("Encoder ab channel image size:",b.shape)
        plt.imshow((b.detach().numpy().transpose(1,2,0)[:,:,0])*128)
        plt.show()
        plt.imshow((b.detach().numpy().transpose(1,2,0)[:,:,1])*128)
        plt.show()
        print("Inception l channel image size:",c.shape)
        plt.imshow(c.detach().numpy().transpose(1,2,0))
        plt.show()
        print("Color resized image size:",d.shape)
        plt.imshow(d.detach().numpy().transpose(1,2,0))
        plt.show()


# # In[9]:


# train_dataset = CustomDataset('data/train','train')


# # In[10]:


# train_dataset.show_rgb(0)
# train_dataset.show_lab_encoder(0)
# train_dataset.show_lab_inception(0)
# train_dataset.show_other_images(0)


# ### Encoder

# In[16]:


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        self.model = self.model.float()
        return self.model(x.float())


# ### Fusion Layer

# In[17]:


class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer,self).__init__()

    def forward(self, inputs, mask=None):
        ip, emb = inputs
        emb = torch.stack([torch.stack([emb],dim=2)],dim=3)
        emb = emb.repeat(1,1,ip.shape[2],ip.shape[3])
        fusion = torch.cat((ip,emb),1)
        return fusion


# ### Decoder

# In[18]:


class Decoder(nn.Module):
    def __init__(self, input_depth):
        super(Decoder,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2.0),
        )

    def forward(self, x):
        return self.model(x)


# ### Network Definition

# In[19]:


class Colorization(nn.Module):
    def __init__(self, depth_after_fusion):
        super(Colorization,self).__init__()
        self.encoder = Encoder()
        self.fusion = FusionLayer()
        self.after_fusion = nn.Conv2d(in_channels=1256, out_channels=depth_after_fusion,kernel_size=1, stride=1,padding=0)
        self.bnorm = nn.BatchNorm2d(256)
        self.decoder = Decoder(depth_after_fusion)

    def forward(self, img_l, img_emb):
        img_enc = self.encoder(img_l)
        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)
        fusion = self.bnorm(fusion)
        return self.decoder(fusion)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


# ### Architecture Pipeline

# In[22]:


if config.load_model_to_train or config.load_model_to_test:
    checkpoint = torch.load(config.model_file_name,map_location=torch.device(config.device))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device) 
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Loaded pretrain model | Previous train loss:',checkpoint['train_loss'], '| Previous validation loss:',checkpoint['val_loss'])
else:
    model = Colorization(256).to(config.device) 
#     model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(),lr=hparams.learning_rate, weight_decay=1e-6)

inception_model = models.inception_v3(pretrained=True).float().to(config.device)
inception_model = inception_model.float()
inception_model.eval()
loss_criterion = torch.nn.MSELoss(reduction='mean').to(config.device)
milestone_list  = list(range(0,hparams.epochs,2))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_list, gamma=hparams.learning_rate_decay)
writer = SummaryWriter()


# In[23]:


print(model)


# ### Data Loaders

# In[24]:


if not config.load_model_to_test:
    train_dataset = CustomDataset('data/train','train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers)
    

    validataion_dataset = CustomDataset('data/validation','validation')
    validation_dataloader = torch.utils.data.DataLoader(validataion_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers)
    
    print('Train:',len(train_dataloader), '| Total Images:',len(train_dataloader)*hparams.batch_size)
    print('Valid:',len(validation_dataloader), '| Total Images:',len(validation_dataloader)*hparams.batch_size)


# ### Training & Validation Pipeline

# In[25]:


if not config.load_model_to_test:
    for epoch in range(hparams.epochs):
        print('Starting epoch:',epoch+1)

        #*** Training step ***
        loop_start = time.time()
        avg_loss = 0.0
        batch_loss = 0.0
        main_start = time.time()
        model.train()

        for idx,(img_l_encoder, img_ab_encoder, img_l_inception, img_rgb, file_name) in enumerate(train_dataloader):
            #*** Skip bad data ***
            if not img_l_encoder.ndim:
                continue

            #*** Move data to GPU if available ***
            img_l_encoder = img_l_encoder.to(config.device)
            img_ab_encoder = img_ab_encoder.to(config.device)
            img_l_inception = img_l_inception.to(config.device)

            #*** Initialize Optimizer ***
            optimizer.zero_grad()

            #*** Forward Propagation ***
            img_embs = inception_model(img_l_inception.float())
            output_ab = model(img_l_encoder,img_embs)

            #*** Back propogation ***
            loss = loss_criterion(output_ab, img_ab_encoder.float())
            loss.backward()

            #*** Weight Update ****
            optimizer.step()

            #*** Reduce Learning Rate ***
            scheduler.step()

            #*** Loss Calculation ***
            avg_loss += loss.item()
            batch_loss += loss.item()

            #*** Print stats after every point_batches ***
            if idx%config.point_batches==0: 
                loop_end = time.time()   
                print('Batch:',idx, '| Processing time for',config.point_batches,':',loop_end-loop_start,'s | Batch Loss:', batch_loss/config.point_batches)
                loop_start = time.time()
                batch_loss = 0.0

        #*** Print Training Data Stats ***
        train_loss = avg_loss/len(train_dataloader)*hparams.batch_size
        writer.add_scalar('Loss/train', train_loss, epoch)
        print('Training Loss:',train_loss,'| Processed in ',time.time()-main_start,'s')

        #*** Validation Step ***       
        avg_loss = 0.0
        loop_start = time.time()
        #*** Intialize Model to Eval Mode for validation ***
        model.eval()
        for idx,(img_l_encoder, img_ab_encoder, img_l_inception, img_rgb, file_name) in enumerate(validation_dataloader):
            #*** Skip bad data ***
            if not img_l_encoder.ndim:
                continue

            #*** Move data to GPU if available ***
            img_l_encoder = img_l_encoder.to(config.device)
            img_ab_encoder = img_ab_encoder.to(config.device)
            img_l_inception = img_l_inception.to(config.device)

            #*** Forward Propagation ***
            img_embs = inception_model(img_l_inception.float())
            output_ab = model(img_l_encoder,img_embs)

            #*** Loss Calculation ***
            loss = loss_criterion(output_ab, img_ab_encoder.float())
            avg_loss += loss.item()

        val_loss = avg_loss/len(validation_dataloader)*hparams.batch_size
        writer.add_scalar('Loss/validation', val_loss, epoch)
        print('Validation Loss:', val_loss,'| Processed in ',time.time()-loop_start,'s')

        #*** Save the Model to disk ***
        checkpoint = {'model': model,'model_state_dict': model.state_dict(), 'optimizer' : optimizer,'optimizer_state_dict' : optimizer.state_dict(),                      'train_loss':train_loss, 'val_loss':val_loss}
        torch.save(checkpoint, config.model_file_name)
        print("Model saved at:",os.getcwd(),'/',config.model_file_name)


# ### Inference

# In[ ]:


test_dataset = CustomDataset('data/test','test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=hparams.num_workers)
print('Test: ',len(test_dataloader), '| Total Image:',len(test_dataloader))


# ##### Convert Tensor Image -> Numpy Image -> Color  Image -> Tensor Image

# In[ ]:


def concatente_and_colorize(im_lab, img_ab):
    # Assumption is that im_lab is of size [1,3,224,224]
    #print(im_lab.size(),img_ab.size())
    np_img = im_lab[0].cpu().detach().numpy().transpose(1,2,0)
    lab = np.empty([*np_img.shape[0:2], 3],dtype=np.float32)
    lab[:, :, 0] = np.squeeze(((np_img + 1) * 50))
    lab[:, :, 1:] = img_ab[0].cpu().detach().numpy().transpose(1,2,0) * 128
    np_img = color.rgb2lab(lab) 
    color_im = torch.stack([torchvision.transforms.ToTensor()(np_img)],dim=0)
    return color_im


# In[ ]:


#*** Inference Step ***
avg_loss = 0.0
loop_start = time.time()
for idx,(img_l_encoder, img_ab_encoder, img_l_inception, img_rgb, file_name) in enumerate(test_dataloader):
        #*** Skip bad data ***
        if not img_l_encoder.ndim:
            continue
            
        #*** Move data to GPU if available ***
        img_l_encoder = img_l_encoder.to(config.device)
        img_ab_encoder = img_ab_encoder.to(config.device)
        img_l_inception = img_l_inception.to(config.device)
        
        #*** Intialize Model to Eval Mode ***
        model.eval()
        
        #*** Forward Propagation ***
        img_embs = inception_model(img_l_inception.float())
        output_ab = model(img_l_encoder,img_embs)
        
        #*** Adding l channel to ab channels ***
        color_img = concatente_and_colorize(torch.stack([img_l_encoder[:,0,:,:]],dim=1),output_ab)
        #img_lab = concatente_and_colorize(torch.stack([img_l_encoder[:,0,:,:]],dim=1),output_ab)
        color_img_jpg = color_img[0].detach().numpy().transpose(1,2,0)
        # plt.imshow(color_img_jpg)
        # plt.show()
        plt.imsave('outputs/'+file_name[0],color_img_jpg)
        
        
#         #*** Printing to Tensor Board ***
        grid = torchvision.utils.make_grid(color_img)
        writer.add_image('Output Lab Images', grid, 0)
        
        #*** Loss Calculation ***
        loss = loss_criterion(output_ab, img_ab_encoder.float())
        avg_loss += loss.item()
        
test_loss = avg_loss/len(test_dataloader)
writer.add_scalar('Loss/test', test_loss, epoch)
print('Test Loss:',avg_loss/len(test_dataloader),'| Processed in ',time.time()-loop_start,'s')


# In[ ]:


writer.close()

