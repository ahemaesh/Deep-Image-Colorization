
# coding: utf-8

'''
Deep Colorization
Deep learning final project for conversion of gray scale images to rgb
Contributors: Bhumi Bhanushali, Avinash Hemaeshwara Raju, Kathan Nilesh Mehta, Atulya Ravishankar
'''

# ### Import Modules

import os
import time
import numpy as np 
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import cv2


# ### Configuration

class Configuration:
    model_file_name = 'pretrained_models/checkpoint'
    load_model_to_train = False
    load_model_to_test = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    point_batches = 500


# ### Hyper Parameters

class HyperParameters:
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    num_workers = 16
    learning_rate_decay = 0.2


config = Configuration()
hparams = HyperParameters()
print('Device:',config.device)


# ### Custom Dataloader

class CustomDataset(Dataset):
    def __init__(self, root_dir, process_type):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir)]
        self.process_type = process_type
        print('File[0]:',self.files[0],'| Total Files:', len(self.files), '| Process:',self.process_type,)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            #*** Read the image from file ***
            self.rgb_img = cv2.imread(os.path.join(self.root_dir,self.files[index]))
            
            if self.rgb_img is None:
                raise Exception

            self.rgb_img = self.rgb_img.astype(np.float32) 
            self.rgb_img /= 255.0 
            
            #*** Resize the color image to pass to encoder ***
            rgb_encoder_img = cv2.resize(self.rgb_img, (224, 224))
            
            #*** Resize the color image to pass to decoder ***
            rgb_resnet_img = cv2.resize(self.rgb_img, (300, 300))
            
            ''' Encoder Images '''
            #*** Convert the encoder color image to normalized lab space ***
            self.lab_encoder_img = cv2.cvtColor(rgb_encoder_img,cv2.COLOR_BGR2Lab) 
            
            #*** Splitting the lab images into l-channel, a-channel, b-channel ***
            l_encoder_img, a_encoder_img, b_encoder_img = self.lab_encoder_img[:,:,0],self.lab_encoder_img[:,:,1],self.lab_encoder_img[:,:,2]
            
            #*** Normalizing l-channel between [-1,1] ***
            l_encoder_img = l_encoder_img/50.0 - 1.0
            
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
            #*** Convert the resnet color image to lab space ***
            self.lab_resnet_img = cv2.cvtColor(rgb_resnet_img,cv2.COLOR_BGR2Lab)
            
            #*** Extract the l-channel of resnet lab image *** 
            l_resnet_img = self.lab_resnet_img[:,:,0]/50.0 - 1.0
             
            #*** Convert the resnet l-image to torch Tensor and stack it in 3 channels ***
            l_resnet_img = torchvision.transforms.ToTensor()(l_resnet_img)
            l_resnet_img = l_resnet_img.expand(3,-1,-1)
            
            ''' return images to data-loader '''
            rgb_encoder_img = torchvision.transforms.ToTensor()(rgb_encoder_img)
            return l_encoder_img, ab_encoder_img, l_resnet_img, rgb_encoder_img, self.files[index]
        
        except Exception as e:
            print('Exception at ',self.files[index], e)
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'

    def show_rgb(self, index):
        self.__getitem__(index)
        print("RGB image size:", self.rgb_img.shape)        
        cv2.imshow(self.rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_lab_encoder(self, index):
        self.__getitem__(index)
        print("Encoder Lab image size:", self.lab_encoder_img.shape)
        cv2.imshow(self.lab_encoder_img)
        c2.waitKey(0)
        cv2.destroyAllWindows()

    def show_lab_resnet(self, index):
        self.__getitem__(index)
        print("Inception Lab image size:", self.lab_resnet_img.shape)
        cv2.imshow(self.lab_resnet_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_other_images(self, index):
        a,b,c,d,_ = self.__getitem__(index)
        print("Encoder l channel image size:",a.shape)
        cv2.imshow((a.detach().numpy().transpose(1,2,0)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Encoder ab channel image size:",b.shape)
        cv2.imshow((b.detach().numpy().transpose(1,2,0)[:,:,0]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow((b.detach().numpy().transpose(1,2,0)[:,:,1]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Inception l channel image size:",c.shape)
        cv2.imshow(c.detach().numpy().transpose(1,2,0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Color resized image size:",d.shape)
        cv2.imshow(d.detach().numpy().transpose(1,2,0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# ### Encoder

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

resnet_model = models.resnet50(pretrained=True,progress=True).float().to(config.device)
resnet_model.eval()
resnet_model = resnet_model.float()

loss_criterion = torch.nn.MSELoss(reduction='mean').to(config.device)
milestone_list  = list(range(0,hparams.epochs,5))
writer = SummaryWriter()
model = Colorization(256)
optimizer = torch.optim.Adam(model.parameters(),lr=hparams.learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_list, gamma=hparams.learning_rate_decay)

if config.load_model_to_train or config.load_model_to_test:
    checkpoint = torch.load(config.model_file_name,map_location=torch.device(config.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k,v in state.items():
            if isinstance(v,torch.Tensor):
                state[k] = v.cuda()
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print('Loaded pretrain model | Previous train loss:',checkpoint['train_loss'], '| Previous validation loss:',checkpoint['val_loss'])
    print('Loaded Schedule :', scheduler)
    print('Loaded Optimizer : ', optimizer)


model = model.to(config.device) 
resnet_model = resnet_model.to(config.device)


# ### Data Loaders

if not config.load_model_to_test:
    train_dataset = CustomDataset('data/train','train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers)
    
    validataion_dataset = CustomDataset('data/validation','validation')
    validation_dataloader = torch.utils.data.DataLoader(validataion_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers)
    
    print('Train:',len(train_dataloader), '| Total Images:',len(train_dataloader)*hparams.batch_size)
    print('Valid:',len(validation_dataloader), '| Total Images:',len(validation_dataloader)*hparams.batch_size)


# ### Training & Validation Pipeline

if not config.load_model_to_test:
    flag = True
    for epoch in range(hparams.epochs):
        print('Starting epoch:',epoch+1)

        #*** Training step ***
        loop_start = time.time()
        avg_loss = 0.0
        batch_loss = 0.0
        main_start = time.time()
        model.train()

        for idx,(img_l_encoder, img_ab_encoder, img_l_resnet, img_rgb, file_name) in enumerate(train_dataloader):
            #*** Skip bad data ***
            if not img_l_encoder.ndim:
                continue

            #*** Move data to GPU if available ***
            img_l_encoder = img_l_encoder.to(config.device)
            img_ab_encoder = img_ab_encoder.to(config.device)
            img_l_resnet = img_l_resnet.to(config.device)

            #*** Initialize Optimizer ***
            optimizer.zero_grad()

            #*** Forward Propagation ***
            img_embs = resnet_model(img_l_resnet.float())
            output_ab = model(img_l_encoder,img_embs)

            if flag:
                print(img_embs.size())
                flag = False
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
                print('Batch:',idx, '| Processing time for',config.point_batches,':',str(loop_end-loop_start)+'s',' | Batch Loss:', batch_loss/config.point_batches)
                loop_start = time.time()
                batch_loss = 0.0

        #*** Print Training Data Stats ***
        train_loss = avg_loss/len(train_dataloader)*hparams.batch_size
        writer.add_scalar('Loss/train', train_loss, epoch)
        print('Training Loss:',train_loss,'| Processed in ',str(time.time()-main_start)+'s')

        #*** Validation Step ***       
        avg_loss = 0.0
        loop_start = time.time()
        #*** Intialize Model to Eval Mode for validation ***
        model.eval()
        for idx,(img_l_encoder, img_ab_encoder, img_l_resnet, img_rgb, file_name) in enumerate(validation_dataloader):
            #*** Skip bad data ***
            if not img_l_encoder.ndim:
                continue

            #*** Move data to GPU if available ***
            img_l_encoder = img_l_encoder.to(config.device)
            img_ab_encoder = img_ab_encoder.to(config.device)
            img_l_resnet = img_l_resnet.to(config.device)

            #*** Forward Propagation ***
            img_embs = resnet_model(img_l_resnet.float())
            output_ab = model(img_l_encoder,img_embs)

            #*** Loss Calculation ***
            loss = loss_criterion(output_ab, img_ab_encoder.float())
            avg_loss += loss.item()

        val_loss = avg_loss/len(validation_dataloader)*hparams.batch_size
        writer.add_scalar('Loss/validation', val_loss, epoch)
        print('Validation Loss:', val_loss,'| Processed in ',str(time.time()-loop_start)+'s')

        #*** Save the Model to disk ***
        checkpoint = {'model_state_dict': model.state_dict(),\
                      'optimizer_state_dict' : optimizer.state_dict(), \
                      'scheduler_state_dict' : scheduler.state_dict(),\
                      'train_loss':train_loss, 'val_loss':val_loss}
        torch.save(checkpoint, config.model_file_name+'.'+str(epoch+1))
        print("Model saved at:",os.getcwd()+'/'+str(config.model_file_name)+'.'+str(epoch+1))

# ### Inference

test_dataset = CustomDataset('data/test','test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=hparams.num_workers)
print('Test: ',len(test_dataloader), '| Total Image:',len(test_dataloader))


# ##### Convert Tensor Image -> Numpy Image -> Color  Image -> Tensor Image

def concatente_and_colorize(im_lab, img_ab):
    # Assumption is that im_lab is of size [1,3,224,224]
    #print(im_lab.size(),img_ab.size())
    np_img = im_lab[0].cpu().detach().numpy().transpose(1,2,0)
    lab = np.empty([*np_img.shape[0:2], 3],dtype=np.float32)
    lab[:, :, 0] = np.squeeze(((np_img + 1) * 50))
    lab[:, :, 1:] = img_ab[0].cpu().detach().numpy().transpose(1,2,0) * 127
    np_img = cv2.cvtColor(lab,cv2.COLOR_Lab2RGB) 
    color_im = torch.stack([torchvision.transforms.ToTensor()(np_img)],dim=0)
    return color_im


#*** Inference Step ***
avg_loss = 0.0
loop_start = time.time()
batch_start = time.time()
batch_loss = 0.0

for idx,(img_l_encoder, img_ab_encoder, img_l_resnet, img_rgb, file_name) in enumerate(test_dataloader):
        #*** Skip bad data ***
        if not img_l_encoder.ndim:
            continue
            
        #*** Move data to GPU if available ***
        img_l_encoder = img_l_encoder.to(config.device)
        img_ab_encoder = img_ab_encoder.to(config.device)
        img_l_resnet = img_l_resnet.to(config.device)
        
        #*** Intialize Model to Eval Mode ***
        model.eval()
        
        #*** Forward Propagation ***
        img_embs = resnet_model(img_l_resnet.float())
        output_ab = model(img_l_encoder,img_embs)
        
        #*** Adding l channel to ab channels ***
        color_img = concatente_and_colorize(torch.stack([img_l_encoder[:,0,:,:]],dim=1),output_ab)
        color_img_jpg = color_img[0].detach().numpy().transpose(1,2,0)
        # cv2.imshow(color_img_jpg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('outputs/'+file_name[0],color_img_jpg*255)
        save_image(color_img[0],'outputs/'+file_name[0])

#       #*** Printing to Tensor Board ***
        grid = torchvision.utils.make_grid(color_img)
        writer.add_image('Output Lab Images', grid, 0)
        
        #*** Loss Calculation ***
        loss = loss_criterion(output_ab, img_ab_encoder.float())
        avg_loss += loss.item()
        batch_loss += loss.item()

        if idx%config.point_batches==0: 
            batch_end = time.time()   
            print('Batch:',idx, '| Processing time for',config.point_batches,':',str(batch_end-batch_start)+'s', '| Batch Loss:', batch_loss/config.point_batches)
            batch_start = time.time()
            batch_loss = 0.0
        
test_loss = avg_loss/len(test_dataloader)
print('Test Loss:',avg_loss/len(test_dataloader),'| Processed in ',str(time.time()-loop_start)+'s')
writer.close() 
