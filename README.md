# PyTorch Implementation of Deep Colorization
#### Our Implementation is derived from the Original Paper 'Deep Koalarization: Image Colorization usingCNNs and Inception-Resnet-v2': https://arxiv.org/pdf/1712.03400.pdf
#### This approach has been tested on Coco-dataset with 70000 train images, 5000 validation images and 10000 test images

### **The Coco dataset can be downloaded on AWS or Google Colab using the below command line arguments:**
>wget -N images.cocodataset.org/zips/train2017.zip<br />
>wget -N images.cocodataset.org/zips/val2017.zip<br />
>wget -N images.cocodataset.org/zips/test2017.zip<br />


### Final Results
![picture](Final_Results.png)


### **Download data from google drive use 'gdown'**
##### Install 'gdown' using 'pip install gdown'
##### Get data from the drive using the command line argument gdown https://drive.google.com/u?/id='FileID' , where FileID has to be replaced based on the id in the get shareable link.
##### Make sure the link to download has public access. Turn off the public access once downloaded

### **Using Tensorboad on AWS**
##### pip install tensorboard
##### run the tensorboard using 'tensorboard --logdir==runs"
##### Note the pytorch version has to be >=1.2


