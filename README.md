# PyTorch Implementation of Deep Colorization
#### Our Implementation is derived from the Original Paper 'Deep Koalarization: Image Colorization usingCNNs and Inception-Resnet-v2': https://arxiv.org/pdf/1712.03400.pdf
#### This approach has been tested on Coco-dataset with 70000 train images, 5000 validation images and 10000 test images

### **The Coco dataset can be downloaded on AWS or Google Colab using the below command line arguments:**
>wget -N images.cocodataset.org/zips/train2017.zip<br />
>wget -N images.cocodataset.org/zips/val2017.zip<br />
>wget -N images.cocodataset.org/zips/test2017.zip<br />


### Final Results
![picture](Final_Results.png)


### **Using Tensorboad on AWS**
> pip install tensorboard <br />
> run the tensorboard using 'tensorboard --logdir==runs" <br />
> Note the pytorch version has to be >=1.2 <br />


