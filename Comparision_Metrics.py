import skimage.measure as measure
import numpy as np
import skimage.io as io
from skimage.transform import resize
import os

filetypes = ['Baseline','Tanh','Skip','No_feature_extractor']

orig_img = [io.imread('Final_Outputs/'+'Original_'+str(i)+'.jpg') for i in range(1,4)]

                     
for i in range(3):
	orig_img[i] = 255*resize(orig_img[i],(224,224,3),anti_aliasing=True)
		
for x in filetypes:
	for i in range(1,4):
		recon_image = io.imread('Final_Outputs/'+x+'_'+str(i)+'.jpg')
		val = measure.compare_psnr(recon_image.astype(np.uint8),orig_img[i-1].astype(np.uint8))
		print(x,i,val)

