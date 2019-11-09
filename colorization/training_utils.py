import pickle
import time
from os.path import join

import matplotlib
import numpy as np
import torch
from skimage import color

matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (10.0, 4.0)
import matplotlib.pyplot as plt


# Avinash : May be uncomment later for viewing

# def plot_evaluation(res, folder_num, epoch):
#     maybe_create_folder(join(dir_root, 'images', folder_num))
#     for k in range(len(res['imgs_l'])):
#         img_gray = l_to_rgb(res['imgs_l'][k][:, :, 0])
#         img_output = lab_to_rgb(res['imgs_l'][k][:, :, 0],
#                                 res['imgs_ab'][k])
#         img_true = lab_to_rgb(res['imgs_l'][k][:, :, 0],
#                               res['imgs_true_ab'][k])
#         top_5 = np.argsort(res['imgs_emb'][k])[-5:]
#         try:
#             top_5 = ' / '.join(labels_to_categories[i] for i in top_5)
#         except:
#             ptop_5 = str(top_5)

#         plt.subplot(1, 3, 1)
#         plt.imshow(img_gray)
#         plt.title('Input (grayscale)')
#         plt.axis('off')
#         plt.subplot(1, 3, 2)
#         plt.imshow(img_output)
#         plt.title('Network output')
#         plt.axis('off')
#         plt.subplot(1, 3, 3)
#         plt.imshow(img_true)
#         plt.title('Target (original)')
#         plt.axis('off')
#         plt.suptitle(top_5, fontsize=7)

#         plt.savefig(join(
#             dir_root, 'images', run_id, '{}_{}.png'.format(epoch, k)))
#         plt.clf()
#         plt.close()

def l_to_rgb(img_l):
    """
    Convert a numpy array (l channel) into an rgb image
    :param img_l:
    :return:
    """
    lab = np.squeeze(255 * (img_l + 1) / 2)
    return color.gray2rgb(lab) / 255


def lab_to_rgb(img_l, img_ab):
    """
    Convert a pair of numpy arrays (l channel and ab channels) into an rgb image
    :param img_l:
    :return:
    """
    lab = np.empty([*img_l.shape[0:2], 3])
    lab[:, :, 0] = np.squeeze(((img_l + 1) * 50))
    lab[:, :, 1:] = img_ab * 127
    return color.lab2rgb(lab)
