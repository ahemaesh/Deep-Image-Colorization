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

# Load Data here
'''
from dataset.shared import dir_tfrecord, dir_metrics, dir_checkpoints, dir_root, \
    maybe_create_folder
from dataset.tfrecords import LabImageRecordReader

labels_to_categories = pickle.load(
    open(join(dir_root, 'imagenet1000_clsid_to_human.pkl'), 'rb'))
'''

def training_pipeline(col, learning_rate, batch_size):
    # Set up training (input queues, graph, optimizer)
    irr = LabImageRecordReader('lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(batch_size, shuffle=True)
    # read_batched_examples = irr.read_one()
    imgs_l = read_batched_examples['image_l']
    imgs_true_ab = read_batched_examples['image_ab']
    imgs_emb = read_batched_examples['image_embedding']
    imgs_ab = col.build(imgs_l, imgs_emb)
    cost, summary = loss_with_metrics(imgs_ab, imgs_true_ab, 'training')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        cost, global_step=global_step)
    return {
        'global_step': global_step,
        'optimizer': optimizer,
        'cost': cost,
        'summary': summary
    }#, irr, read_batched_examples


def evaluation_pipeline(col, number_of_images):
    # Set up validation (input queues, graph)
    irr = LabImageRecordReader('val_lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(number_of_images, shuffle=False)
    imgs_l_val = read_batched_examples['image_l']
    imgs_true_ab_val = read_batched_examples['image_ab']
    imgs_emb_val = read_batched_examples['image_embedding']
    imgs_ab_val = col.build(imgs_l_val, imgs_emb_val)
    cost, summary = loss_with_metrics(imgs_ab_val, imgs_true_ab_val,
                                      'validation')
    return {
        'imgs_l': imgs_l_val,
        'imgs_ab': imgs_ab_val,
        'imgs_true_ab': imgs_true_ab_val,
        'imgs_emb': imgs_emb_val,
        'cost': cost,
        'summary': summary
    }


def metrics_system(run_id, sess):
    # Merge all the summaries and set up the writers
    train_writer = tf.summary.FileWriter(join(dir_metrics, run_id), sess.graph)
    return train_writer

def plot_evaluation(res, run_id, epoch):
    maybe_create_folder(join(dir_root, 'images', run_id))
    for k in range(len(res['imgs_l'])):
        img_gray = l_to_rgb(res['imgs_l'][k][:, :, 0])
        img_output = lab_to_rgb(res['imgs_l'][k][:, :, 0],
                                res['imgs_ab'][k])
        img_true = lab_to_rgb(res['imgs_l'][k][:, :, 0],
                              res['imgs_true_ab'][k])
        top_5 = np.argsort(res['imgs_emb'][k])[-5:]
        try:
            top_5 = ' / '.join(labels_to_categories[i] for i in top_5)
        except:
            ptop_5 = str(top_5)

        plt.subplot(1, 3, 1)
        plt.imshow(img_gray)
        plt.title('Input (grayscale)')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img_output)
        plt.title('Network output')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(img_true)
        plt.title('Target (original)')
        plt.axis('off')
        plt.suptitle(top_5, fontsize=7)

        plt.savefig(join(
            dir_root, 'images', run_id, '{}_{}.png'.format(epoch, k)))
        plt.clf()
        plt.close()