
#****************************************#
#***          Import Modules          ***#
#****************************************#
import os
import time
import numpy as np 
import torch
import torch.nn as nn
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
    batch_size = 100
    learning_rate = 0.001

    val_number_of_images = 10
    total_train_images = 130 * 500
    batches = int(total_train_images/batch_size)

config = Configuration()
hparams = HyperParameters()


#****************************************#
#***       Architecture Pipeline      ***#
#****************************************#
model = Colorization(256)  
# model.encoder.apply(init_weights)
# model.decoder.apply(init_weights)
# model.fusion.apply(init_weights)
loss_criterion = torch.nn.MSELoss(reduction='mean').to(config.device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=1e-6)


#****************************************#
#***  Training & Validation Pipeline  ***#
#****************************************#

# To Do : Bhumi: Load the data from a class
#train_dataset = 
#val_dataset = 

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=hparams.batch_size)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=hparams.batch_size)


evaluations_ops = evaluation_pipeline(col, hparams.val_number_of_images)


for epoch in range(epochs):
    print('Starting epoch:',epoch, 'total images', hparams.total_train_images)
    # Training step
    loop_start = time.time()
    for batch in range(batches):

        optimizer.zero_grad()

        
        feature, outputs = model(feats,evalMode=False)

        loss = criterion(outputs, labels.long())
        loss.backward()
        
        scheduler.step()
        optimizer.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_closs.parameters():
            param.grad.data *= (1. / hp.closs_weight)
        optimizer_closs.step()
        
        avg_loss += loss.item()
        if batch%config.point_batches==0: 
            loop_end = time.time()   
            print('Batch:' batch ,'of', batches, '| Processing time for',config.point_batches,'batches:',loop_end-loop_start)
            loop_start = time.time()

        # res = sess.run(opt_operations)
        # global_step = res['global_step']
        # print_log('Cost: {} Global step: {}'
        #           .format(res['cost'], global_step), run_id)
        # summary_writer.add_summary(res['summary'], global_step)


    # Save the variables to disk
    checkpoint = {'model': model,'model_state_dict': model.state_dict(),\
                  'optimizer' : optimizer,'optimizer_state_dict' : optimizer.state_dict(),\
                  'train_loss':train_loss, 'train_acc':train_acc,
                  'val_loss':val_loss, 'val_acc':val_acc}
    torch.save(checkpoint, config.model_file_name)
    print("Model saved at:",os.getcwd(),'/',config.model_file_name)
    print('')


    # Evaluation step on validation
    res = sess.run(evaluations_ops)
    summary_writer.add_summary(res['summary'], global_step)
    plot_evaluation(res, run_id, epoch)