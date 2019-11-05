import numpy as np 
import torch

class HyperParameters:
    run_id = 'run1'
    epochs = 100
    val_number_of_images = 10
    total_train_images = 130 * 500
    batch_size = 100
    learning_rate = 0.001
    batches = total_train_images // batch_size

hparams = HyperParameters()

col = Colorization(256)
opt_operations = training_pipeline(col, learning_rate, batch_size)
evaluations_ops = evaluation_pipeline(col, val_number_of_images)
summary_writer = metrics_system(hparams.run_id, sess)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(hparams.run_id)

