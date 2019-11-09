import numpy as np 
import torch
from colorization import Colorization

class HyperParameters:
    run_id = 'run1'
    epochs = 1
    val_number_of_images = 10
    total_train_images = 130 * 500
    batch_size = 100
    learning_rate = 0.001
    batches = int(total_train_images/batch_size)

hparams = HyperParameters()

col = Colorization(256)  # Class from network definition
opt_operations = training_pipeline(col, lhparams.earning_rate, hparams.batch_size)
evaluations_ops = evaluation_pipeline(col, hparams.val_number_of_images)
# summary_writer = metrics_system(hparams.run_id, sess)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(hparams.run_id)

for epoch in range(epochs):
    print('Starting epoch:',epoch, 'total images', total_train_images)
    # Training step
    for batch in range(batches):
        print('Batch:' batch ,'of', batches)
        # res = sess.run(opt_operations)
        # global_step = res['global_step']
        # print_log('Cost: {} Global step: {}'
        #           .format(res['cost'], global_step), run_id)
        # summary_writer.add_summary(res['summary'], global_step)
        opt_operations
        evaluations_ops

    # Save the variables to disk
    checkpoint = {'model': model,'model_state_dict': model.state_dict(),'optimizer' : optimizer,'optimizer_state_dict' : optimizer.state_dict()}
    print("Model saved in:",save_path)


    # Evaluation step on validation
    res = sess.run(evaluations_ops)
    summary_writer.add_summary(res['summary'], global_step)
    plot_evaluation(res, run_id, epoch)