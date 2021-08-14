# Created by Thomas Chia
# Paper by https://arxiv.org/pdf/2004.09927.pdf

import os

from tensorflow.keras.callbacks import (TensorBoard, 
                                        LearningRateScheduler)
from tensorflow.keras.optimizers import Adam
import numpy as np

from utils.configs import configs


def checkpoints_cb(epoch, model, configs):
    """Checkpoint callback for training the model."""
    frequency = configs.checkpoint_frequency
    checkpoint_dir = os.path.join(
        configs.work_dir, 
        "checkpoints",
        f"ttnet-{epoch+1}.ckpt")

    if (epoch % frequency) == 0:
        model.save_weights(checkpoint_dir)
        print()
        print("Checkpoints saved to: ", checkpoint_dir)
    

def scheduler(epoch, lr=1e-3):
    halving_rate = int(epoch/3) + 1
    lr=lr/2**halving_rate
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

adam_optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8)

tensorboard_logdir = os.path.join(configs.work_dir, "logging")

if os.path.isdir(tensorboard_logdir):
    pass
else:
    os.makedirs(tensorboard_logdir)

tensorboard_cb = TensorBoard(log_dir=tensorboard_logdir)


def printProgressBar(
    iter, 
    total, 
    run_type,
    epoch = '', 
    rmse = '',
    pce = '', 
    spce = '', 
    iou = '',):
    """Training Progress Bar"""
    decimals = 1
    length = 20
    fill = '█'
    printEnd = "\r"

    # Convert tensor values into just float values
    try:
        pce = str(pce.numpy().astype(np.float16))
        spce = str(spce.numpy().astype(np.float16))
        rmse = str(rmse.numpy().astype(np.float16))
        iou = str(iou.numpy().astype(np.float16))
    except:
        pass

    percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iter / float(total)))
    filledLength = int(length * iter // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print(f'\r{run_type} Epoch: {epoch} |{bar}| {percent}%' \
          f' {run_type} RMSE {rmse}' \
          f' {run_type} PCE: {pce}' \
          f' {run_type} SPCE: {spce}' \
          f' {run_type} IOU: {iou}             ', 
        end = '\r')

    # Print New Line on Complete
    if iter == total: 
        print()




