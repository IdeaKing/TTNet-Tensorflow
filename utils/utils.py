# Created by Thomas Chia
# Large portions of code were based on 
# https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
# Paper by https://arxiv.org/pdf/2004.09927.pdf

import os

from tensorflow.keras.callbacks import (TensorBoard, 
                                        LearningRateScheduler)
from tensorflow.keras.optimizers import Adam
import numpy as np

from utils.configs import configs

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

tb_callback = TensorBoard(log_dir=tensorboard_logdir)


def printProgressBar(
    iter, 
    total, 
    run_type,
    epoch = '', 
    ce = '', 
    wce = '', 
    dicebce = '',):
    """Training Progress Bar"""
    decimals = 1
    length = 20
    fill = '█'
    printEnd = "\r"

    # Convert tensor values into just float values
    try:
        ce = str(ce.numpy().astype(np.float16))
        wce = str(wce.numpy().astype(np.float16))
        dicebce = str(dicebce.numpy().astype(np.float16))
    except:
        pass

    percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iter / float(total)))
    filledLength = int(length * iter // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print(f'\r {run_type} Epoch: {epoch} |{bar}| {percent}%' \
          f' {run_type} CE Loss: {ce}' \
          f' {run_type} Weighted CE Loss: {wce}' \
          f' {run_type} DICE-BCE Loss: {dicebce}             ', 
        end = '\r')

    # Print New Line on Complete
    if iter == total: 
        print()




