# Created by Thomas Chia
# Large portions of code were based on 
# https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
# Paper by https://arxiv.org/pdf/2004.09927.pdf

import os

from tensorflow.keras.callbacks import (TensorBoard, LearningRateScheduler)
from tensorflow.keras.optimizers import Adam

from utils.configs import configs

def scheduler(epoch, lr=1e-3):
    halving_rate = int(epoch/3) + 1
    lr=lr/2**halving_rate
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

adam_optimizer = Adam(
    learning_rate=lr_scheduler,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8)

tensorboard_logdir = os.path.join(configs.work_dir, "logging")

if os.path.isdir(tensorboard_logdir):
    pass
else:
    os.makedirs(tensorboard_logdir)

tb_callback = TensorBoard(log_dir=tensorboard_logdir)




