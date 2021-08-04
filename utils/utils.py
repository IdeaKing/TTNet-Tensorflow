# Created by Thomas Chia
# Large portions of code were based on https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
# Paper by https://arxiv.org/pdf/2004.09927.pdf

from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam

def scheduler(epoch, lr=1e-3):
    halving_rate = int(epoch/3) + 1
    lr=lr/2**halving_rate
    return lr

lr_scheduler = callbacks.LearningRateScheduler(scheduler)

optimizer = Adam(
    learning_rate=lr_scheduler,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8)





