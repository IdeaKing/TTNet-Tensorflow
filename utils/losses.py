# Created by Thomas Chia
# Based on the TTNet Loss Functions
# DOC: 2021-08-02

from tensorflow.keras.backend import (epsilon)
from tensorflow.keras.losses import Loss
from tensorflow import Tensor, reshape, device
import numpy as np
from numpy import mean, log, sum, clip
from tensorflow.python.keras.backend import dtype

class CrossEntropyTT(Loss):
    """The ball detection cross entropy loss function."""
    def __init__(self, w, h):
        super(CrossEntropyTT, self).__init__()
        self.w = w
        self.h = h

    def call(self, pred_position, target_position, axis):
        pred_position = np.asarray(pred_position)
        target_position = np.asarray(target_position)
        if axis == "x":
            loss_ball = - sum(target_position*log(pred_position)) / self.w
            return loss_ball
        else:
            loss_ball = - sum(target_position*log(pred_position)) / self.h
            return loss_ball

class WeightedCrossEntropyTT(Loss):
    """The events spotting loss function."""
    def __init__(self, weight_ratio=np.array([1, 3], dtype=np.float32), number_events=2):
        super(WeightedCrossEntropyTT, self).__init__()
        self.weight_ratio = weight_ratio
        self.number_events = number_events

    def call(self, pred_events, target_events):
        target_events = np.asarray(target_events)
        pred_events = np.asarray(pred_events)

        loss = self.weight_ratio * - sum(self.number_events * log(pred_events + epsilon())) / self.number_events
        return loss


class SmoothDICE(Loss):
    def __init__(self):
        super(SmoothDICE, self).__init__()

    def call(self, pred_seg, target_seg):
        loss = 1. - ((sum(2 * pred_seg * target_seg) + epsilon) / 
            (sum(pred_seg) + sum(target_seg) + epsilon))
        return loss

class BinaryCrossEntropy(Loss):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    def call(self, pred_seg, target_seg):
        loss = - mean(target_seg * log(pred_seg + epsilon) + 
            (1 - target_seg) * log(1 - pred_seg + epsilon))
        return loss

class SegmDICEBCE(Loss):
    """Segmentation Loss Function."""
    def __init__(self, coefficient=1e-4):
        super(SegmDICEBCE, self).__init__()
        self.bce = BinaryCrossEntropy()
        self.dice = SmoothDICE()
        self.coefficient = coefficient
    
    def call(self, pred_seg, target_seg):
        target_seg = np.asarray(target_seg, dtype=np.float32)
        loss_bce = self.bce(pred_seg, target_seg)
        loss_dice = self.dice(pred_seg, target_seg)
        loss_seg = (1 - self.coefficient) * loss_dice + self.coefficient * loss_bce
        return loss_seg

