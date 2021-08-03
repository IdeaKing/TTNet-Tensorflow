# Created by Thomas Chia
# Based on the TTNet Loss Functions
# DOC: 2021-08-02

from tensorflow.keras.backend import (mean, log, epsilon, sum)
from tensorflow.keras.losses import Loss
from tensorflow import Tensor, reshape, device

class CrossEntropyTT(Loss):
    """The ball detection cross entropy loss function."""
    def __init__(self, w, h):
        super(CrossEntropyTT, self).__init__()
        self.w = w
        self.h = h

    def call(self, pred_position, target_position):
        x_pred = pred_position[:, :self.w]
        y_pred = pred_position[:, self.w:]

        x_target = target_position[:, :self.w]
        y_target = target_position[:, self.w:]

        loss_ball_x = - mean(x_target * log(x_pred + epsilon) + 
            (1 - x_target) * log(1 - x_pred + epsilon))
        loss_ball_y = - mean(y_target * log(y_pred + epsilon) + 
            (1 - y_target) * log(1 - y_pred + epsilon))

        return loss_ball_x + loss_ball_y

class WeightedCrossEntropyTT(Loss):
    """The events spotting loss function."""
    def __init__(self, weight_ratio=(1,3), number_events=2):
        super(WeightedCrossEntropyTT, self).__init__()
        self.weight_ratio = weight_ratio
        self.number_events = number_events

    def call(self, pred_events, target_events):
        self.weight_ratio = reshape(Tensor(self.weight_ratio), [1, 2])
        with device("/GPU:0"):
            loss = mean(self.weights * (target_events * log(pred_events + epsilon) + 
                (1. - target_events) * log(1 - pred_events + epsilon)))
        return loss

class SmoothDICE(Loss):
    def __init__(self):
        super(SmoothDICE, self).__init__()

    def forward(self, pred_seg, target_seg):
        loss = 1. - ((sum(2 * pred_seg * target_seg) + epsilon) / 
            (sum(pred_seg) + sum(target_seg) + epsilon))
        return loss

class BinaryCrossEntropy(Loss):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    def forward(self, pred_seg, target_seg):
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
        target_seg = target_seg.float()
        loss_bce = self.bce(pred_seg, target_seg)
        loss_dice = self.dice(pred_seg, target_seg)
        loss_seg = (1 - self.coefficient) * loss_dice + self.coefficient * loss_bce
        return loss_seg

