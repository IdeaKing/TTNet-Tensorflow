# Created by Thomas Chia
# Based on the TTNet Loss Functions
# DOC: 2021-08-02

import numpy as np
# from numpy import log, sum
from tensorflow import convert_to_tensor, cast, float32
from tensorflow.keras.losses import Loss, binary_crossentropy
from tensorflow.keras.backend import log, sum, flatten

class CrossEntropyTT(Loss):
    """The ball detection cross entropy loss function."""
    def __init__(self, w, h):
        super(CrossEntropyTT, self).__init__()
        self.w = w
        self.h = h

    def call(self, pred_position, target_position, axis):
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
        target_events = cast(target_events, dtype=float32)
        loss = self.weight_ratio * - sum(target_events * log(pred_events)) / self.number_events
        return loss


class SmoothDICE(Loss):
    def __init__(self):
        super(SmoothDICE, self).__init__()

    def call(self, pred_seg, target_seg):
        loss = sum(2 * pred_seg * target_seg) / sum(pred_seg) + sum(target_seg)
        return loss

class DiceBce(Loss):
    """Segmentation Loss Function.
    https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    """
    def __init__(self):
        super(DiceBce, self).__init__()
    
    def call(init, inputs, targets):
        targets = convert_to_tensor(targets, dtype=float)
        smooth = 1e-6
        # Flatten label and prediction tensors
        inputs = flatten(inputs)
        targets = flatten(targets)
        # Calculate the loss
        BCE =  binary_crossentropy(targets, inputs)
        intersection = sum(targets * inputs)
        dice_loss = 1 - (2*intersection + smooth) / (sum(targets) + sum(inputs) + smooth)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
def SegmDICEBCE(inputs, targets, smooth=1e-6):
    targets = convert_to_tensor(targets, dtype=float)
    #flatten label and prediction tensors
    inputs = flatten(inputs)
    targets = flatten(targets)
    
    BCE =  binary_crossentropy(targets, inputs)
    intersection = sum(targets * inputs)
    dice_loss = 1 - (2*intersection + smooth) / (sum(targets) + sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE

