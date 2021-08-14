from tensorflow.keras.metrics import Metric
import numpy as np
from tensorflow.python.ops.summary_ops_v2 import image


class PercentCorrectEvents(Metric):
    def __init__(self, name="PCE", **kwargs):
        super(PercentCorrectEvents, self).__init__(name=name, **kwargs)
        self.correct_events = self.add_weight(name="PCE", initializer="zeros")
    
    def update_state(self, targets, logits, sample_weight=None):
        """Percentage of Correct Events

        :param prediction_events: prediction of event spotting, size: (2,)
        :param target_events: target/ground-truth of event spotting, size: (2,)
        :return:
        """
        logits = np.array(logits, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        logits[logits >= 0.5] = 1.
        logits[logits < 0.5] = 0.
        targets[targets >= 0.5] = 1.
        targets[targets < 0.5] = 0.
        diffs = logits - targets
        
        for diff in diffs:
            # Check correct or not
            if sum(diff) != 0:  # Incorrect
                ret_pce = 0
                self.correct_events.assign_add(ret_pce)
            else:  # Correct
                ret_pce = 1
                self.correct_events.assign_add(ret_pce)

    def result(self):
        return self.correct_events

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.correct_events.assign(0.0)


class SmoothPercentCorrectEvents(Metric):
    def __init__(self, configs, name="SPCE", **kwargs):
        super(SmoothPercentCorrectEvents, self).__init__(name=name, **kwargs)
        self.scorrect_events = self.add_weight(name="SPCE", initializer="zeros")
        self.thresh = configs.threshold_spce
    
    def update_state(self, targets, logits, sample_weight=None):
        """Smooth Percentage of Correct Events

        :param prediction_events: prediction of event spotting, size: (2,)
        :param target_events: target/ground-truth of event spotting, size: (2,)
        :param thresh: the threshold for the difference between the prediction and ground-truth
        :return:
        """
        logits = np.array(logits, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        diff = np.abs(logits - targets)
        
        for dif in diff:
            if np.sum(dif > self.thresh) > 0:
                ret_spce = 0
                self.scorrect_events.assign_add(ret_spce)
            else:
                ret_spce = 1
                self.scorrect_events.assign_add(ret_spce)
    
    def result(self):
        return self.scorrect_events
    
    def reset_states(self):
        self.scorrect_events.assign(0.0)


class IntersectionOfUnion(Metric):
    def __init__(self, configs, name="IOU", **kwargs):
        super(IntersectionOfUnion, self).__init__(name=name, **kwargs)
        self.IoU = self.add_weight(name="IOU", initializer="zeros")
        self.smooth = configs.iou_smooth_rate
    
    def update_state(self, targets, logits, sample_weight=None):
        #flatten label and prediction tensors
        logits = np.array(logits).flatten()
        targets = np.array(logits).flatten()

        logit = logits.flatten()
        target = targets.flatten()

        intersection = np.sum(np.dot(target, logit))
        total = np.sum(target) + np.sum(logit)
        union = total - intersection
        
        IoU = 1 - ((intersection + self.smooth) / (union + self.smooth))
        self.IoU.assign(IoU)
    
    def result(self):
        return self.IoU

    def reset_states(self):
        self.IoU.assign(0.0)