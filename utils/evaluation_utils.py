import numpy as np
from tensorflow.keras.metrics import MeanIoU


def PCE(prediction_events, target_events):
    """Percentage of Correct Events

    :param prediction_events: prediction of event spotting, size: (2,)
    :param target_events: target/ground-truth of event spotting, size: (2,)
    :return:
    """
    prediction_events[prediction_events >= 0.5] = 1.
    prediction_events[prediction_events < 0.5] = 0.
    target_events[target_events >= 0.5] = 1.
    target_events[target_events < 0.5] = 0.
    diff = prediction_events - target_events
    # Check correct or not
    if np.sum(diff) != 0:  # Incorrect
        ret_pce = 0
    else:  # Correct
        ret_pce = 1
    return ret_pce


def SPCE(prediction_events, target_events, thresh=0.25):
    """Smooth Percentage of Correct Events

    :param prediction_events: prediction of event spotting, size: (2,)
    :param target_events: target/ground-truth of event spotting, size: (2,)
    :param thresh: the threshold for the difference between the prediction and ground-truth
    :return:
    """
    diff = np.abs(prediction_events - target_events)
    if np.sum(diff > thresh) > 0:
        ret_spce = 0
    else:
        ret_spce = 1
    return ret_spce