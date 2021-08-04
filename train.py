# Create custom metrics...

from models.ttnet import ttnet
from utils.configs import configs
from utils.losses import *
from utils.dataset import Dataset
from utils.evaluation_utils import *
from utils.utils import *

import tensorflow as tf

def train(train_data, validation_data, configs=configs):
    epochs = configs.num_epochs
    batch_size = configs.batch_size
    resume_from_checkpoint = configs.resume_training

    if resume_from_checkpoint > 0:
        checkpoint_dir = os.path.join(
            configs.work_dir, "checkpoints", "ttnet.ckpt")
    
    # Multiloss training loop
    # https://stackoverflow.com/questions/59690188/
    # https://www.youtube.com/watch?v=KrS94hG4VU0
    ce_loss_fn = CrossEntropyTT(w=configs.width, h=configs.height) # Needs WH
    wce_loss_fn = WeightedCrossEntropyTT()
    segm_loss_fn = SegmDICEBCE()
    adam_optimizer = configs.optimizer

    # Metrics
    mIOU = MeanIoU(num_classes=3)

    # Create the model here
    model = ttnet(dims=configs.image_size) # 380 x 240 x number of frames
    for epoch in range(resume_from_checkpoint, epochs):
        for step, (x_images, y_ball_position, y_events, y_mask) in enumerate(
            train_data):
            # Training Step
            with tf.GradientTape() as tape:
                detection_logits, events_logits, mask_logits = model(
                    x_images, training=True)
                
                # Ball detection losses
                ce_loss = ce_loss_fn.call(detection_logits, y_ball_position)
                # Event detection losses
                wce_loss = wce_loss_fn.call(events_logits, y_events)
                # Mask segmentation losses
                segm_loss = segm_loss_fn.call(mask_logits, y_mask)
                # Average all three losses
                avg_loss = (ce_loss + wce_loss + segm_loss)/3 

            # Update Gradients
            grads = tape.gradient(
                [ce_loss, wce_loss, segm_loss, avg_loss], model.trainable_weights)
            adam_optimizer.apply_gradients(
                zip(grads, model.trainable_weights))

            # Training Metric Update    
            training_iou_metric = mIOU.update_state(y_true=y_mask, y_pred=mask_logits)

        # Validation Step
        for step, (val_images, val_ball_position, val_events, val_mask) in enumerate(
            validation_data):
            val_detection_logits, val_events_logits, val_mask_logits = model(
                    val_images, training=True)
            # Ball detection losses
            val_ce_loss = ce_loss_fn.call(val_detection_logits, val_ball_position)
            # Event detection losses
            val_wce_loss = wce_loss_fn.call(val_events_logits, val_events)
            # Mask segmentation losses
            val_segm_loss = segm_loss_fn.call(val_mask_logits, val_mask)
            # Average all three losses
            val_avg_loss = (ce_loss + wce_loss + segm_loss)/3             


        print(f"Epoch: {epoch} Step: {step} Training RMSE: {training_rmse_metric} \
                Training SPCE: {training_spce_metric} Training IOU: {training_iou_metric} \
                Validation RMSE: {val_rmse_metric} Validation SPCE: {val_spce_metric} \
                Validation IOU: {val_iou_metric}")


    




            



