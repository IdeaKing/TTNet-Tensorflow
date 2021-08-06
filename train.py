# Create custom metrics...

import os

import tensorflow as tf

from models.ttnet import ttnet
from utils.utils import *
from utils.configs import configs
from utils.losses import *
from utils.data_utils import data_preparer, data_split
from utils.dataset import TTNetDataset
from utils.evaluation_utils import *


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

    width = configs.processed_image_shape[0]
    height = configs.processed_image_shape[1]

    ce_loss_fn = CrossEntropyTT(w=width, h=height)
    wce_loss_fn = WeightedCrossEntropyTT()
    segm_loss_fn = SegmDICEBCE()

    # Metrics
    mIOU = MeanIoU(num_classes=3)

    # Create the model here
    model_dims = (
        configs.processed_image_shape[0],
        configs.processed_image_shape[1],
        configs.num_frames_sequence * 3)

    model = ttnet(dims=model_dims) # 380 x 128 x number of frames
    for epoch in range(resume_from_checkpoint, epochs):
        for step, (x_images, y_ball_position, y_events, y_mask) in enumerate(
            train_data):
            # Training Step
            with tf.GradientTape() as tape:
                print(np.asarray(x_images).shape)
                detectionx_logits, detectiony_logits, events_logits, mask_logits = model(
                    x_images, training=True)
                
                print(y_ball_position)
                print(detectionx_logits)
                print(detectiony_logits)

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

        tb_callback.set_model(model)
        print(f"Epoch: {epoch} Step: {step}")

if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    # Get and prepare the data
    events_infor, events_labels = data_preparer(configs=configs)
    # Split the data in training and validation sets
    events_infor, events_labels, v_events_infor, v_events_labels = data_split(
        events_infor, events_labels, configs)
    # Instantiate the TTNetDataset Class
    ttnet_dataset_creator = TTNetDataset(
        events_infor=events_infor,
        org_size=configs.original_image_shape,
        input_size=configs.processed_image_shape,
        configs=configs)
    validation_dataset_creator = TTNetDataset(
        events_infor=events_infor,
        org_size=configs.original_image_shape,
        input_size=configs.processed_image_shape,
        configs=configs)
    # Create the training and validation datasets
    ttnet_dataset = ttnet_dataset_creator.get_dataset()
    validation_dataset = validation_dataset_creator.get_dataset()

    # Begin training the dataset
    train(
        train_data=ttnet_dataset,
        validation_data=validation_dataset,
        configs=configs)
    