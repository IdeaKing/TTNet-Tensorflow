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


def train(train_data, validation_data, t_events_infor, configs=configs):
    # Common Model Parameters
    epochs = configs.num_epochs
    batch_size = configs.batch_size
    resume_from_checkpoint = configs.resume_training
    width = configs.processed_image_shape[0]
    height = configs.processed_image_shape[1]
    step_size = t_events_infor.shape[0]

    if resume_from_checkpoint != 0:
        checkpoint_dir = os.path.join(
            configs.work_dir, "checkpoints", "ttnet.ckpt")

    # Initialize Losses
    ce_loss_fn = CrossEntropyTT(w=width, h=height)
    wce_loss_fn = WeightedCrossEntropyTT()

    # Initialize Metrics
    mIOU = MeanIoU(num_classes=3)

    # Create the model here
    model_dims = (
        configs.processed_image_shape[0],
        configs.processed_image_shape[1],
        configs.num_frames_sequence * 3)
    model = ttnet(dims=model_dims) # 380 x 128 x (number of frames x 3)
    
    # Multiloss training loop
    # https://stackoverflow.com/questions/59690188/
    # https://www.youtube.com/watch?v=KrS94hG4VU0
    for epoch in range(resume_from_checkpoint, epochs):
        # Initiate the training progress bar
        printProgressBar(
            iter=0, 
            total=step_size,
            run_type="Train")
        for step, (x_images, y_ball_position_x, 
            y_ball_position_y, y_events, y_mask) in enumerate(train_data):
            # Training Step
            with tf.GradientTape() as tape:
                detection_x_logits, detection_y_logits, events_logits, mask_logits = model(
                    x_images, training=True)
                # Ball detection losses
                ce_loss_x = ce_loss_fn.call(
                    detection_x_logits, y_ball_position_x, axis = "x")
                ce_loss_y = ce_loss_fn.call(
                    detection_y_logits, y_ball_position_y, axis = "y")
                ce_loss = ce_loss_x + ce_loss_y
                # Event detection losses
                wce_loss = wce_loss_fn.call(events_logits, y_events)
                # Mask segmentation losses
                segm_loss = SegmDICEBCE(mask_logits, y_mask)

            # Update Gradients
            grads = tape.gradient(
                [ce_loss, wce_loss, segm_loss], model.trainable_variables)
            adam_optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
            # Update the training progess bar
            printProgressBar(
                iter=step,
                total=step_size,
                run_type="Train",
                epoch=epoch+1,
                ce=ce_loss,
                wce=wce_loss,
                dicebce=segm_loss)
            # Training Metric Update    
            # training_iou_metric = mIOU.update_state(y_true=y_mask, y_pred=mask_logits)

        # Validation Step
        for step, (val_images, val_ball_position_x, val_ball_position_y, val_events, val_mask) in enumerate(
            validation_data):
            val_det_x_logits, val_det_y_logits, val_events_logits, val_events_logits, val_mask_logits = model(
                    val_images, training=True)
            # Ball detection losses
            ce_loss_x = ce_loss_fn.call(
                val_det_x_logits, val_ball_position_x, axis = "x")
            ce_loss_y = ce_loss_fn.call(
                val_det_y_logits, val_ball_position_y, axis = "y")
            ce_loss = ce_loss_x + ce_loss_y
            # Event detection losses
            wce_loss = wce_loss_fn.call(val_events_logits, val_events)
            # Mask segmentation losses
            segm_loss = SegmDICEBCE(val_mask_logits, val_mask)    

        tb_callback.set_model(model)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    # Get and prepare the data
    events_infor, events_labels = data_preparer(configs=configs)
    # Split the data in training and validation sets
    print("Splitting the dataset into train and validation Sets.")
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
    print("Creating the training dataset.")
    ttnet_dataset = ttnet_dataset_creator.get_dataset()
    print("Creating the validation dataset.")
    validation_dataset = validation_dataset_creator.get_dataset()

    # Begin training the dataset
    print("Begining training.")
    train(
        train_data=ttnet_dataset,
        validation_data=validation_dataset,
        t_events_infor=events_infor,
        configs=configs)
    