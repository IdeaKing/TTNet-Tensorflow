# Created by Thomas Chia
# Command for training: python train_ball_loc.py --work-dir "test-ball-loc-001"

import os

import tensorflow as tf
import numpy as np

from models.ttnet import ttnet
from utils.utils import *
from utils.configs import configs
from utils.losses import *
from utils.data_utils import data_preparer, data_split
from utils.dataset import TTNetDataset
from utils.metrics import *


def train(train_data, validation_data, t_size, v_size, configs=configs):
    """The training loop for TTNet."""
    # Common Model Parameters
    epochs = configs.num_epochs
    batch_size = configs.batch_size
    resume_from_checkpoint = configs.resume_training
    width = configs.processed_image_shape[0]
    height = configs.processed_image_shape[1]
    step_size = int(t_size.shape[0]/batch_size)
    val_step_size = int((v_size.shape[0]/configs.num_frames_sequence)/batch_size)
    print("val_step_size", val_step_size)
    print("v size", v_size.shape)

    # Initialize Losses
    ce_loss_fn = CrossEntropyTT(w=width, h=height)

    # Initialize Metrics
    RMSE = tf.keras.metrics.RootMeanSquaredError(name="RMSE")

    # Create the model here
    model_dims = (
        width,
        height,
        configs.num_frames_sequence * 3)
    model = ttnet(dims=model_dims, ball_detection_stage=True) # 380 x 128 x (number of frames x 3)

    # Load from checkpoints if needed
    if resume_from_checkpoint != 0:
        checkpoint_dir = os.path.join(
            configs.work_dir, 
            "checkpoints", 
            f"ttnet-{resume_from_checkpoint}.ckpt")
        model.load_weights(checkpoint_dir)

    # Multiloss training loop
    # https://stackoverflow.com/questions/59690188/
    # https://www.youtube.com/watch?v=KrS94hG4VU0
    for epoch in range(resume_from_checkpoint, epochs):
        for step, (x_images, y_ball_position_x, 
            y_ball_position_y, y_events, y_mask) in enumerate(train_data):
            # Training Step
            with tf.GradientTape() as tape:
                # Train the model
                (detection_x_logits, detection_y_logits) = model(
                        x_images, training=True)

                # print("logits", detection_y_logits)
                # print("gt", y_ball_position_y)
                # Ball detection losses
                ce_loss = ce_loss_fn.call(
                    y_ball_position_x, 
                    detection_x_logits,
                    y_ball_position_y, 
                    detection_y_logits)


            # Update Gradients
            grads = tape.gradient(
                [ce_loss], model.trainable_variables)
            adam_optimizer.apply_gradients(
                zip(grads, model.trainable_variables))

            # Training Metric Update
            RMSE.update_state(
                y_ball_position_x, detection_x_logits)
            RMSE.update_state(
                y_ball_position_y, detection_y_logits)

            # Update the training progess bar
            printProgressBar(
                iter=step,
                total=step_size,
                run_type="Train",
                epoch=epoch+1,
                rmse=RMSE.result(),
                pce=ce_loss)

        # Reset the training metrics
        RMSE.reset_states()

        checkpoints_cb(epoch=epoch, model=model, configs=configs)

        # Validation Step
        for step, (val_images, val_ball_position_x, 
            val_ball_position_y, val_events, val_mask) in enumerate(
            validation_data):
            # Run the Model
            (val_det_x_logits, val_det_y_logits) = model(
                    val_images, training=True)
            # Ball detection losses
            ce_loss = ce_loss_fn.call(
                val_ball_position_x, 
                val_det_x_logits,
                val_ball_position_y, 
                val_det_y_logits)

            # Validation metric update
            RMSE.update_state(
                y_ball_position_x, detection_x_logits)
            RMSE.update_state(
                y_ball_position_y, detection_y_logits)

            # Update the validation progess bar
            printProgressBar(
                iter=step,
                total=val_step_size,
                run_type="Validation",
                epoch=epoch+1,
                rmse=RMSE.result(),
                pce=ce_loss)
            

        # Reset the validation metrics
        RMSE.reset_states()

        # Apply the callbacks
        tensorboard_cb.set_model(model)
                
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
        t_size=events_infor,
        v_size=v_events_infor,
        configs=configs)
    