import os

import tensorflow as tf
import numpy as np
import cv2

from models.ttnet import ttnet
from utils.data_utils import data_preparer
from utils.configs import configs
from utils.dataset import TTNetDataset
from utils.metrics import (PercentCorrectEvents,
                           SmoothPercentCorrectEvents,
                           IntersectionOfUnion)

def test(test_data, test_size, configs=configs):
    """Test the model on testing dataset."""
    # Common model parameters
    width = configs.processed_image_shape[0]
    height = configs.processed_image_shape[1]
    frame_sequence = configs.num_frames_sequence
    testing_epoch = configs.testing_epoch
    save_results = configs.save_outputs
    test_step_size = int(np.array(test_size).shape[0])

    # Initialize Metrics
    RMSE = tf.keras.metrics.RootMeanSquaredError(name="RMSE")
    PCE = PercentCorrectEvents()
    SPCE = SmoothPercentCorrectEvents(configs=configs)
    IOU = IntersectionOfUnion(configs=configs)

    # Create the model
    model_dims = (
        width,
        height,
        frame_sequence * 3)
    model = ttnet(dims=model_dims)

    # Load the checkpoints
    checkpoint_dir = os.path.join(
        configs.work_dir, 
        "checkpoints", 
        f"ttnet-{testing_epoch}.ckpt")
    model.load_weights(checkpoint_dir)

    for step, (test_images, test_ball_pos_x, test_ball_pos_y, 
        test_events, test_mask) in enumerate(test_data):
        # Test the model on dataset
        (detection_x_logits, detection_y_logits, 
            events_logits, mask_logits) = model(
                test_images, training=False)
        
        # Training Metric Update
        RMSE.update_state(
            test_ball_pos_x, detection_x_logits)
        RMSE.update_state(
            test_ball_pos_y, detection_y_logits)
        PCE.update_state(
            test_events, events_logits)
        SPCE.update_state(
            test_events, events_logits)
        IOU.update_state(
            test_mask, mask_logits)

        # Print out statistics between true and false.
        print(f"-------------------------Step: {step}-------------------------")
        print("Ball Position GT: (" + 
              str(test_ball_pos_x) + ", " + str(test_ball_pos_y) +
              ") Ball Position Prediction: (" + 
              str(detection_x_logits) + ", " + str(detection_y_logits) + 
              ") RMSE: " + str(RMSE.result())+ "\n")
        print("Events GT: " + str(test_events) +
              " Events Prediction: " + str(events_logits) + 
              " PCE: " + str(PCE.result()) + "\n")
        print("IOU: " + str(IOU.result()) + "\n")

        # Save the results if required
        if save_results:
            # Convert to numpy arrays
            det_x = detection_x_logits.numpy()
            det_y = detection_y_logits.numpy()
            events = events_logits.numpy()
            mask = mask_logits.numpy() 

            # Create and/or find the save directory
            log_dir = os.path.join(
                configs.work_dir,
                "testing")
            
            if os.path.isdir(log_dir):
                pass
            else:
                os.makedirs(log_dir)
            fn = f"image-{str(step).zfill(5)}"
            fp = os.path.join(log_dir, fn)
            f = open(fp + ".txt", "w")
            # More to do here
            f.close()

            cv2.imwrite(fp + ".jpg", mask)
        # Reset the metrics
        RMSE.reset_states()
        PCE.reset_states()
        SPCE.reset_states()
        IOU.reset_states()


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    # Get and prepare the data
    events_infor, events_labels = data_preparer(
        configs=configs,
        dataset_type="testing")
    # Instantiate the TTNetDataset Class
    testing_dataset_creator = TTNetDataset(
        events_infor=events_infor,
        org_size=configs.original_image_shape,
        input_size=configs.processed_image_shape,
        configs=configs,
        dataset_type="testing")
    # Create the testing dataset
    print("Creating the testing dataset.")
    ttnet_dataset = testing_dataset_creator.get_dataset()

    # Begin training the dataset
    print("Begining testing.")
    test(
        test_data=ttnet_dataset,
        test_size=events_infor,
        configs=configs)
