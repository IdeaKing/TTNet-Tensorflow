# By thomas chia

# Command to run: python test_ball_loc.py --work-dir "test-ball-loc-001"
import os

import tensorflow as tf
import numpy as np
import cv2

from models.ttnet import ttnet
from utils.data_utils import data_preparer, ball_position
from utils.configs import configs
from utils.dataset import TTNetDataset

def test(test_data, test_size, configs=configs):
    """Test the model on testing dataset."""
    # Common model parameters
    width = configs.processed_image_shape[0]
    height = configs.processed_image_shape[1]
    frame_sequence = configs.num_frames_sequence
    testing_epoch = configs.testing_epoch
    save_results = configs.save_outputs

    # Initialize Metrics
    RMSE = tf.keras.metrics.RootMeanSquaredError(name="RMSE")

    # Create the model
    model_dims = (
        width,
        height,
        frame_sequence * 3)
    model = ttnet(dims=model_dims, ball_detection_stage=True)

    # Load the checkpoints
    checkpoint_dir = os.path.join(
        configs.work_dir, 
        "checkpoints", 
        f"ttnet-{testing_epoch}.ckpt")
    model.load_weights(checkpoint_dir)

    for step, (test_images, test_ball_pos_x, test_ball_pos_y, 
        test_events, test_mask) in enumerate(test_data):
        # Test the model on dataset
        (detection_x_logits, detection_y_logits) = model(
                test_images, training=False)
        
        # Training Metric Update
        RMSE.update_state(
            test_ball_pos_x, detection_x_logits)
        RMSE.update_state(
            test_ball_pos_y, detection_y_logits)

        # Update ball positions
        print("gt")
        test_x, test_y = ball_position(
            x_logits=test_ball_pos_x,
            y_logits=test_ball_pos_y,
            configs=configs)
        print("pred")
        pred_x, pred_y = ball_position(
            x_logits=detection_x_logits,
            y_logits=detection_y_logits,
            configs=configs)
        
        # Print out statistics between true and false.
        print(f"-------------------------Step: {step}-------------------------")
        print("Ball Position GT: (" + 
              str(test_x) + ", " + str(test_y) +
              ") Ball Position Prediction: (" + 
              str(pred_x) + ", " + str(pred_y) + 
              ") RMSE: " + str(RMSE.result())+ "\n")
        

        #print("pred_x", detection_x_logits)

        #print("actual_x", test_ball_pos_x)

        # Save the results if required
        if save_results==2:
            # Convert to numpy arrays
            det_x = detection_x_logits.numpy()
            det_y = detection_y_logits.numpy()

            # f = open(fp + ".txt", "w")
            # f.write("")
            # More to do here
            # f.close()

        # Reset the metrics
        RMSE.reset_states()

        #break

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
