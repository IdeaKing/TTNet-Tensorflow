import os

import tensorflow as tf
import numpy as np
import cv2

from models.ttnet import ttnet
from utils.data_utils import data_preparer, ball_position
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
        print("Events GT: " + str(test_events) +
              " Events Prediction: " + str(events_logits) + 
              " PCE: " + str(PCE.result()) + "\n")
        print("IOU: " + str(IOU.result()) + "\n")

        # Save the results if required
        if save_results==2:
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
    """
    events_infor = np.asarray(events_infor, dtype=object)
    ball_position_x = events_infor[:, 1].tolist()
    ball_position_y = events_infor[:, 2].tolist()

    position_x_ds = tf.data.Dataset.from_tensor_slices(ball_position_x)
    position_y_ds = tf.data.Dataset.from_tensor_slices(ball_position_y)
    ds = tf.data.Dataset.zip((position_x_ds, position_y_ds))


    for pos_x, pos_y in ds.as_numpy_iterator():
        x, y = ball_position(pos_x, pos_y, configs=configs)
        print("full x", pos_x)
        print("full y", pos_y)

        print("shape x", len(pos_x))
        print("shape y", len(pos_y))
        # print("pos x", pos_x)
        # if x != 0:
        print("x", x)
        print("y", y)
        # print("pos_x", pos_x)
        print("the value x", pos_x[np.argmax(pos_x)])
        print("the value y", pos_y[np.argmax(pos_y)])
        break
            
    """
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

    """
    for step, (test_images, test_ball_pos_x, test_ball_pos_y, 
        test_events, test_mask) in enumerate(ttnet_dataset):

        pos_x = np.array(test_ball_pos_x)
        pos_y = np.array(test_ball_pos_y)

        x, y = ball_position(pos_x, pos_y, configs=configs)
        print("x", x)
        print("y", y)

        break

    """
    
    # Begin training the dataset
    print("Begining testing.")
    test(
        test_data=ttnet_dataset,
        test_size=events_infor,
        configs=configs)
