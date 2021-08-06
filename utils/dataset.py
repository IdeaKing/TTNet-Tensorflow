# Created by Thomas Chis
# 2021-08-05
# TTNet-Tensorflow

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import data


class TTNetDataset():
    def __init__(self, events_infor, org_size, input_size, configs):
        self.events_infor = events_infor
        self.w_org = org_size[0]
        self.h_org = org_size[1]
        self.w_input = input_size[0]
        self.h_input = input_size[1]
        self.w_resize_ratio = self.w_org / self.w_input
        self.h_resize_ratio = self.h_org / self.h_input
        self.configs = configs

    def parse_images(self, images: np.ndarray):
        """Open and perform operations on all images.
        Parameters:
            images (np.ndarray): Array of image filepaths
        Returns:
            image_stack (np.array): Stack of processed images
        """
        # Processing if the image is a group of images.
        image_stack = []
        for image_path in images:
            image_path = tf.compat.as_str_any(image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.w_input, self.h_input))
            image_stack.append(image)
        image_stack = np.dstack(image_stack)
        return image_stack

    def parse_masks(self, mask_path: np.ndarray):
        """Open and perform operations on the masks."""
        mask = cv2.imread(tf.compat.as_str_any(mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = np.asarray(mask).astype(np.int8)
        return mask

    def coordinate_adjustment(self, ball_position: np.ndarray):
        """Change the position coordinates of the ball to scale to training."""
        ball_position[0] = ball_position[0]/self.w_resize_ratio
        ball_position[1] = ball_position[1]/self.h_resize_ratio
        ball_position = np.asarray(ball_position, dtype=np.int32)
        return ball_position

    def configure_for_performance(self, ds):
        """Originally created by Yan Gobeil
        towardsdatascience.com/what-is-the-best-input-pipeline-to-train-image-classification-models-with-tf-keras-eb3fe26d3cc5
        """
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.configs.batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def get_dataset(self):
        """Creates and zips the dataset."""
        # Separate the data and convert into lists
        events_infor = np.asarray(self.events_infor)
        image_fps = events_infor[:, 0].tolist()
        ball_position = events_infor[:, 1].tolist()
        target_events = events_infor[:, 2].tolist()
        segmentation_fp = events_infor[:, 3].tolist()
        # Convert all of the data into tensor slices
        image_ds = data.Dataset.from_tensor_slices(image_fps)
        position_ds = data.Dataset.from_tensor_slices(ball_position)
        mask_ds = data.Dataset.from_tensor_slices(segmentation_fp)
        events_ds = data.Dataset.from_tensor_slices(target_events)
        # Map the associated function to the tensor slices
        print("Running image dataset.")
        image_ds = image_ds.map(
            lambda x: tf.numpy_function(
                self.parse_images, inp=[x], Tout=[tf.uint8]),
            num_parallel_calls=data.experimental.AUTOTUNE)
        print("Running mask dataset.")
        mask_ds = mask_ds.map(
            lambda x: tf.numpy_function(
                self.parse_masks, inp=[x], Tout=[tf.int8]),
            num_parallel_calls=data.experimental.AUTOTUNE)
        print("Running position dataset.")
        position_ds = position_ds.map(
            lambda x: tf.numpy_function(
                self.coordinate_adjustment, inp=[x], Tout=tf.int32),
            num_parallel_calls=data.experimental.AUTOTUNE)
        ds = data.Dataset.zip((image_ds, position_ds, events_ds, mask_ds))
        print("Dataset zipped.")
        ds = self.configure_for_performance(ds=ds)
        return ds
        
if __name__ == "__main__":
    # Test to see if dataset creation was successful
    from data_utils import *
    from configs import configs

    events_infor, events_labels = data_preparer(configs=configs)

    ttnet_dataset_creator = TTNetDataset(
        events_infor=events_infor,
        org_size=configs.original_image_shape,
        input_size=configs.processed_image_shape)

    ttnet_dataset = ttnet_dataset_creator.get_dataset()
