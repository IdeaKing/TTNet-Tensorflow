# Created by Thomas Chia
# https://towardsdatascience.com/what-is-the-best-input-pipeline-to-train-image-classification-models-with-tf-keras-eb3fe26d3cc5
import numpy as np
import cv2
from tensorflow import data

import tensorflow as tf

# tf.config.run_functions_eagerly(True)
class Dataset(data):
    def __init__(self, events_infor, org_size, input_size, transform=True):
        self.events_infor = events_infor
        self.w_org = org_size[0]
        self.h_org = org_size[1]
        self.w_input = input_size[0]
        self.h_input = input_size[1]
        self.w_resize_ratio = self.w_org / self.w_input
        self.h_resize_ratio = self.h_org / self.h_input
        self.transform = transform
    
    def parse_images(self, images: tf.Tensor):
        """Open and perform operations on all images."""
        # Processing if the image is a group of images.
        #tf.executing_eagerly()
        #print(images.numpy())  # list(images.as_numpy_iterator()))n # tf dataset stuff are run in graph mode and thus .numpy does not work
        print("image path --------------------------------------")
        tf.print(type(images))
        print("end ===================================")
        #print("---------------------")
        image_stack = []
        for image_path in images:

            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            # print(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.w_input, self.h_input))
            image_stack.append(image)
        # https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
        image_stack = np.dstack(image_stack)
        return image_stack

    def parse_masks(self, mask_path):
        """Open and perform operations on all images."""
        # Processing if the image is a mask.
        mask = cv2.imread(str(mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[mask < 75] = 0.
        mask = mask[mask >= 75] = 1.
        return mask

    def coordinate_adjustment(self, ball_position):
        """Change the position coordinates of the ball to scale to training."""
        ball_position[0] = ball_position[0]/self.w_resize_ratio
        ball_position[1] = ball_position[1]/self.w_resize_ratio
        return ball_position

    def configure_mask(self, mask):
        """Configure the segmentation mask."""
        mask = tf.io.read_file(mask)
        mask = tf.image.decode_jpeg(mask, channels=3)
        
        # Segmentation mask should be 0 or 1
        mask = mask[mask < 75] = 0.
        mask = mask[mask >= 75] = 1.
        return mask
    
    def augment(self, image, coordinate, mask):
        """Augmentation techniques used by paper:
        - Random Cropping +- 15%
        - Rotation +- 15% 
        - Horizontal Flip 
        - Random Brightness, Contrast, Hueshifts
        """
        return "Might work on this later..."

    def dataset(self):

        events_infor = np.asarray(self.events_infor)
        image_fps = events_infor[:,0].tolist()
        ball_position = events_infor[:,1].tolist()
        target_events = events_infor[:,2].tolist()
        segmentation_fp = events_infor[:,3].tolist()

        # image_stack, ball_position, mask = self.augment(image, coordinate, mask)
        image_ds = data.Dataset.from_tensor_slices(image_fps)
        position_ds = data.Dataset.from_tensor_slices(ball_position)
        mask_ds = data.Dataset.from_tensor_slices(segmentation_fp)
        events_ds = data.Dataset.from_tensor_slices(target_events)


        # for z in image_ds:
            # print(z.numpy())

        # print(list(image_ds.as_numpy_iterator()))

        # position_ds = position_ds.map(self.coordinate_adjustment, num_parallel_calls=data.experimental.AUTOTUNE)
        image_ds = image_ds.map(self.parse_images, num_parallel_calls=data.experimental.AUTOTUNE)

        # image_ds = image_ds.map(lambda x: tf.py_function(self.parse_images, inp=[x], Tout=tf.float32), num_parallel_calls=data.experimental.AUTOTUNE)
        # mask_ds = mask_ds.map(self.parse_masks, num_parallel_calls=data.experimental.AUTOTUNE)

        ds = data.Dataset.zip((image_ds, position_ds, mask_ds, events_ds))
        return ds

if __name__ == "__main__":
    from data_utils import *
    from configs import configs

    events_infor, events_labels = data_preparer(configs=configs)

    ttnet_data = Dataset(
        events_infor=events_infor, org_size=configs.original_image_shape, input_size=configs.processed_image_shape)

    ttnet_dataset = ttnet_data.dataset()
        
        
        




