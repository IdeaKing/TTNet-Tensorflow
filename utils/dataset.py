# Created by Thomas Chia
# https://towardsdatascience.com/what-is-the-best-input-pipeline-to-train-image-classification-models-with-tf-keras-eb3fe26d3cc5

import numpy as np
import cv2
from tensorflow import data

class Dataset():
    def __init__(self, events_infor, org_size, input_size, transform=True):
        self.events_infor = events_infor
        self.w_org = org_size[0]
        self.h_org = org_size[1]
        self.w_input = input_size[0]
        self.h_input = input_size[1]
        self.w_resize_ratio = self.w_org / self.w_input
        self.h_resize_ratio = self.h_org / self.h_input
        self.transform = transform
    
    def parse_images(self, images, mask=False):
        """Open and perform operations on all images."""
        if mask:
            # Processing if the image is a mask.
            image = cv2.imread(images)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        else:
            # Processing if the image is a group of images.
            image_stack = []
            for image_path in images:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.w_input, self.h_input))
                image_stack.append(image)
            # https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
            image_stack = np.dstack(image_stack)
            return image_stack
    
    def coordinate_adjustment(self, ball_position):
        """Change the position coordinates of the ball to scale to training."""
        ball_position[0] = ball_position[0]/self.w_resize_ratio
        ball_position[1] = ball_position[1]/self.w_resize_ratio
        return ball_position

    def configure_mask(self, mask):
        """Configure the segmentation mask."""
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
        image_fps = self.events_infor[0]
        ball_position = self.events_infor[1]
        target_events = self.events_infor[2]
        segmentation_fp = self.events_infor[3]

        image_stack = self.parse_images(images=image_fps, mask=False)
        ball_position = self.coordinate_adjustment(ball_position=ball_position)
        mask = self.parse_images(images=segmentation_fp, mask=True)

        # image_stack, ball_position, mask = self.augment(image, coordinate, mask)
        image_ds = data.Dataset.from_tensor_slices(image_stack)
        position_ds = data.Dataset.from_tensor_slices(ball_position)
        mask_ds = data.Dataset.from_tensor_slices(mask)
        events_ds = data.Dataset.from_tensor_slices(target_events)

        image_ds = image_ds.map(self.parse_images(), num_parallel_calls=data.experimental.AUTOTUNE)
        position_ds = position_ds.map(self.coordinate_adjustment(), num_parallel_calls=data.experimental.AUTOTUNE)
        mask_ds = mask_ds.map(self.parse_images(mask=True), num_parallel_calls=data.experimental.AUTOTUNE)

        ds = data.Dataset.zip((image_ds, position_ds, mask_ds, events_ds))
        return ds




        
        
        




