# Created by Thomas Chia
# Based on the TTNet Paper

import os
import json

import numpy as np


def smooth_event_labelling(event_class, smooth_idx, event_frameidx):
    """Smooth event labeling.

    Target values were constructed as sin(n*pi/8).
    """
    target_events = np.zeros((2, ))
    if event_class < 2:
        n = smooth_idx - event_frameidx
        target_events[event_class] = np.cos(n * np.pi / 8)
        target_events[target_events < 0.01] = 0.
    return target_events


def data_split(events_infor, events_labels, configs):
    """Training data and validation data split.

    Notes:
        Ideally, all inputs should be shuffled.
        OR the outputs are shuffled, either way works.
    """
    events_infor = np.asarray(events_infor, dtype=object)
    events_labels = np.asarray(events_labels)
    split_ratio = configs.validation_split
    split_count = int(len(list(events_labels)) * split_ratio)
    print("The split count is: " + str(split_count))

    train_events_infor = []
    train_events_labels = []
    validation_events_infor = []
    validation_events_labels = []

    train_events_infor.append(events_infor[0:split_count])
    train_events_labels.append(events_labels[0:split_count])
    validation_events_infor.append(events_infor[split_count::])
    validation_events_labels.append(events_labels[split_count::])

    train_events_infor = np.asarray(train_events_infor)[0]
    validation_events_infor = np.asarray(validation_events_infor)[0]

    return (train_events_infor, train_events_labels,
            validation_events_infor, validation_events_labels)


def gaussian_distribution(ball_position_gt_x, ball_position_gt_y, configs):
    """Section 5.1 TTNet Architecture.
    Utilizes the ball's position as the miu, and average 
    ball size for that range as the sigma.
    """
    mu_x = ball_position_gt_x
    mu_y = ball_position_gt_y
    sigma = int(1) # Average std for the ball shape, subject to change
    frame_width = configs.processed_image_shape[0]
    frame_height = configs.processed_image_shape[1]
    # Templates for the distribution
    template_x = np.arange(frame_width, dtype=np.float32)
    template_y = np.arange(frame_height, dtype=np.float32)
    dist_x = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (template_x - mu_x)**2 / (2 * sigma**2))
    dist_y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (template_y - mu_y)**2 / (2 * sigma**2))
    return dist_x, dist_y

def coordinate_adjustment(ball_position, pos_type, configs):
    """Change the position coordinates of the ball to scale to training."""
    if pos_type == "x":
        ball_position = ball_position/(configs.original_image_shape[0]/configs.processed_image_shape[0]) 
        ball_position = np.asarray(ball_position, dtype=np.int32)
        return ball_position
    else:
        ball_position = ball_position/(configs.original_image_shape[1]/configs.processed_image_shape[1]) 
        ball_position = np.asarray(ball_position, dtype=np.int32)
        return ball_position

def gaussian_distribution(ball_position_gt, pos_type, configs):
    """Section 5.1 TTNet Architecture.
    Utilizes the ball's position as the miu, and average 
    ball size for that range as the sigma.
    """
    mu = ball_position_gt
    sigma = int(1) # Average std for the ball shape, subject to change
    frame_width = configs.processed_image_shape[0] 
    frame_height = configs.processed_image_shape[1] 
    if pos_type == "x":
        template_x = np.arange(frame_width, dtype=np.float32)
        dist = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (template_x - mu)**2 / (2 * sigma**2))
        return dist
    else:
        template_y = np.arange(frame_height, dtype=np.float32)
        dist = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (template_y - mu)**2 / (2 * sigma**2))
        return dist

def data_preparer(configs, dataset_type="training"):
    """Prepare the dataset for TF data."""
    if dataset_type == "training":
        num_frames_from_event = int(
            (configs.num_frames_sequence - 1) / 2)
        annos_dir = os.path.join(
            configs.data_dir, dataset_type, "annotations")
        images_dir = os.path.join(
            configs.data_dir, dataset_type, "images")
        games_list = os.listdir(images_dir)

        events_infor = []
        events_labels = []

        # Parse through each game folder.
        for game in games_list:
            ball_annos_path = os.path.join(
                annos_dir, game, "ball_markup.json")
            events_annos_path = os.path.join(
                annos_dir, game, "events_markup.json")
            # Load ball annotations
            json_ball = open(ball_annos_path)
            ball_annos = json.load(json_ball)
            # Load events annotations
            json_events = open(events_annos_path)
            events_annos = json.load(json_events)
            # Parse through each even in the JSON file.
            for event_frame, event_type in events_annos.items():
                event_frame = int(event_frame)
                # Get frames if event is important
                if event_type != "empty_event":
                    smooth_frames = [
                        idx for idx in range(
                            event_frame - num_frames_from_event,
                            event_frame + num_frames_from_event + 1)]
                for frame in smooth_frames:
                    sub_smooth_frames = [
                        idx for idx in range(
                            frame - num_frames_from_event,
                            frame + num_frames_from_event + 1)]
                    # Add the image paths into a list.
                    image_paths = []
                    for sub_smooth_idx in sub_smooth_frames:
                        image_path = os.path.join(
                            images_dir,
                            game,
                            "img_{:06d}.jpg".format(sub_smooth_idx))
                        image_paths.append(image_path)
                    # Get the last frame then append the ball position
                    last_frame = frame + num_frames_from_event
                    if '{}'.format(last_frame) not in ball_annos.keys():
                        # print('smooth_idx: {} -
                        # no ball position for the frame idx
                        # {}'.format(event_frame, last_frame))
                        continue
                    ball_position_xy = ball_annos["{}".format(last_frame)]
                    ball_position_x = np.array(
                        ball_position_xy["x"],
                        dtype=np.int)
                    ball_position_x = coordinate_adjustment(
                        ball_position=ball_position_x, 
                        pos_type="x",
                        configs=configs)
                    ball_position_x = gaussian_distribution(
                        ball_position_gt=ball_position_x,
                        pos_type="x",
                        configs=configs)
                    ball_position_y = np.array(
                        ball_position_xy["y"],
                        dtype=np.int)
                    ball_position_y = coordinate_adjustment(
                        ball_position=ball_position_y, 
                        pos_type="y",
                        configs=configs)
                    ball_position_y = gaussian_distribution(
                        ball_position_gt=ball_position_y,
                        pos_type="y",
                        configs=configs)
                    # if ball_position_xy[0] < 0 or ball_position_xy[1] < 0:
                    # continue

                    # Get the path to the segmentation frame from last frame
                    seg_path = os.path.join(
                        annos_dir,
                        game,
                        "segmentation_masks", "{}.png".format(last_frame))
                    if not os.path.isfile(seg_path):
                        # print("smooth_idx: {} -
                        # The segmentation path
                        # {} is invalid".format(frame, seg_path))
                        continue
                    events_dict = events_dict = {
                        "bounce": 0,
                        "net": 1,
                        "empty_event": 2}

                    event_class = events_dict[event_type]

                    target_events = smooth_event_labelling(
                        event_class, frame, event_frame)
                    events_infor.append(
                        [image_paths,
                         ball_position_x,
                         ball_position_y,
                         target_events,
                         seg_path])

                    if (target_events[0] == 0) and (target_events[1] == 0):
                        event_class = 2
                    events_labels.append(event_class)

    return events_infor, events_labels
