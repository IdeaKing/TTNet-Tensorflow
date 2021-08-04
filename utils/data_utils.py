from utils.configs import configs

import os
import json
import numpy as np

def smooth_event_labelling(event_class, smooth_idx, event_frameidx):
    """Smooth event labeling.
    
    Target values were constructed as sin(n*pi/8)."""
    target_events = np.zeros((2,))
    if event_class < 2:
        n = smooth_idx - event_frameidx
        target_events[event_class] = np.cos(n * np.pi / 8)
        target_events[target_events < 0.01] = 0.
    return target_events

def data_preparer(configs, dataset_type="training"):
    """Prepare the dataset for TF data."""
    if dataset_type=="training":
        num_frames_from_event = int((configs.num_frames_sequence - 1) / 2)
        annos_dir = os.path.join(configs.data_dir, dataset_type, "annotations")
        images_dir = os.path.join(configs.data_dir, dataset_type, "images")
        games_list = os.listdir(images_dir)

        events_infor = []
        events_labels = []

        # Parse through each game folder.
        for game in games_list:
            ball_annos_path = os.path.join(annos_dir, game, "ball_markup.json")
            events_annos_path = os.path.join(annos_dir, game, "events_markup.json")
            # Load ball annotations
            json_ball = open(ball_annos_path)
            ball_annos = json.load(json_ball)
            # Load events annotations
            json_events = open(events_annos_path)
            events_annos = json.load(json_events)
            # Parse through each even in the JSON file.
            for event_frame, event_type in events_annos.items():
                event_frame = int(event_frame)
                # If there is a meaningful event, then get the frames around the event.
                if event_type != "empty_event":
                    smooth_frames = [idx for idx in range(event_frame - num_frames_from_event,
                                                event_frame + num_frames_from_event + 1)]
                for frame in smooth_frames:
                    sub_smooth_frames = [idx for idx in range(frame - num_frames_from_event,
                                                              frame + num_frames_from_event + 1)]
                    
                    # Add the image paths into a list.
                    image_paths = []
                    for sub_smooth_idx in sub_smooth_frames:
                        image_path = os.path.join(images_dir, game, "img_{:06d}.jpg".format(sub_smooth_idx))
                        image_paths.append(image_path)

                    # Get the last frame in the sequence so that we can append the ball position
                    last_frame = frame + num_frames_from_event
                    ball_position_xy = ball_annos["{}".format(last_frame)]
                    ball_position_xy = np.array([ball_position_xy["x"], ball_position_xy["y"]], dtype=np.int)

                    # if ball_position_xy[0] < 0 or ball_position_xy[1] < 0:
                        # continue
                    
                    # Get the path to the segmentation frame using the last frame
                    seg_path = os.path.join(annos_dir, game, "segmentation_masks", "{}.png".format(last_frame))
                    if not os.path.isfile(seg_path):
                        print("smooth_idx: {} - The segmentation path {} is invalid".format(frame, seg_path))
                        continue
                    events_dict = events_dict = {
                        "bounce": 0,
                        "net": 1,
                        "empty_event": 2}

                    event_class = events_dict[event_type]

                    target_events = smooth_event_labelling(event_class, frame, event_frame)
                    events_infor.append([image_paths, ball_position_xy, target_events, seg_path])

                    if (target_events[0] == 0) and (target_events[1] == 0):
                        event_class = 2
                    events_labels.append(event_class)

    return events_infor, events_labels
