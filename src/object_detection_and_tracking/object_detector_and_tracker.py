import json
import os
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

from object_tracking_models import CentroidTracker
from src import paths

warnings.filterwarnings("ignore", category=UserWarning)


class DetectorAndTracker:

    def __init__(self, network, weights_file_path):
        # Initializing detection model
        self.network = network
        if os.path.isfile(weights_file_path):
            try:
                self.network.set_state_dict(torch.load(weights_file_path))
            except:
                print('Quitting: Cannot load weights to the network')
                return
        else:
            print('Quitting: Network weight\'s file does not exists')
            return

        if torch.cuda.is_available():
            self.device = 'cuda'
            torch.cuda.init()
        else:
            self.device = 'cpu'

        self.detection_probability_threshold = 0.95

        # Initializing tracker models for red and green coin(s)
        self.red_coin_tracker = CentroidTracker()
        self.green_coins_tracker = CentroidTracker()

        # Checking and creating output folder
        self.output_folder_path = os.path.join(paths.data_folder_path, 'detected_and_tracked_objects')
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def detect_and_track_objects(self, video_file_path):
        # Checking if video file path exist
        if not os.path.exists(video_file_path):
            print('Quitting: Video path does not exits')
            return False

        video = cv2.VideoCapture(video_file_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.network.eval()
        self.network.to(self.device)

        annotations = []

        # Searching frames for objects
        print('Searching Objects ...')
        progress_bar = tqdm(total=num_frames, unit=' frames')
        for frame_idx in range(num_frames):
            _, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255

            # Forward passing through neural networks
            predicted_annotations = self.network.predict_batch(
                torch.unsqueeze(
                    torch.as_tensor(np.transpose(frame, [2, 0, 1]), dtype=torch.float32),
                    dim=0
                ).to(self.device)
            )[0]

            # Copying tensors to RAM and converting to numpy arrays
            predicted_annotations['labels'] = predicted_annotations['labels'].to('cpu').detach().numpy()
            predicted_annotations['boxes'] = predicted_annotations['boxes'].to('cpu').detach().numpy()
            predicted_annotations['scores'] = predicted_annotations['scores'].to('cpu').detach().numpy()

            # Sorting according to labels
            sorting_idxes = np.argsort(predicted_annotations['labels'])
            predicted_annotations['labels'] = predicted_annotations['labels'][sorting_idxes]
            predicted_annotations['boxes'] = predicted_annotations['boxes'][sorting_idxes]
            predicted_annotations['scores'] = predicted_annotations['scores'][sorting_idxes]

            red_coin_coordinates = []
            green_coins_coordinates = []

            # Building frame annotations and extracting coordinates for red and green coins
            frame_annotations = []
            for i in range(len(predicted_annotations['scores'])):
                if predicted_annotations['scores'][i] > self.detection_probability_threshold:
                    if predicted_annotations['labels'][i] == 1 or predicted_annotations['labels'][i] == 2:
                        frame_annotations.append(
                            {
                                'label': int(predicted_annotations['labels'][i]),
                                'coordinates':
                                    (
                                        float((predicted_annotations['boxes'][i][0] + predicted_annotations['boxes'][i][2]) / 2),
                                        float((predicted_annotations['boxes'][i][1] + predicted_annotations['boxes'][i][3]) / 2)
                                    )
                            }
                        )
                    elif predicted_annotations['labels'][i] == 3:
                        red_coin_coordinates.append(
                            (
                                float((predicted_annotations['boxes'][i][0] + predicted_annotations['boxes'][i][2]) / 2),
                                float((predicted_annotations['boxes'][i][1] + predicted_annotations['boxes'][i][3]) / 2)
                            )
                        )
                    else:
                        green_coins_coordinates.append(
                            (
                                float((predicted_annotations['boxes'][i][0] + predicted_annotations['boxes'][i][2]) / 2),
                                float((predicted_annotations['boxes'][i][1] + predicted_annotations['boxes'][i][3]) / 2)
                            )
                        )

            # Tracking objects
            self.red_coin_tracker.update(red_coin_coordinates)
            self.green_coins_tracker.update(green_coins_coordinates)

            # Building frame annotations
            frame_annotations.append(
                {
                    'label': 3,
                    'instances': self.red_coin_tracker.get_objects_dictionary()
                }
            )
            frame_annotations.append(
                {
                    'label': 4,
                    'instances': self.green_coins_tracker.get_objects_dictionary()
                }
            )

            # Adding to over all annotations
            annotations.append(frame_annotations)

            progress_bar.update(1)
        progress_bar.close()

        # Writing annotations to file
        annotation_file_name = os.path.basename(video_file_path).replace('.mp4', '.json')
        with open(os.path.join(self.output_folder_path, annotation_file_name), 'w') as file:
            json.dump(annotations, file, indent=4)

        return True

# Driver Code
# network = ResNet50_FasterRCNN()
# weights_file_path = os.path.join(paths.resrc_folder_path, 'trained models', 'ResNet50_FasterRCNN', 'primary.pt')
# detector = DetectorAndTracker(network, weights_file_path)
# video_file_path = os.path.join(paths.data_folder_path, 'raw', '77_bonus.mp4')
# detector.detect_and_track_objects(video_file_path)
