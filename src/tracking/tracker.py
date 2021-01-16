import json
import os

from tqdm import tqdm

import paths
from src.tracking.tracker_models import CentroidTracker


class Tracker:
    def __init__(self):
        self.red_coin_tracker = CentroidTracker()
        self.green_coins_tracker = CentroidTracker()

        self.output_folder_path = os.path.join(paths.data_folder_path, 'tracked objects')
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def track_objects(self, annotations_file_path):
        if not os.path.exists(annotations_file_path):
            print('Quitting: Annotations file path does not exist')
            return

        with open(annotations_file_path) as file:
            annotations = json.load(file)

        num_frames = len(annotations)

        tracked_objects_file_name = os.path.basename(annotations_file_path)
        tracked_objects = []

        print('Tracking ...')
        progress_bar = tqdm(total=num_frames, unit=' frames')
        for frame_idx in range(num_frames):
            objects = annotations[frame_idx]['objects']
            red_coin_coordinates = []
            green_coins_coordinates = []
            for object in objects:
                if object['label'] == 3:
                    red_coin_coordinates.append(object['coordinates'])
                elif object['label'] == 4:
                    green_coins_coordinates.append(object['coordinates'])

            self.red_coin_tracker.update(red_coin_coordinates)
            self.green_coins_tracker.update(green_coins_coordinates)

            tracked_objects.append(
                {
                    'frame_idx': frame_idx,
                    'objects': [
                        {
                            'label': 3,
                            'coordinates': self.red_coin_tracker.object_coordinates.copy()
                        },
                        {
                            'label': 4,
                            'coordinates': self.green_coins_tracker.object_coordinates.copy()
                        }
                    ]
                }
            )

            progress_bar.update(1)
        progress_bar.close()

        with open(os.path.join(self.output_folder_path, tracked_objects_file_name), 'w') as file:
            json.dump(tracked_objects, file, indent=4)


tracker = Tracker()
annotations_file_path = os.path.join(paths.data_folder_path, 'detected objects', '77_bonus.json')
tracker.track_objects(annotations_file_path)
