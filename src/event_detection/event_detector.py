import json
import os
from collections import defaultdict

import paths
from event_detection_models import DistanceThreshold


class Detector:

    def __init__(self):

        self.output_folder_path = os.path.join(paths.data_folder_path, 'detected_events')
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def detect_motion(self, annotations_file_path):
        # Checking if annotations file exist
        if not os.path.exists(annotations_file_path):
            print('Quitting: Annotations file path does not exist')
            return False

        # Reading annotations
        with open(annotations_file_path) as file:
            annotations = json.load(file)

        # Extracting coordinates for each coin to build individual event's detector for each object instance
        coin_coordinates = defaultdict(lambda: [])
        coin_events = defaultdict(lambda: [])
        chronological_events = []
        events_file_name = os.path.basename(annotations_file_path)

        num_frames = len(annotations)
        for frame_idx in range(num_frames):
            frame_annotations = annotations[frame_idx]
            for object in frame_annotations:
                if object['label'] == 3 or object['label'] == 4:
                    label = object['label']
                    for instance in object['instances']:
                        tracker_id = instance['tracker_id']
                        coin_coordinates[(label, tracker_id)].append(instance['coordinates'])

        for key in coin_coordinates.keys():
            coin_events[key] = DistanceThreshold().detect_events(coin_coordinates[key])
            for event in coin_events[key]:
                chronological_events.append(
                    {
                        'frame_idx': event[0],
                        'label': key[0],
                        'tracker_id': key[1],
                        'moving': event[1],
                        'coordinates': event[2]
                    }
                )

        # Sorting events according to frame index
        chronological_events.sort(key=lambda event: event['frame_idx'])

        # Writing detected events
        with open(os.path.join(self.output_folder_path, events_file_name), 'w') as file:
            json.dump(chronological_events, file, default=bool, indent=4)

        return True

# Driver Code
# detector = Detector()
# annotations_file_path = os.path.join(paths.data_folder_path, 'detected_and_tracked_objects', '77_bonus.json')
# detector.detect_motion(annotations_file_path)
