import os
import numpy as np
import json

class Grader:
    def __init__(self):
        pass

    def detect_events(self, annotations_file_path):
        if not os.path.exists(annotations_file_path):
            print('Quitting: Annotations file path does not exist')
            return

        with open(annotations_file_path) as file:
            annotations = json.load(file)

        num_frames = len(annotations)
        