import os

import paths
from regions_detector_models import AveragedCircleTransforms

detector = AveragedCircleTransforms()
video_file_path = os.path.join(paths.data_folder_path, 'raw', '77_bonus.mp4')
annotations_file_path = os.path.join(paths.data_folder_path, 'detected_and_tracked_objects', '77_bonus.json')
detector.generate_region_masks(video_file_path, annotations_file_path)
