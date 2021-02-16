import os

import cv2

import paths
from regions_detector_models import AveragedCircleTransforms


class Detector:
    def __init__(self):

        self.output_folder_path = os.path.join(paths.data_folder_path, 'region_masks')
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def detect_regions(self, video_file_path, annotations_file_path):
        if not os.path.exists(video_file_path):
            print('Quitting: Video file path does not exist')
            return

        if not os.path.exists(annotations_file_path):
            print('Quitting: Annotations file path does not exist')
            return

        region_masks = AveragedCircleTransforms().generate_region_masks(video_file_path, annotations_file_path)

        region_masks_folder_path = os.path.join(self.output_folder_path, os.path.splitext(os.path.basename(video_file_path))[0])
        if not os.path.isdir(region_masks_folder_path):
            os.makedirs(region_masks_folder_path)
        for region_key in region_masks.keys():
            cv2.imwrite(os.path.join(region_masks_folder_path, '{}.png'.format(region_key)), region_masks[region_key])


detector = Detector()
video_file_path = os.path.join(paths.data_folder_path, 'raw', '77_bonus.mp4')
annotations_file_path = os.path.join(
    paths.data_folder_path,
    'detected_and_tracked_objects',
    '77_bonus.json'
)
detector.detect_regions(video_file_path, annotations_file_path)
