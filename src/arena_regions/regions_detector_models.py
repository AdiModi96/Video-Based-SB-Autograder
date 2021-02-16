import json
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import paths


class AveragedCircleTransforms:

    def __init__(self):
        self.region_properties = {
            'white': {
                'boundary_radius_range': (10, 10),
                'hsv_range': None
            },
            'orange': {
                'boundary_radius_range': (45, 65),
                'hsv_range': None
            },
            'purple': {
                'boundary_radius_range': (100, 130),
                'hsv_range': None
            },
            'blue': {
                'boundary_radius_range': (160, 190),
                'hsv_range': None
            },
            'green': {
                'boundary_radius_range': None,
                'hsv_range': [
                    (35, 0, 0),
                    (70, 255, 255)
                ]
            }
        }
        self.num_frames_to_skip_for_observations = 10

        self.output_folder_path = os.path.join(paths.data_folder_path, 'region_masks')
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def _process_frame(self, frame):
        processed_frame = frame.copy()
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.GaussianBlur(processed_frame, ksize=(5, 5), sigmaX=3, sigmaY=3)
        processed_frame = cv2.Laplacian(processed_frame, ddepth=cv2.CV_8U, ksize=5)
        processed_frame = cv2.erode(processed_frame, kernel=np.ones(shape=(3, 3), dtype=np.uint8))
        processed_frame = cv2.dilate(processed_frame, kernel=np.ones(shape=(3, 3), dtype=np.uint8))
        _, processed_frame = cv2.threshold(processed_frame, thresh=25, maxval=255, type=cv2.THRESH_BINARY)
        processed_frame = cv2.dilate(processed_frame, kernel=np.ones(shape=(3, 3), dtype=np.uint8))
        return processed_frame

    def generate_region_masks(self, video_file_path, annotations_file_path):

        # Checking if video file path exist
        if not os.path.exists(video_file_path):
            print('Quitting: Video file path does not exist')
            return

        # Checking if annotations file path exist
        if not os.path.exists(annotations_file_path):
            print('Quitting: Annotations file path does not exist')
            return

        video = cv2.VideoCapture(video_file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        y, x = np.mgrid[0:video.get(cv2.CAP_PROP_FRAME_HEIGHT), 0:video.get(cv2.CAP_PROP_FRAME_WIDTH)]

        with open(annotations_file_path) as file:
            annotations = json.load(file)
        arena_center_coordinates = tuple(map(int, annotations[0][0]['coordinates']))

        region_boundary_radii = {
            'white': 10,
            'orange': -1,
            'purple': -1,
            'blue': -1,
        }

        region_masks = defaultdict(
            lambda: np.zeros(
                shape=(
                    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                )
            )
        )

        num_observations = 0
        print('Detecting regions ...')
        progress_bar = tqdm(total=num_frames, unit=' frames')
        for frame_idx in range(0, num_frames, self.num_frames_to_skip_for_observations):

            _, frame = video.read()

            for region_key in region_boundary_radii:
                if region_key != 'white':

                    min_radius, max_radius = self.region_properties[region_key]['boundary_radius_range']
                    processed_frame = self._process_frame(frame)

                    # # Visualization of processed frame
                    # cv2.imshow('Processed frame', processed_frame)
                    # key = cv2.waitKey(0)
                    # if key == 27:
                    #     cv2.destroyAllWindows()

                    processed_frame_instance = processed_frame.copy()
                    processed_frame_instance[
                        (x - arena_center_coordinates[0]) ** 2 + (y - arena_center_coordinates[1]) ** 2 < min_radius ** 2
                        ] = 0
                    processed_frame_instance[
                        (x - arena_center_coordinates[0]) ** 2 + (y - arena_center_coordinates[1]) ** 2 > max_radius ** 2
                        ] = 0

                    detected_circles = cv2.HoughCircles(
                        processed_frame_instance,
                        cv2.HOUGH_GRADIENT,
                        dp=1,
                        minDist=100,
                        param1=40,
                        param2=10,
                        minRadius=min_radius,
                        maxRadius=max_radius
                    )

                    if detected_circles is not None:
                        radius = detected_circles[0, 0, 2]
                        region_boundary_radii[region_key] = ((region_boundary_radii[region_key] * num_observations) + radius) / (num_observations + 1)

            _, green_region = cv2.threshold(
                cv2.inRange(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL),
                    lowerb=self.region_properties['green']['hsv_range'][0],
                    upperb=self.region_properties['green']['hsv_range'][1]
                ),
                thresh=128,
                maxval=1,
                type=cv2.THRESH_BINARY
            )
            region_masks['green'] += green_region

            num_observations += 1
            progress_bar.update(self.num_frames_to_skip_for_observations)
        progress_bar.close()

        print('Generating masks ...')
        cumulative_mask = np.zeros(
            shape=(
                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            )
        )
        # Generating green region
        region_masks['green'][region_masks['green'] >= (num_observations // 2)] = 255
        region_masks['green'][region_masks['green'] != 255] = 0
        region_masks['green'][
            (x - arena_center_coordinates[0]) ** 2 + (y - arena_center_coordinates[1]) ** 2 > region_boundary_radii['blue'] ** 2
            ] = 0
        cumulative_mask = cv2.bitwise_or(cumulative_mask, region_masks['green'])

        # Generating white region
        region_masks['white'][
            (x - arena_center_coordinates[0]) ** 2 + (y - arena_center_coordinates[1]) ** 2 <= region_boundary_radii['white'] ** 2
            ] = 255
        region_masks['white'] = cv2.bitwise_and(region_masks['white'], cv2.bitwise_not(cumulative_mask))
        cumulative_mask = cv2.bitwise_or(cumulative_mask, region_masks['white'])

        # Generating orange region
        region_masks['orange'][
            (x - arena_center_coordinates[0]) ** 2 + (y - arena_center_coordinates[1]) ** 2 <= region_boundary_radii['orange'] ** 2
            ] = 255
        region_masks['orange'] = cv2.bitwise_and(region_masks['orange'], cv2.bitwise_not(cumulative_mask))
        cumulative_mask = cv2.bitwise_or(cumulative_mask, region_masks['orange'])

        # Generating purple region
        region_masks['purple'][
            (x - arena_center_coordinates[0]) ** 2 + (y - arena_center_coordinates[1]) ** 2 <= region_boundary_radii['purple'] ** 2
            ] = 255
        region_masks['purple'] = cv2.bitwise_and(region_masks['purple'], cv2.bitwise_not(cumulative_mask))
        cumulative_mask = cv2.bitwise_or(cumulative_mask, region_masks['purple'])

        # Generating blue region
        region_masks['blue'][
            (x - arena_center_coordinates[0]) ** 2 + (y - arena_center_coordinates[1]) ** 2 <= region_boundary_radii['blue'] ** 2
            ] = 255
        region_masks['blue'] = cv2.bitwise_and(region_masks['blue'], cv2.bitwise_not(cumulative_mask))
        cumulative_mask = cv2.bitwise_or(cumulative_mask, region_masks['blue'])

        # Writing generated masks
        region_masks_folder_path = os.path.join(self.output_folder_path, os.path.splitext(os.path.basename(video_file_path))[0])
        if not os.path.isdir(region_masks_folder_path):
            os.makedirs(region_masks_folder_path)
        for region_key in region_masks.keys():
            cv2.imwrite(os.path.join(region_masks_folder_path, '{}.png'.format(region_key)), region_masks[region_key])

        return True
