import os

import cv2
import numpy as np
from tqdm import tqdm

import paths


class Trimmer:

    def __init__(self):
        # Mean limit to classify a frame as black
        self.mean_limit = 20

        # Checking and creating output folder
        self.output_folder_path = os.path.join(paths.data_folder_path, 'preprocessed', 'trimmed')
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def __is_frame_black(self, frame):
        result = np.mean(frame) <= self.mean_limit
        return result

    def trim(self, video_file_path):
        # Checking if video file path exist
        if not os.path.exists(video_file_path):
            print('Quitting: Video path does not exits')
            return False

        video = cv2.VideoCapture(video_file_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Detecting separator(s)
        anchor_frame_idxes = []
        previous_frame_black = False
        print('Finding separators (black frame(s)) ...')
        progress_bar = tqdm(total=num_frames, unit=' frames')
        for frame_idx in range(num_frames):
            read, frame = video.read()

            if read:
                if self.__is_frame_black(frame):
                    if not previous_frame_black:
                        anchor_frame_idxes.append(frame_idx)
                    previous_frame_black = True
                else:
                    previous_frame_black = False

            progress_bar.update(1)
        progress_bar.close()

        # Checking for the number of separator(s) detected
        if len(anchor_frame_idxes) == 0:
            print('Quitting: No separator (black frame) found')
            return False
        elif len(anchor_frame_idxes) == 1:
            start_frame_idx, end_frame_idx = anchor_frame_idxes[0], num_frames
        elif len(anchor_frame_idxes) == 2:
            start_frame_idx, end_frame_idx = anchor_frame_idxes[0], anchor_frame_idxes[1]
        else:
            print('Quitting: More than 2 separators (black frames) found')
            return False

        # Writing trimmed video
        video_file_name = os.path.basename(video_file_path)

        video_writer = cv2.VideoWriter(
            os.path.join(self.output_folder_path, video_file_name),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            frame_size
        )
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        print('Trimming ...')
        progress_bar = tqdm(total=end_frame_idx - start_frame_idx, unit=' frames')
        for frame_idx in range(start_frame_idx, end_frame_idx):
            read, frame = video.read()

            if read:
                _, frame = video.read()
                if not self.__is_frame_black(frame):
                    video_writer.write(frame)

            progress_bar.update(1)
        progress_bar.close()
        video_writer.release()

        return True
