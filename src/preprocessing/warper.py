import os

import cv2
import numpy as np
from cv2 import aruco
from tqdm import tqdm

from src import paths


class PerspectiveWarper:

    def __init__(self):
        # Checking and creating output folder
        self.output_folder_path = os.path.join(paths.data_folder_path, 'preprocessed', 'warped')
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def __find_anchor_points(self, image):
        # Finding centers of aruco markers
        aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_1000)
        aruco_params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
        anchor_points = []
        for corner in corners:
            center = np.mean(corner, axis=1)[0]
            anchor_points.append([center[0], center[1]])

        # Sorting aruco markers according to their ids
        if len(anchor_points) == 4:
            _, anchor_points = zip(*sorted(zip(ids, anchor_points), key=lambda element: element[0]))

        return anchor_points

    def warp_shallow(self, video_file_path, frame_size):
        try:
            # Checking if video file path exist
            if not os.path.exists(video_file_path):
                raise FileNotFoundError('Video file path does not exits')

            video = cv2.VideoCapture(video_file_path)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)

            anchor_points_found = False
            warping_matrix = None

            # Searching frame(s) for anchor points
            print('Searching frame with anchor points ...')
            progress_bar = tqdm(total=num_frames, unit=' frames')
            for frame_idx in range(num_frames):
                _, frame = video.read()

                anchor_points = np.array(self.__find_anchor_points(frame), dtype=np.float32)
                if len(anchor_points) == 4:
                    anchor_points_found = True

                    warped_anchor_points = np.array(
                        [
                            (frame_size[0], 0),
                            (0, 0),
                            (0, frame_size[1]),
                            (frame_size[0], frame_size[1]),
                        ],
                        dtype=np.float32
                    )

                    # Building warping matrix
                    warping_matrix = cv2.getPerspectiveTransform(
                        anchor_points,
                        warped_anchor_points
                    )
                    progress_bar.update(num_frames)
                    break

                progress_bar.update(1)
            progress_bar.close()
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Checking if anchor points found
            if not anchor_points_found:
                raise RuntimeError('Anchor points not found')

            # Writing warped video
            video_file_name = os.path.basename(video_file_path)

            video_writer = cv2.VideoWriter(
                os.path.join(self.output_folder_path, video_file_name),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                frame_size
            )
            print('Warping ...')
            progress_bar = tqdm(total=num_frames, unit=' frames')
            for frame_idx in range(num_frames):
                _, frame = video.read()

                # Applying warping to frames
                frame = cv2.warpPerspective(frame, warping_matrix, dsize=frame_size)
                video_writer.write(frame)
                progress_bar.update(1)

            video_writer.release()
            progress_bar.close()

            return True

        except FileNotFoundError as error:
            print('File Not Found Error: {}'.format(error))
            return False

        except RuntimeError as error:
            print('Runtime Error: {}'.format(error))
            return False

        except:
            print('Unknown error')
            return False

    def warp_deep(self, video_file_path, frame_size):
        try:
            # Checking if video file path exist
            if not os.path.exists(video_file_path):
                raise FileNotFoundError('Video file path does not exits')

            video = cv2.VideoCapture(video_file_path)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)

            # Searching frame(s) for anchor points
            video_anchor_points = np.ndarray(shape=(num_frames, 4, 2), dtype=np.float32)
            anchor_points_found = True

            print('Searching frames for anchor points ...')
            progress_bar = tqdm(total=num_frames, unit=' frames')
            for frame_idx in range(num_frames):
                _, frame = video.read()

                anchor_points = np.array(self.__find_anchor_points(frame), dtype=np.float32)
                if len(anchor_points) == 4:
                    video_anchor_points[frame_idx] = np.asarray(anchor_points)
                else:
                    anchor_points_found = False
                    break
                progress_bar.update(1)
            progress_bar.close()
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Checking if anchor points found for each frame
            if not anchor_points_found:
                raise RuntimeError('Anchor points not found')

            # Writing warped video
            video_file_name = os.path.basename(video_file_path)

            video_writer = cv2.VideoWriter(
                os.path.join(self.output_folder_path, video_file_name),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                frame_size
            )
            print('Warping ...')
            progress_bar = tqdm(total=num_frames, unit=' frames')
            for frame_idx in range(num_frames):
                _, frame = video.read()

                warped_anchor_points = np.array(
                    [
                        (frame_size[0], 0),
                        (0, 0),
                        (0, frame_size[1]),
                        (frame_size[0], frame_size[1]),
                    ],
                    dtype=np.float32
                )

                # Building warping matrix and applying warping to each frame
                warping_matrix = cv2.getPerspectiveTransform(
                    video_anchor_points[frame_idx],
                    warped_anchor_points
                )

                frame = cv2.warpPerspective(frame, warping_matrix, dsize=frame_size)
                video_writer.write(frame)
                progress_bar.update(1)

            progress_bar.close()
            video_writer.release()

            return True

        except FileNotFoundError as error:
            print('File not found error: {}'.format(error))
            return False

        except RuntimeError as error:
            print('Runtime error: {}'.format(error))
            return False

        except:
            print('Unknown error')
            return False
