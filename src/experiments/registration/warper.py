import os

import cv2
import numpy as np
from cv2 import aruco
from tqdm import tqdm

from src import paths


class PerspectiveWarper:
    def __init__(self):
        self.output_folder_path = os.path.join(paths.experiments_folder_path, 'registration', 'videos', 'warped')
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def find_anchor_points(self, image):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_1000)
        aruco_params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
        anchor_points = []
        for corner in corners:
            center = np.mean(corner, axis=1)[0]
            anchor_points.append([center[0], center[1]])

        if len(anchor_points) == 4:
            _, anchor_points = zip(*sorted(zip(ids, anchor_points), key=lambda element: element[0]))

        return anchor_points

    def warp_shallow(self, video_file_path, frame_size):
        try:
            if not os.path.exists(video_file_path):
                raise FileNotFoundError('Video file path does not exits')

            video = cv2.VideoCapture(video_file_path)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)

            anchor_points_found = False
            warping_matrix = None

            print('Searching frame with anchor points ...')
            progress_bar = tqdm(total=num_frames, unit=' frames')
            for frame_idx in range(num_frames):
                _, frame = video.read()

                anchor_points = np.array(self.find_anchor_points(frame), dtype=np.float32)
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

                    warping_matrix = cv2.getPerspectiveTransform(
                        anchor_points,
                        warped_anchor_points
                    )
                    progress_bar.update(num_frames)
                    break

                progress_bar.update(1)
            progress_bar.close()
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if not anchor_points_found:
                raise RuntimeError('Anchor points not found')

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

                frame = cv2.warpPerspective(frame, warping_matrix, dsize=frame_size)
                video_writer.write(frame)
                progress_bar.update(1)

            video_writer.release()
            progress_bar.close()

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

    def warp_deep(self, video_file_path, frame_size):

        try:
            if not os.path.exists(video_file_path):
                raise FileNotFoundError('Video file path does not exits')

            video = cv2.VideoCapture(video_file_path)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)

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
                anchor_points = np.array(self.find_anchor_points(frame), dtype=np.float32)
                if len(anchor_points) == 4:
                    video_anchor_points = np.asarray(anchor_points)
                    warped_anchor_points = np.array(
                        [
                            (frame_size[0], 0),
                            (0, 0),
                            (0, frame_size[1]),
                            (frame_size[0], frame_size[1]),
                        ],
                        dtype=np.float32
                    )

                    warping_matrix = cv2.getPerspectiveTransform(
                        video_anchor_points,
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

        except:
            print('Unknown error')
            return False

# Driver Code
# warp = PerspectiveWarper()
# videos_folder_path = os.path.join(paths.experiments_folder_path, 'registration', 'videos', 'raw')
# video_file_paths = [
#     os.path.join(videos_folder_path, video_file_name) for video_file_name in os.listdir(videos_folder_path)
# ]
# frame_size = (1080, 1080)
# for video_file_path in video_file_paths:
#     warp.warp_deep(video_file_path, frame_size)
#     print()
