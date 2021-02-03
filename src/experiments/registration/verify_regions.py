import os

import cv2
import numpy as np

import paths

masks = {
    'arena_center': cv2.imread(
        os.path.join(paths.experiments_folder_path, 'registration', 'binary_maps', 'arena_center.png'),
        cv2.IMREAD_GRAYSCALE
    ),
    'orange': cv2.imread(
        os.path.join(paths.experiments_folder_path, 'registration', 'binary_maps', 'orange.png'),
        cv2.IMREAD_GRAYSCALE
    ),
    'inner_purple': cv2.imread(
        os.path.join(paths.experiments_folder_path, 'registration', 'binary_maps', 'inner_purple.png'),
        cv2.IMREAD_GRAYSCALE
    ),
    'blue': cv2.imread(
        os.path.join(paths.experiments_folder_path, 'registration', 'binary_maps', 'blue.png'),
        cv2.IMREAD_GRAYSCALE
    ),
    'green': cv2.imread(
        os.path.join(paths.experiments_folder_path, 'registration', 'binary_maps', 'green.png'),
        cv2.IMREAD_GRAYSCALE
    ),
    'outer_purple': cv2.imread(
        os.path.join(paths.experiments_folder_path, 'registration', 'binary_maps', 'outer_purple.png'),
        cv2.IMREAD_GRAYSCALE
    )
}

alpha = 0.5
COLORS = {
    'arena_center': (255, 255, 255),
    'orange': (0, 69, 255),
    'inner_purple': (255, 0, 255),
    'blue': (255, 191, 0),
    'green': (50, 205, 50),
    'outer_purple': (255, 0, 255),
}

region_segments = np.zeros(shape=(1080, 1080, 3), dtype=np.uint8)

for mask_key in masks.keys():
    layer = np.zeros(shape=(1080, 1080, 3), dtype=np.uint8)
    layer[:, :, 0], layer[:, :, 1], layer[:, :, 2] = COLORS[mask_key][0], COLORS[mask_key][1], COLORS[mask_key][2]
    layer[np.logical_not(masks[mask_key])] = 0
    region_segments += layer

    # # Visualization of Layer
    # cv2.imshow(mask_key, mask)
    # key = cv2.waitKey(0)
    # if key == 27:
    #     cv2.destroyWindow(mask_key)

videos_folder_path = os.path.join(paths.experiments_folder_path, 'registration', 'videos', 'warped')
video_file_paths = [os.path.join(videos_folder_path, file_name) for file_name in os.listdir(videos_folder_path)]
for video_file_path in video_file_paths:
    video_file_name = os.path.basename(videos_folder_path)

    video = cv2.VideoCapture(video_file_path)
    num_frames = int(video.get(cv2.CAP_PROP_FPS))
    fps = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(num_frames):
        frame_read, frame = video.read()

        if frame_read:
            frame = cv2.addWeighted(frame, 0.85, region_segments, 0.15, 5)

            cv2.imshow(video_file_name, frame)
            key = cv2.waitKey(int(1000 / 30))
            if key == 27:
                break

    cv2.destroyAllWindows()
    video.release()
