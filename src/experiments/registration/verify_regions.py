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

COLORS = {
    'arena_center': (255, 255, 255),
    'orange': (0, 52, 127),
    'inner_purple': (175, 0, 117),
    'blue': (127, 73, 0),
    'green': (8, 76, 0),
    'outer_purple': (175, 0, 117),
}

region_segments = np.zeros(shape=(1080, 1080, 3), dtype=np.uint8)

for mask_key in masks.keys():
    layer = np.zeros(shape=(1080, 1080, 3), dtype=np.uint8)
    layer[:, :, 0], layer[:, :, 1], layer[:, :, 2] = COLORS[mask_key][0], COLORS[mask_key][1], COLORS[mask_key][2]
    masks[mask_key] = cv2.Laplacian(
        masks[mask_key],
        ddepth=cv2.CV_8U,
    )
    masks[mask_key] = cv2.dilate(masks[mask_key], kernel=np.ones(shape=(5, 5), dtype=np.uint8))
    layer[np.logical_not(masks[mask_key])] = 0
    region_segments += layer

    # # Visualization of individual mask (layer)
    # cv2.imshow(mask_key, cv2.resize(layer, dsize=(720, 720)))
    # key = cv2.waitKey(0)
    # if key == 27:
    #     cv2.destroyWindow(mask_key)

# Visualization of collective overlay layer
cv2.imshow('Region Mask', cv2.resize(region_segments, dsize=(720, 720)))
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyWindow('Region Mask')

alpha = 0.25
playback_speed = 1.5
videos_folder_path = os.path.join(paths.experiments_folder_path, 'registration', 'videos', 'warped')
video_file_paths = [os.path.join(videos_folder_path, file_name) for file_name in os.listdir(videos_folder_path)]
for video_file_path in video_file_paths:
    video_file_name = os.path.basename(video_file_path)

    video = cv2.VideoCapture(video_file_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    for frame_idx in range(num_frames):
        frame_read, frame = video.read()

        if frame_read:
            frame = cv2.addWeighted(frame, alpha, region_segments, 1 - alpha, 5)

            cv2.imshow(video_file_name, cv2.resize(frame, dsize=(720, 720)))
            key = cv2.waitKey(int(playback_speed * 1000 / fps))
            if key == 27:
                break

    cv2.destroyAllWindows()
    video.release()
