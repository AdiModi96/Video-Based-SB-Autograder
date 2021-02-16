import os

import cv2
import numpy as np

import paths

video_file_path = os.path.join(paths.data_folder_path, 'raw', '77_bonus.mp4')
region_masks_folder_path = os.path.join(paths.data_folder_path, 'region_masks', '77_bonus')

region_masks = {}
for region_mask_file_name in os.listdir(region_masks_folder_path):
    key = os.path.splitext(region_mask_file_name)[0]
    region_masks[key] = cv2.imread(
        os.path.join(region_masks_folder_path, region_mask_file_name),
        cv2.IMREAD_GRAYSCALE
    )

COLORS = {
    'white': (255, 255, 255),
    'orange': (0, 52, 127),
    'purple': (175, 0, 117),
    'blue': (127, 73, 0),
    'green': (8, 76, 0),
}

cumulative_region_boundaries = np.zeros(shape=(720, 1280, 3), dtype=np.uint8)

for region_key in region_masks.keys():
    layer = np.zeros(shape=(720, 1280, 3), dtype=np.uint8)
    layer[:, :, 0], layer[:, :, 1], layer[:, :, 2] = COLORS[region_key][0], COLORS[region_key][1], COLORS[region_key][2]
    region_masks[region_key] = cv2.Laplacian(
        region_masks[region_key],
        ddepth=cv2.CV_8U,
    )

    region_masks[region_key] = cv2.dilate(region_masks[region_key], kernel=np.ones(shape=(5, 5), dtype=np.uint8))
    layer[np.logical_not(region_masks[region_key])] = 0
    cumulative_region_boundaries += layer

    # # Visualization of individual mask (layer)
    # cv2.imshow(region_key, layer)
    # key = cv2.waitKey(0)
    # if key == 27:
    #     cv2.destroyWindow(region_key)

# Visualization of collective overlay layer
cv2.imshow('Cumulative Region Boundaries', cumulative_region_boundaries)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

alpha = 0.25
playback_speed = 1.5
video_file_name = os.path.basename(video_file_path)

video = cv2.VideoCapture(video_file_path)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
for frame_idx in range(num_frames):
    frame_read, frame = video.read()

    if frame_read:
        frame = cv2.addWeighted(frame, alpha, cumulative_region_boundaries, 1 - alpha, 5)

        cv2.imshow(video_file_name, frame)
        key = cv2.waitKey(int(playback_speed * 1000 / fps))
        if key == 27:
            break

cv2.destroyAllWindows()
video.release()
