import json
import os

import cv2

import paths

LABEL_COLORS = {
    1: (255, 255, 255),
    2: (0, 255, 255),
    3: (0, 0, 255),
    4: (0, 255, 0)
}

video_file_path = os.path.join(paths.data_folder_path, 'raw', '77_bonus.mp4')
annotations_file_path = os.path.join(paths.data_folder_path, 'detected_and_tracked_objects', '77_bonus.json')

with open(annotations_file_path) as file:
    annotations = json.load(file)

video = cv2.VideoCapture(video_file_path)
num_frames = len(annotations)
fps = video.get(cv2.CAP_PROP_FPS)
playback_speed = 3.0

for frame_idx in range(num_frames):
    _, frame = video.read()
    frame_annotations = annotations[frame_idx]

    for object in frame_annotations:

        if object['label'] == 1 or object['label'] == 2:
            label = object['label']
            coordinates = object['coordinates']

            frame = cv2.circle(
                frame,
                center=(int(coordinates[0]), int(coordinates[1])),
                color=LABEL_COLORS[label],
                radius=7,
                thickness=2,
            )

            frame = cv2.putText(
                frame,
                text='{}'.format(label),
                org=(int(coordinates[0]), int(coordinates[1]) - 2),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=LABEL_COLORS[object['label']],
                thickness=1
            )

        else:
            label = object['label']
            for instance in object['instances']:
                instance_id = instance['tracker_id']
                coordinates = instance['coordinates']

                frame = cv2.circle(
                    frame,
                    center=(int(coordinates[0]), int(coordinates[1])),
                    color=LABEL_COLORS[object['label']],
                    radius=7,
                    thickness=2,
                )

                frame = cv2.putText(
                    frame,
                    text='{}/{}'.format(label, instance_id),
                    org=(int(coordinates[0]), int(coordinates[1]) - 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75,
                    color=LABEL_COLORS[object['label']],
                    thickness=1
                )

    cv2.imshow('Object Detection & Tracking for Video: 77_bonus', frame)
    key = cv2.waitKey(int(1000 / (playback_speed * fps)))
    if key == 27:
        break
