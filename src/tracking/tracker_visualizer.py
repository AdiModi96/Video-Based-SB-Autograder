import cv2
import numpy
import os
import json
import paths

COLORS = {
    3: (0, 0, 255),
    4: (0, 255, 0)
}

tracked_objects_file_path = os.path.join(paths.data_folder_path, 'tracked objects', '77_bonus.json')
video_file_path = os.path.join(paths.data_folder_path, 'pre-processed', 'warped', '77_bonus.mp4')

with open(tracked_objects_file_path) as file:
    json_file = json.load(file)

video = cv2.VideoCapture(video_file_path)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)

for frame_idx in range(num_frames):
    _, frame = video.read()

    objects = json_file[frame_idx]['objects']
    for object in objects:
        for instance in object['coordinates']:
            coordinates = object['coordinates'][instance]
            frame = cv2.circle(
                frame,
                center=(int(coordinates[0]), int(coordinates[1])),
                color=COLORS[object['label']],
                radius=7,
                thickness=2,
            )
            frame = cv2.putText(
                frame,
                text='{}/{}'.format(object['label'], instance),
                org=(int(coordinates[0]), int(coordinates[1])),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.2,
                color=COLORS[object['label']],
                thickness=1
            )

    cv2.imshow('Tracked Objects', frame)
    key = cv2.waitKey(1000 // int(fps))
    if key == 27:
        break