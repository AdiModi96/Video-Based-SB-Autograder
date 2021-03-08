import json
import os
from collections import defaultdict
from collections import deque

import cv2

import paths

LABEL_COLORS = {
    1: (255, 255, 255),
    2: (0, 255, 255),
    3: (0, 0, 255),
    4: (0, 255, 0)
}

LABEL_NAME = {
    1: 'arena_center',
    2: 'robot',
    3: 'red',
    4: 'green'
}

STATE = {
    True: 'moving',
    False: 'stationary'
}

STATE_COLORS = {
    True: (0, 255, 0),
    False: (0, 0, 255)
}

video_file_path = os.path.join(paths.data_folder_path, 'raw', '77_bonus.mp4')
events_file_path = os.path.join(paths.data_folder_path, 'detected_events', '77_bonus.json')

with open(events_file_path) as file:
    events = deque(json.load(file))

video = cv2.VideoCapture(video_file_path)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)

event_update_message = ''
points = defaultdict(lambda: [])
color = (0, 0, 0)
video.set(cv2.CAP_PROP_POS_FRAMES, 300)
for frame_idx in range(300, num_frames):
    _, frame = video.read()

    if len(events) > 0 and events[0]['frame_idx'] == frame_idx:
        label = events[0]['label']
        moving = events[0]['moving']
        tracker_id = events[0]['tracker_id']

        event_update_message = '{} coin {} at frame number: {}'.format(
            LABEL_NAME[label],
            STATE[moving],
            frame_idx
        )

        points[(label, tracker_id)].append(events[0]['coordinates'])
        color = STATE_COLORS[moving]
        events.popleft()

    notification_panel = frame.copy()
    notification_panel[-50:, :] = 0
    frame = cv2.addWeighted(frame, 0.2, notification_panel, 0.8, 0.0)

    for key in points.keys():
        for i in range(1, len(points[key])):
            point_1 = (int(points[key][i - 1][0]), int(points[key][i - 1][1]))
            point_2 = (int(points[key][i][0]), int(points[key][i][1]))
            frame = cv2.line(
                frame,
                point_1,
                point_2,
                color=LABEL_COLORS[key[0]],
                thickness=2
            )

            distance = round(((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2) ** 0.5, 3)

            frame = cv2.putText(
                frame,
                text='Distance: {}'.format(distance),
                org=((point_1[0] + point_2[0]) // 2, (point_1[1] + point_2[1]) // 2),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.75,
                color=LABEL_COLORS[key[0]],
                thickness=1
            )

    frame = cv2.putText(
        frame,
        text='Notification: {}'.format(event_update_message),
        org=(50, 700),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.75,
        color=color,
        thickness=1
    )

    cv2.imshow('Event Detection for Video: 77_bonus', frame)
    # key = cv2.waitKey(1000 // int(fps))
    key = cv2.waitKey(10)
    if key == 27:
        break
