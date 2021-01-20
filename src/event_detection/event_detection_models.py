import numpy as np
from tqdm import tqdm


class DistanceThreshold:

    def __init__(self):
        self.reference_step = 30
        self.motion_distance_threshold = 20
        self.stationary_distance_threshold = 2
        self.moving = False

    def euclidean_distance(self, point_1, point_2):
        return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5

    def detect_events(self, coordinates):
        num_frames = len(coordinates)
        coordinates = np.array(coordinates)

        events = []
        progress_bar = tqdm(total=num_frames, unit=' frames')
        frame_idx = 0

        while frame_idx < num_frames:
            reference_frame_idx = min(num_frames - 1, frame_idx + self.reference_step)
            if not self.moving:
                if self.euclidean_distance(coordinates[frame_idx], coordinates[reference_frame_idx]) > self.motion_distance_threshold:
                    events.append((frame_idx, True, coordinates[frame_idx].tolist()))
                    self.moving = True
            else:
                if self.euclidean_distance(coordinates[reference_frame_idx], coordinates[frame_idx]) <= self.stationary_distance_threshold:
                    events.append((frame_idx, False, coordinates[frame_idx].tolist()))
                    self.moving = False
                    frame_idx = reference_frame_idx

            progress_bar.n = frame_idx
            progress_bar.refresh()
            frame_idx += 1
        progress_bar.close()

        return events
