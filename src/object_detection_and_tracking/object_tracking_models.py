from collections import OrderedDict, defaultdict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, num_frames_to_deregister=60):
        self.next_tracker_id = 1
        self.object_coordinates = OrderedDict()
        self.num_frames_to_deregister = num_frames_to_deregister
        self.num_frames_disappeared = defaultdict(lambda: 0)

    def register(self, coordinates):
        self.object_coordinates[self.next_tracker_id] = tuple(coordinates)
        self.num_frames_disappeared[self.next_tracker_id] = 0
        self.next_tracker_id += 1

    def deregister(self, tracker_id):
        del self.object_coordinates[tracker_id]
        del self.num_frames_disappeared[tracker_id]

    def update(self, new_coordinates):

        current_coordinates = np.array([self.object_coordinates[tracker_id] for tracker_id in self.object_coordinates.keys()])
        new_coordinates = np.array(new_coordinates)

        if len(new_coordinates) == 0:
            for tracker_id_to_deregister in self.object_coordinates.keys():
                self.num_frames_disappeared[tracker_id_to_deregister] += 1
                if self.num_frames_disappeared[tracker_id_to_deregister] > self.num_frames_to_deregister:
                    self.deregister(tracker_id_to_deregister)

        elif len(current_coordinates) == 0:
            for coordinates in new_coordinates:
                self.register(coordinates)

        else:
            tracker_ids = list(self.object_coordinates.keys())

            pairwise_distances = dist.cdist(current_coordinates, new_coordinates)

            current_unmatched_idxes = set(range(len(current_coordinates)))
            new_unmatched_idxes = set(range(len(new_coordinates)))

            for i in range(min(len(current_coordinates), len(new_coordinates))):
                min_idx = np.argmin(pairwise_distances)
                current_matched_idx, new_matched_idx = min_idx // len(new_coordinates), min_idx % len(new_coordinates)

                self.object_coordinates[tracker_ids[current_matched_idx]] = tuple(new_coordinates[new_matched_idx])
                self.num_frames_disappeared[tracker_ids[current_matched_idx]] = 0
                current_unmatched_idxes.remove(current_matched_idx)
                new_unmatched_idxes.remove(new_matched_idx)

                pairwise_distances[current_matched_idx, :] = np.inf
                pairwise_distances[:, new_matched_idx] = np.inf

            if len(current_unmatched_idxes) > 0:
                tracker_ids_to_deregister = []
                for current_unmatched_idx in current_unmatched_idxes:
                    self.num_frames_disappeared[tracker_ids[current_unmatched_idx]] += 1
                    if self.num_frames_disappeared[tracker_ids[current_unmatched_idx]] > self.num_frames_to_deregister:
                        tracker_ids_to_deregister.append(tracker_ids[current_unmatched_idx])

                for tracker_id_to_deregister in tracker_ids_to_deregister:
                    self.deregister(tracker_id_to_deregister)

            elif len(new_unmatched_idxes) > 0:
                for new_unmatched_idx in new_unmatched_idxes:
                    self.register(new_coordinates[new_unmatched_idx])

    def get_objects_dictionary(self):
        objects = []
        for tracker_id in self.object_coordinates.keys():
            objects.append(
                {
                    'tracker_id': tracker_id,
                    'coordinates': self.object_coordinates[tracker_id]
                }
            )
        return objects
