from collections import deque, defaultdict
import numpy as np
from scipy.spatial import distance
import queue
import cv2
from token_detection import TokenDetectionSystem

class TokenTracking:
    def __init__(self, shared_queue, history_size=5):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.token_history = defaultdict(lambda: deque(maxlen=history_size))
        self.shared_queue = shared_queue
        self.max_dissapeared = history_size

    def register(self, centroid, color):
        self.objects[self.next_object_id] = (centroid, color)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, classifications):
        """ Updates the tracker with new detections. """
        if len(classifications) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        new_centroids = np.array([x[1] for x in classifications])
        new_colors = [x[2] for x in classifications]
        print(f"new_centroids: {new_centroids} new_colors: {new_colors}")

        if len(self.objects) == 0:
            for i in range(len(new_centroids)):
                self.register(new_centroids[i], new_colors[i])
        else:
            object_ids = list(self.objects.keys())
            existing_centroids = np.array([self.objects[oid][0] for oid in object_ids])

            D = distance.cdist(existing_centroids, new_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)

            used_rows, used_cols = set(), set()
            max_distance = 10
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row,col] > max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = (new_centroids[col], new_colors[col])
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(new_centroids[col], new_colors[col])

        current_counts = defaultdict(int)
        for _, color in self.objects.values():
            current_counts[color] += 1

        for color, count in current_counts.items():
            self.token_history[color].append(count)

        for color in self.token_history:
            if color not in current_counts:
                self.token_history[color].append(0)

        return self.objects

    def get_smoothed_counts(self):
        """ Returns smoothed token counts. """
        return {color: int(np.mean(list(counts))) for color, counts in self.token_history.items()}

    def run(self):
        """ Runs tracking system in an infinite loop"""
        while True:
            classifications = self.shared_queue.get(block=True)
            print(f"tracking received: {classifications} of length {len(classifications)}")
            self.update(classifications)
            smoothed_counts = self.get_smoothed_counts()
            print(f"Smoothed Token Counts: {smoothed_counts}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python token_tracker.py <history_size>")
        sys.exit(1)

    history_size = int(sys.argv[1])
    shared_queue = queue.Queue()

    tracker = TokenTracking(shared_queue, history_size)
    tracker.run()
