from collections import deque, defaultdict
import numpy as np
from scipy.spatial import distance
import queue
import cv2
from token_detection import TokenDetectionSystem

class TokenTracking:
    def __init__(self, shared_queue_d_to_t, history_size=10, stability_threshold=10):
        """
        Initialize the TokenTracking system.

        Parameters:
            shared_queue (queue.Queue): Shared queue for receiving classifications from the detection system.
            history_size (int): Length of the temporal filter for smoothing counts.
            stability_threshold (int): Number of consecutive frames required for a token to be considered stable.
        """
        self.next_object_id = 0
        self.objects = {}
        self.stable_objects = {}
        self.disappeared = {}
        self.token_history = defaultdict(lambda: deque(maxlen=history_size))
        self.stability_counter = defaultdict(int)
        self.shared_queue_d_to_t = shared_queue_d_to_t
        self.max_disappeared = stability_threshold
        self.stability_threshold = stability_threshold
        print("Token Tracking System Initialized")

    def register(self, centroid, color):
        """
        Register a new object with a given centroid and color.

        Parameters:
            centroid (tuple): The (x, y) coordinates of the object's centroid.
            color (str): The color of the object.
        """
        print(f"REGISTERING OBJECT {self.next_object_id}")
        self.objects[self.next_object_id] = (centroid, color)
        self.disappeared[self.next_object_id] = 0
        self.stability_counter[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Deregister an object by its ID.

        Parameters:
            object_id (int): The ID of the object to deregister.
        """
        print(f"DEREGISTERING OBJECT {object_id}")
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.stability_counter[object_id]

    def update(self, classifications):
        """
        Update the tracker with new detections.

        Parameters:
            classifications (list): List of classifications, where each classification is a tuple (label, centroid, color).

        Returns:
            dict: Updated objects being tracked.
        """
        if len(classifications) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        new_centroids = np.array([x[1] for x in classifications])
        new_colors = [x[2] for x in classifications]

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
                if D[row, col] > max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = (new_centroids[col], new_colors[col])
                self.disappeared[object_id] = 0
                self.stability_counter[object_id] += 1
                if self.stability_counter[object_id] == self.stability_threshold:
                    print(f"{new_colors[col].upper()} token added at {new_centroids[col]}")
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    print(f"{self.objects[object_id][1].upper()} token removed from {self.objects[object_id][0]}")
                    self.deregister(object_id)
                else:
                    self.stability_counter[object_id] = max(0, self.stability_counter[object_id] - 1)

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
        """
        Returns smoothed token counts.

        Returns:
            dict: Smoothed counts of tokens for each color.
        """
        smoothed_counts = {}
        for color, counts in self.token_history.items():
            smoothed_counts[color] = int(np.mean(list(counts)))
        return smoothed_counts

    def show_smoothed_counts(self, smoothed_counts):
        """
        Show the smoothed token counts on the frame.

        Parameters:
            smoothed_counts (dict): The smoothed counts of tokens for each color.
        """
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        y_offset = 30  # Starting Y position for the text
        color_map = {
            "yellow": (0, 255, 255),
            "magenta": (255, 0, 255),
            "cyan": (255, 255, 0),
            "green": (0, 255, 0)
        }
        for color in ["yellow", "magenta", "cyan", "green"]:
            count = smoothed_counts.get(color, 0)
            cv2.putText(frame, f"{color}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color_map[color], 2)
            y_offset += 30  # Move down for the next color count
        cv2.imshow("Token tracker frame", frame)

    def run(self):
        """
        Runs the tracking system in an infinite loop.
        """
        while True:
            classifications = self.shared_queue_d_to_t.get(block=True)
            self.update(classifications)
            smoothed_counts = self.get_smoothed_counts()
            print(f"Smoothed TOKEN COUNTS: {smoothed_counts}")
            self.show_smoothed_counts(smoothed_counts)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python token_tracker.py <history_size>")
        sys.exit(1)

    history_size = int(sys.argv[1])
    shared_queue = queue.Queue()

    tracker = TokenTracking(shared_queue, history_size)
    tracker.run()
