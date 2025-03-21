from collections import deque, defaultdict
import numpy as np
from scipy.spatial import distance
import queue
import cv2

class TokenTracking:
    def __init__(self, shared_queue_d_to_t, history_size=10, stability_threshold=5):
        self.next_object_id = 0
        self.objects = {}  # Stores {object_id: (x, y, label, deque)}
        self.available_ids = set()  # Stores deregistered IDs for reuse
        self.shared_queue_d_to_t = shared_queue_d_to_t
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        print("Token Tracking System Initialized")

    def register(self, x, y, color_label):
        """Registers a new object with a deque of max length 5."""
        if self.available_ids:
            object_id = min(self.available_ids)  # Reuse the smallest available ID
            self.available_ids.remove(object_id)
        else:
            object_id = self.next_object_id
            self.next_object_id += 1
        
        self.objects[object_id] = (x, y, color_label, deque([1], maxlen=self.history_size))

    def deregister(self, object_id):
        """Removes an object from tracking and makes its ID available for reuse."""
        if object_id in self.objects:
            del self.objects[object_id]
            self.available_ids.add(object_id)  # Mark ID as available

    def update(self, classifications):
        matched_objects = set()
        
        for classification in classifications:
            (box_coords, (x, y), color_label) = classification
            matched = False
            
            for object_id, (ox, oy, o_label, o_deque) in list(self.objects.items()):
                if o_label == color_label and distance.euclidean((x, y), (ox, oy)) < 200:  # Proximity threshold
                    o_deque.append(1)  # Mark as seen
                    self.objects[object_id] = (x, y, color_label, o_deque)
                    matched_objects.add(object_id)
                    matched = True
                    break
            
            if not matched:
                self.register(x, y, color_label)
        
        for object_id in list(self.objects.keys()):
            if object_id not in matched_objects:
                self.objects[object_id][3].append(0)  # Mark as unseen
            
            # Check deque mean every update
            if np.mean(self.objects[object_id][3]) < 0.7:  # If less than 70% of the time seen
                self.deregister(object_id)

    def get_smoothed_counts(self):
        smoothed_counts = defaultdict(int)
        for _, (_, _, color_label, obj_deque) in self.objects.items():
            if sum(obj_deque) >= self.stability_threshold:
                smoothed_counts[color_label] += 1
        return smoothed_counts

    def show_smoothed_counts(self, smoothed_counts):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        y_offset = 30
        color_map = {"yellow": (0, 255, 255), "magenta": (255, 0, 255), "cyan": (255, 255, 0), "green": (0, 255, 0)}
        for color in ["yellow", "magenta", "cyan", "green"]:
            count = smoothed_counts.get(color, 0)
            cv2.putText(frame, f"{color}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color_map[color], 2)
            y_offset += 30
        cv2.imshow("Token tracker frame", frame)

    def run(self):
        while True:
            classifications = self.shared_queue_d_to_t.get(block=True)
            self.update(classifications)
            smoothed_counts = self.get_smoothed_counts()
            print(f"Smoothed TOKEN COUNTS: {smoothed_counts}")
            self.show_smoothed_counts(smoothed_counts)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
