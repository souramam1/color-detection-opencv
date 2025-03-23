from collections import deque, defaultdict
import numpy as np
from scipy.spatial import distance
import cv2
from filterpy.kalman import KalmanFilter

class TokenTracking:
    def __init__(self, shared_queue_d_to_t, history_size=10, stability_threshold=5, max_disappear=5):
        self.next_object_id = 0
        self.objects = {}  # {object_id: (x, y, color_label, kalman_filter, deque, disappear_count)}
        self.available_ids = set()  # Deregistered IDs for reuse
        self.shared_queue_d_to_t = shared_queue_d_to_t
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        self.max_disappear = max_disappear  # Frames before considering an object removed
        print("Token Tracking System Initialized")

    def create_kalman_filter(self, x, y):
        """Initialize a simple 2D Kalman Filter for object tracking."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                         [0, 1, 0, 0]])
        kf.P *= 1000  # Initial uncertainty
        kf.R *= 10  # Measurement noise
        kf.Q *= 0.1  # Process noise
        kf.x = np.array([[x], [y], [0], [0]])  # Initial state
        return kf

    def register(self, x, y, color_label):
        """Registers a new token with tracking state."""
        if self.available_ids:
            object_id = min(self.available_ids)
            self.available_ids.remove(object_id)
        else:
            object_id = self.next_object_id
            self.next_object_id += 1
        
        kalman_filter = self.create_kalman_filter(x, y)
        self.objects[object_id] = (x, y, color_label, kalman_filter, deque([1], maxlen=self.history_size), 0)

    def deregister(self, object_id):
        """Removes a token from tracking."""
        if object_id in self.objects:
            del self.objects[object_id]
            self.available_ids.add(object_id)  # Allow ID reuse

    def update(self, classifications):
        """Updates tracking state based on new detections."""
        matched_objects = set()
        current_detections = {}

        # Extract detection data
        for classification in classifications:
            (box_coords, (x, y), color_label) = classification
            current_detections[(x, y, color_label)] = box_coords

        # Match existing objects to new detections
        for object_id, (ox, oy, o_label, kf, o_deque, disappear_count) in list(self.objects.items()):
            kf.predict()  # Predict next state
            
            best_match = None
            min_distance = float("inf")
            for (x, y, color_label), _ in current_detections.items():
                if o_label == color_label:
                    dist = distance.euclidean((x, y), (ox, oy))
                    if dist < 100 and dist < min_distance:  # Adjust threshold if needed
                        min_distance = dist
                        best_match = (x, y, color_label)

            if best_match:
                x, y, color_label = best_match
                kf.update(np.array([[x], [y]]))  # Update filter
                self.objects[object_id] = (x, y, color_label, kf, o_deque, 0)
                o_deque.append(1)  # Mark as seen
                matched_objects.add(object_id)
                del current_detections[best_match]

        # Register new objects
        for (x, y, color_label), _ in current_detections.items():
            self.register(x, y, color_label)

        # Mark unseen objects and deregister if missing for too long
        for object_id in list(self.objects.keys()):
            if object_id not in matched_objects:
                ox, oy, o_label, kf, o_deque, disappear_count = self.objects[object_id]
                o_deque.append(0)  # Mark as unseen
                disappear_count += 1
                self.objects[object_id] = (ox, oy, o_label, kf, o_deque, disappear_count)

                # Remove if unseen for too long
                if disappear_count > self.max_disappear:
                    self.deregister(object_id)

    def detect_events(self):
        """Detects newly placed or removed tokens based on stable presence."""
        placed_tokens = []
        removed_tokens = []

        for object_id, (x, y, color_label, kf, obj_deque, disappear_count) in self.objects.items():
            stability_score = sum(obj_deque)

            # Detect new placements
            if len(obj_deque) == self.history_size and stability_score >= self.stability_threshold:
                placed_tokens.append((object_id, x, y, color_label))

            # Detect removals
            if disappear_count == self.max_disappear:
                removed_tokens.append((object_id, color_label))

        return placed_tokens, removed_tokens

    def run(self):
        """Main loop for tracking and event detection."""
        while True:
            classifications = self.shared_queue_d_to_t.get(block=True)
            self.update(classifications) 

            placed_tokens, removed_tokens = self.detect_events()

            if placed_tokens:
                print(f"New tokens placed: {placed_tokens}")
            if removed_tokens:
                print(f"Tokens removed: {removed_tokens}")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
