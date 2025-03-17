import time
import queue
import threading
from token_tracker import TokenTracking
from token_detection import TokenDetectionSystem

# Shared queue between detection and tracking
shared_queue = queue.Queue()

# Instantiate the token detection system and trackerq
detector = TokenDetectionSystem(shared_queue=shared_queue)  
tracker = TokenTracking(shared_queue=shared_queue)

# Run both in separate threads
detection_thread = threading.Thread(target=detector.run, daemon=True)
tracking_thread = threading.Thread(target=tracker.run, daemon=True)

detection_thread.start()
tracking_thread.start() 


while True:
    time.sleep(10)
