import cv2
from picamera2 import Picamera2
from PIL import Image
import time
from util import get_limits

yellow = [0, 255, 255]  # yellow in BGR 
red = [0,0,255] # red in BGR
blue = [255,0,0] # blue in BGR

# Initialize Picamera2
picam2 = Picamera2()
start_time = time.time()
# Configure the camera for preview
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
# End timing for configuration
end_time = time.time()
# Calculate and display the time taken
config_time = end_time - start_time
print(f"Camera configuration took {config_time:.2f} seconds.")
# Start the camera
picam2.start()
time.sleep(2)  # Allow time for the camera to initialize

try:
    while True:
        # Capture a frame
        frame = picam2.capture_array() #PiCamera frames are in RGB format instead of the BGR format that the webcams use!!!
        # Convert to BGR
        frame_bgr = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        # Convert to HSV
        hsvImage = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lowerLimit, upperLimit = get_limits(color=blue) #

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

except KeyboardInterrupt:
    print("Live feed interrupted by user.")

finally:
    # Cleanup
    cv2.destroyAllWindows()
    picam2.stop()
    picam2.close()
