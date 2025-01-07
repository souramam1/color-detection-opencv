import cv2
from picamera2 import Picamera2
import time

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
        
        frame_bgr = cv2.cvtColor(frame,cv2.COLORRGB2BGR)

        # Display the frame in a window
        cv2.imshow("Live Feed", frame_bgr)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Live feed interrupted by user.")

finally:
    # Cleanup
    cv2.destroyAllWindows()
    picam2.stop()
    picam2.close()
