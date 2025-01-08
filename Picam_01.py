import numpy as np
import cv2
from picamera2 import Picamera2
from PIL import Image
import time
from collections import deque

class Camera:
    def __init__(self, resolution=(640, 480), format="RGB888"):
        self.picam2 = Picamera2()
        self.resolution = resolution
        self.format = format
        self.configure_camera()

    def configure_camera(self):
        """Configure the camera with specified resolution and format"""
        start_time = time.time()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": self.format, "size": self.resolution}))
        end_time = time.time()
        config_time = end_time - start_time
        print(f"Camera configuration took {config_time:.2f} seconds.")

    def start(self):
        """Start the camera"""
        self.picam2.start()
        time.sleep(2)  # Allow time for the camera to initialize

    def capture_frame(self):
        """Capture a frame from the camera and return it in BGR format"""
        frame = self.picam2.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR format

    def stop(self):
        """Stop the camera"""
        self.picam2.stop()
        self.picam2.close()

class ImageProcessor:
    def __init__(self, color_name, smoothing_window_size=5):
        self.color_name = color_name
        self.smoothing_window_size = smoothing_window_size
        self.kernel = np.ones((5, 5), "uint8")  # Kernel for dilation
        self.object_counts = deque(maxlen=smoothing_window_size)

        # Hardcoded HSV limits for red, green, and blue
        self.color_ranges = {
            "red": (np.array([136, 87, 111], np.uint8), np.array([180, 255, 255], np.uint8)),
            "green": (np.array([25, 52, 72], np.uint8), np.array([102, 255, 255], np.uint8)),
            "blue": (np.array([94, 80, 2], np.uint8), np.array([120, 255, 255], np.uint8))
        }
        if color_name not in self.color_ranges:
            raise ValueError(f"Invalid color name: {color_name}. Choose from {list(self.color_ranges.keys())}.")

    def process_frame(self, frame):
        """Process the frame and return the image with bounding box if found"""
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_limit, upper_limit = self.color_ranges[self.color_name]
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

        mask = cv2.dilate(mask, self.kernel)  # Dilate the mask to fill holes

        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Draw bounding box

        return frame

    def get_smoothed_count(self, object_count):
        """Smooth the object count using a moving average"""
        self.object_counts.append(object_count)
        return int(np.mean(self.object_counts))

class CameraApp:
    def __init__(self, color_name):
        self.camera = Camera()
        self.processor = ImageProcessor(color_name=color_name)

    def run(self):
        """Main method to start the live feed and process frames"""
        self.camera.start()

        try:
            while True:
                frame = self.camera.capture_frame()
                processed_frame = self.processor.process_frame(frame)

                # Count objects in the frame
                hsv_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
                lower_limit, upper_limit = self.processor.color_ranges[self.processor.color_name]
                mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                object_count = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 300:  # Only count larger contours
                        object_count += 1
                
                # Show the smoothed object count
                smoothed_count = self.processor.get_smoothed_count(object_count)
                cv2.putText(processed_frame, f"Detected: {smoothed_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the processed frame
                cv2.imshow('Processed Frame', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Live feed interrupted by user.")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        cv2.destroyAllWindows()
        self.camera.stop()

if __name__ == "__main__":
    app = CameraApp(color_name="blue")  # Set color for detection ("red", "green", or "blue")
    app.run()
