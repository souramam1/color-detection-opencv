import cv2
from picamera2 import Picamera2
from PIL import Image
import time
from util import get_limits

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
        return frame
        # return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #UNCOMMENT IF IF PICAMERA USES RGB

    def stop(self):
        """Stop the camera"""
        self.picam2.stop()
        self.picam2.close()

class ImageProcessor:
    def __init__(self, color):
        self.color = color

    def process_frame(self, frame):
        """Process the frame and return the image with bounding box if found"""
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_limit, upper_limit = get_limits(color=self.color)
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5) # This is in BGR format
        
        return frame

class CameraApp:
    def __init__(self, color):
        self.camera = Camera()
        self.processor = ImageProcessor(color=color)

    def run(self):
        """Main method to start the live feed and process frames"""
        self.camera.start()

        try:
            while True:
                frame = self.camera.capture_frame()
                processed_frame = self.processor.process_frame(frame)
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
    app = CameraApp(color=[255, 0, 0])  # Set color for detection (blue in BGR)
    app.run()
