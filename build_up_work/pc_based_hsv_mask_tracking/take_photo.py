import cv2
import os

"""Take image for testing purposes
Saved as: captured_image.jpg in the same directory"""

class WebcamCapture:
    def __init__(self, webcam_index=1):
        self.webcam = cv2.VideoCapture(webcam_index)
        if not self.webcam.isOpened():
            raise Exception("Could not open webcam")

    def capture_frame(self):
        """Capture a frame from the webcam."""
        ret, frame = self.webcam.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame

    def save_frame(self, frame, filename="captured_image.jpg"):
        """Save the captured frame to a file."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        cv2.imwrite(file_path, frame)
        print(f"Image saved as {file_path}")

    def release_webcam(self):
        """Release the webcam."""
        self.webcam.release()

    def capture_and_save_image(self, filename="captured_image.jpg"):
        """Capture an image from the webcam and save it to a file."""
        frame = self.capture_frame()
        if frame is not None:
            self.save_frame(frame, filename)
        self.release_webcam()

if __name__ == "__main__":
    webcam_capture = WebcamCapture()
    webcam_capture.capture_and_save_image("captured_image.jpg")