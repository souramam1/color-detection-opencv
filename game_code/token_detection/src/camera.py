import cv2

class Camera:
    def __init__(self, webcam_index=1):
        self.webcam = cv2.VideoCapture(webcam_index)

    def capture_frame(self):
        """Capture a frame from the webcam.
        
            Returns:
                frame: NumPy array
        """
        ret, frame = self.webcam.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame

    def cleanup(self):
        """Release the webcam and close all OpenCV windows."""
        self.webcam.release()
        cv2.destroyAllWindows()