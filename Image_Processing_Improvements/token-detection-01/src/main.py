from color_detection import ColorDetection
from contour_processing import ContourProcessing
from camera import Camera
from HSV_Calib_01 import HSVCalibrator
import cv2

class TokenDetectionSystem:
    def __init__(self, camera_index=1, calib_index=1):
        self.camera = Camera()
        self.contour_processing = ContourProcessing()
        self.color_detection = ColorDetection()
        if calib_index == 1:
            self.hsv_calib = HSVCalibrator()
            self.hsv_calib.run()
            self.color_detection = ColorDetection(final_hsv_calib=self.hsv_calib.calibrated_values)
            

    def run(self):
        
        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is not None:
                    contoured_frame, bgr_frame, roi = self.contour_processing.process_frame(frame)
                    self.color_detection.draw_contours(contoured_frame,roi, self.contour_processing.detected_token_contours, bgr_frame)
                    self.contour_processing.show_result(contoured_frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            self.camera.cleanup()

if __name__ == "__main__":
    detection_system = TokenDetectionSystem()
    detection_system.run()