from color_detection import ColorDetection
from contour_processing import ContourProcessing
from camera import Camera
from HSV_Calib_01 import HSVCalibrator
from white_patch_capture import WhitePatchCapture


import numpy as np
import cv2

class TokenDetectionSystem:
    def __init__(self, camera_index=1, calib_index=0):
        self.camera = Camera()
        self.contour_processing = ContourProcessing()
        self.color_detection = ColorDetection()
        self.color_classification = ColorClassification()
        self.white_patch_capture = WhitePatchCapture()
        
        if calib_index == 1:
            self.hsv_calib = HSVCalibrator()
            self.hsv_calib.run()
            self.color_detection = ColorDetection(final_hsv_calib=self.hsv_calib.calibrated_values)
            

    def run(self):
        
        # Take a photo of white background to obtain white patch for white balancing
        frame = self.camera.capture_frame()
        # Define a region of interest (ROI) in the center of the captured frame
    
        roi = (57, 49, 467, 355)  # Example ROI, adjust as needed
        image_with_rectangles, image_patch = self.white_patch_capture.whitepatch_balancing(frame)
        cv2.imshow('White Patch Calibration', image_with_rectangles)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is not None:
                    contoured_frame, bgr_frame, roi = self.contour_processing.process_frame_old(frame)
                    self.color_detection.draw_contours(contoured_frame, self.contour_processing.detected_token_contours, bgr_frame, image_patch)
                    self.contour_processing.show_result(contoured_frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            self.camera.cleanup()

if __name__ == "__main__":
    
    detection_system = TokenDetectionSystem(calib_index=0)
    detection_system.run()