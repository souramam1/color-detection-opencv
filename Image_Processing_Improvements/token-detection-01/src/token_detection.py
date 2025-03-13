from hsv_color_detection import ColorDetection
from contour_processing import ContourProcessing
from camera import Camera
from hsv_calib import HSVCalibrator
from white_patch_capture import WhitePatchCapture
from knn_model_interfacer import KnnInterfacer


import numpy as np
import cv2

class TokenDetectionSystem:
    def __init__(self, camera_index=1):
        
        self.camera = Camera()
        self.contour_processing = ContourProcessing()
        self.color_classification = KnnInterfacer()
        self.white_patch_capture = WhitePatchCapture()
        
        

    def run(self,model_path, scaler_path):
        ''' Run the token detection system: Each run requires having no tokens on the board initially and calibrating to a white background
        
            Parameters:
                model+path/scaler_path (str): Paths to the classification models used to identify colours
                
                
            The screen will display live feedback of token detection with labels.
        '''
        # take photo of white background to obtain white patch for white balancing
        frame = self.camera.capture_frame()
    
        # displays location of rectangle segment grabbed from image for "image_patch" 
        image_with_rectangles, image_patch = self.white_patch_capture.select_image_patch(frame)
        cv2.imshow('White Patch Calibration', image_with_rectangles)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is not None:
                    #isolate token coordinates
                    isolated_token_coords = self.contour_processing.process_frame(frame, image_patch)
                    #classify tokens into colour groups 
                    labelled_frame, classifications = self.color_classification.classify_and_label_tokens(frame,isolated_token_coords,model_path,scaler_path)
                    #display classified tokens
                    #self.contour_processing.show_result(labelled_frame, "Final frame with classification")
                    
                    yield labelled_frame, classifications # Yield frame & token detections
                    
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            self.camera.cleanup()

if __name__ == "__main__":
    
    detection_system = TokenDetectionSystem()
    
    model = r"Image_Processing_Improvements\token-detection-training\models\2025-03-11-16-18_knn_model.pkl"
    scaler = r"Image_Processing_Improvements\token-detection-training\models\2025-03-11-16-18_scaler.pkl"
    
    detection_system.run(model,scaler)