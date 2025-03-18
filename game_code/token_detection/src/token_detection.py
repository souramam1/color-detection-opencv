from hsv_color_detection import ColorDetection
from contour_processing import ContourProcessing
from camera import Camera
from hsv_calib import HSVCalibrator
from white_patch_capture import WhitePatchCapture
from knn_model_interfacer import KnnInterfacer
import queue
import numpy as np
import cv2

class TokenDetectionSystem:
    def __init__(self, 
                 shared_queue,
                 model=r"game_code\token_model_training\models\2025-03-11-16-18_knn_model.pkl", 
                 scaler = r"game_code\token_model_training\models\2025-03-11-16-18_scaler.pkl", 
                 camera_index=1):
        
        '''Initialise the Token Detection System

            Parameters:0
                camera_index (int): Index of the camera to be used for token detection
                model (str): Path to the classification model used to identify colours
                scaler (str): Path to the scaler used to preprocess data before classification
        '''
        
        self.camera = Camera()
        self.contour_processing = ContourProcessing()
        self.color_classification = KnnInterfacer()
        self.white_patch_capture = WhitePatchCapture()
        self.model_path = model
        self.scaler_path = scaler
        self.shared_queue = shared_queue
        
        

    def run(self):
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
                    labelled_frame, classifications = self.color_classification.classify_and_label_tokens(frame,isolated_token_coords,self.model_path,self.scaler_path)
                    #display classified tokens
                    self.contour_processing.show_result(labelled_frame, "Final frame with classification")
                    
                    #yield labelled_frame, classifications # Yield frame & token detections
                    print(f"length of classifications to be sent are: {len(classifications)}")
                    self.shared_queue.put(classifications)
                    
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            self.camera.cleanup()

if __name__ == "__main__":
    script_queue = queue.Queue()
    model = r"game_code\token_model_training\models\2025-03-11-16-18_knn_model.pkl"
    scaler =  r"game_code\token_model_training\models\2025-03-11-16-18_scaler.pkl"
    detection_system = TokenDetectionSystem(script_queue)
    detection_system.run()