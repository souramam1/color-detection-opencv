import os
import sys
import cv2
import time
import numpy as np
from camera_01 import CameraLabelling
from pickle import load


module_path = os.path.abspath(r"C:\Users\MaiaRamambason\OneDrive - Imperial College London\Desktop\Year5\UCL_BIP\Object_Detection_Trial\color-detection-opencv\Image_Processing_Improvements\token-detection-01\src")
if module_path not in sys.path:
    print("Adding module path to sys.path")
    sys.path.append(module_path)

import white_patch_capture   
from white_patch_capture import WhitePatchCapture

class ModelTester:
    
    def __init__(self, output_folder):
        self.knn = None
        self.camera = CameraLabelling()
        self.white_patch_capture = WhitePatchCapture()

    def process_frame(self, frame):
        """Process the frame using the KNN model, to identify and mark token colours"""
        print("Processing frame...")
            # Check the number of channels in the input frame
        if len(frame.shape) == 2:
            print("Input frame is grayscale.")
        elif len(frame.shape) == 3:
            print(f"Input frame has {frame.shape[2]} channels.")
        
        # Placeholder for your actual image processing code
        
        #White patch balancing using the white_patch_capture.py code
        image_with_rectangles, image_patch = self.white_patch_capture.whitepatch_balancing(frame)
        cv2.imshow('White Patch Calibration', image_with_rectangles)
        #cv2.waitKey(0)
        #Balanced frame
        balanced_frame = self.white_patch_capture.calculate_image_max(frame, image_patch)
        cv2.imshow("Balanced Frame", balanced_frame)
        cv2.waitKey(0)
        #Gray frame
        gray = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2GRAY)
        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny Edge Detection
        edges = cv2.Canny(blurred, 20, 100)
        cv2.imshow("Canny Edge Detection", edges)
        cv2.waitKey(0)
        #Find all canny contours
        contours_canny, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw all canny contours on frame for testing
        frame_copy = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(frame_copy, contours_canny, -1, (255, 0, 0), 1)
        
        roi = self.find_largest_roi(contours_canny)
        
        isolated_token_coords = self.isolate_roi_contours(frame,contours_canny, roi)
        
        
        
        return isolated_token_coords


    def find_largest_roi(self, contours):
        """Find the largest region of interest in the frame."""
        largest_area = 0
        roi = None
        for contour in contours:
            _, size, _ = cv2.minAreaRect(contour)
            area = size[0] * size[1]
            if area > 90000 and area > largest_area:
                largest_area = area
                x, y, w, h = cv2.boundingRect(contour)
                roi = (x, y, w, h)
                print(f"ROI: {roi}")
        return roi
    
    def isolate_roi_contours(self, test_frame, contours, roi):
        """Isolate the token contours within the region of interest."""       
        roi_x, roi_y, roi_w, roi_h = roi
        cv2.rectangle(test_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
        
        isolated_token_rectangles = []
        
        for contour in contours:
            # minAreaRect can deal with rotated rectangles
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            x, y = rect[0][0], rect[0][1]
            width, height = rect[1][0], rect[1][1]
            area = width * height
            
            if 300 < area <= 800:
                if roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h:
                    isolated_token_rectangles.append(rect)
                                
        return isolated_token_rectangles
        
            
    def classify_and_label_tokens(self, camera_frame, isolated_token_rectangles):
        """Classify each isolated token using the KNN model and label it with its predicted color."""
        
        # Load the trained KNN model
        model_path = r"Image_Processing_Improvements\token-detection-training\models\knn_model_2025-03-10.pkl"
        with open(model_path, "rb") as file:
            self.knn = load(file)

        # Calculate time taken for this for loop
        start_time = time.time()
        # Iterate through each detected token
        for rect in isolated_token_rectangles:
            
            box = cv2.boxPoints(rect)  # Get the four corner points of the rotated rect
            box = np.int32(box)  # Convert to integer
            x, y = int(rect[0][0]), int(rect[0][1])  # Center of the rectangle

            # Get width and height of the bounding box
            width, height = int(rect[1][0]), int(rect[1][1])

            # Define destination points for perspective transform
            dst_pts = np.array(
                [[0, height-1],
                 [0, 0], 
                 [width-1, 0],
                 [width-1, height-1]], dtype="float32")

            # Order the points to correctly apply perspective transform
            rect_pts = np.array(sorted(box, key=lambda p: (p[1], p[0])), dtype="float32")  # Sort by y, then x

            # Compute the transformation matrix
            matrix = cv2.getPerspectiveTransform(rect_pts, dst_pts)

            # Apply the transformation to get a properly rotated ROI
            token_roi = cv2.warpPerspective(camera_frame, matrix, (width, height))

            # Convert to HSV
            hsv_roi = cv2.cvtColor(token_roi, cv2.COLOR_BGR2HSV)
            
            
            features = self.identify_token_features(hsv_roi,token_roi)

            # Predict the color label using the KNN model
            predicted_label = self.knn.predict(features)[0]


            # Draw the label on the frame
            cv2.drawContours(camera_frame, [box], 0, (0, 255, 0), 2)  # Green rectangle around the token
            cv2.putText(camera_frame, predicted_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        end_time = time.time()
        classifying_t = end_time - start_time
        print(f"classifying time is: {classifying_t}")
        return camera_frame

    def identify_token_features(self, hsv_roi,token_roi):
        # Compute the mean HSV and RGB values within the token
        mean_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0)
        mean_rgb = np.mean(token_roi.reshape(-1, 3), axis=0)

        # Create feature array: [Hue, Saturation, Value, Red, Green, Blue]
        features = np.hstack((mean_hsv, mean_rgb)).reshape(1, -1)
        return features

        
        

    def run(self):

        # Taking image of board
        print("TAKING IMAGE IN 2 SECONDS... GET READY!")
        time.sleep(2)
        try:
            frame = self.camera.capture_frame()
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return
        
        frame = self.camera.capture_frame()

        if frame is not None:
            isolated_token_coords = self.process_frame(frame)
            classified_frame = self.classify_and_label_tokens(frame,isolated_token_coords)
            cv2.imshow("classified frame", classified_frame)
        else:
            print("Failed to capture frame.")

if __name__ == "__main__":
    while True:   
        output_folder = "Image_Processing_Improvements/token-detection-training/labelled_data"  # Change this to your actual output folder path
        test_kernel = ModelTester(output_folder)
        print("TokenLabeller instance created.")
        test_kernel.run()
