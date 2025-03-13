import os
import sys
import cv2
import time
import numpy as np
from pickle import load


class KnnInterfacer:
    
    def __init__(self):
        self.knn = None
        self.scaler = None
       
    def classify_and_label_tokens(self, frame: (np.ndarray) , rectangles: list, model_path_in: str, scaler_path_in: str):
        """Classify each isolated token using the KNN model and label it with its predicted color - then display the frame.
        
            Parameters:
                frame: Input frame to append labels to and return for display/feedback, a NumPy array
                rectangles: List of the isolated token coordinates within the ROI, used to isolate frames for classification
                model_path_in: Path to learning model
                scaler_path_in: Path to scaler model
                   
            Returns:
                frame: frame displaying tokens with associated labels, a NumPy array
                classifications: list of classified tokens with each token : box, (centre_x,centre_y), predicted_label (str)
        """
        
        # Load the trained KNN model
        model_path = model_path_in
        with open(model_path, "rb") as file:
            self.knn = load(file)
         
        #Load the saved scaler   
        scaler_path = scaler_path_in
        with open(scaler_path, "rb") as file:
            self.scaler = load(file)
            

        # Calculate time taken for this for loop
        start_time = time.time()
        
        classifications = [] #store classification results
        
        # Iterate through each detected token
        for rect in rectangles:
            
            box = cv2.boxPoints(rect)  # Get the four corner points of the rotated rect
            box = np.int32(box)  # Convert to integer
            centre, size, angle = rect
            x, y = int(centre[0]), int(centre[1])  # Center of the rectangle
            width, height = int(size[0]), int(size[1])

            # Get width and height of the bounding box
           

            # Define destination points for perspective transform
            dst_pts = np.array(
                [[0, height-1],
                 [0, 0], 
                 [width-1, 0],
                 [width-1, height-1]
                 ], dtype="float32")


            # Compute the transformation matrix
            matrix = cv2.getPerspectiveTransform(np.float32(box), dst_pts)

            # Apply the transformation to get a properly rotated ROI
            isolated_token = cv2.warpPerspective(frame, matrix, (width, height))
            
            # Check if the extracted token is empty
            if isolated_token is None or isolated_token.size == 0:
                print("Warning: Empty or invalid token detected, skipping...")
                continue  # Skip this iteration

            # Convert to HSV
            hsv_roi = cv2.cvtColor(isolated_token, cv2.COLOR_BGR2HSV)
            
            # Extract the features from the isolated token, to then classify
            features = self.identify_token_features(hsv_roi, isolated_token)
            normalised_features = self.scaler.transform(features)
            normalised_features[:,0] *= 3 #Apply the same hue scaling
        

            # Check for NaN values in normalised_features
            if np.any(np.isnan(normalised_features)):
                print("Warning: NaN values detected in features, skipping this token...")
                continue  # Skip this iteration if any NaN value is found

            # Predict the color label using the KNN model
            predicted_label = self.knn.predict(normalised_features)[0]
            print(f"predicted label is: {predicted_label}")
            
            # Store the classification results
            classifications.append((box, (x, y), predicted_label))
            
            # For timing purposes
            end_time = time.time()
            classifying_t = end_time - start_time
            print(f"classifying time is: {classifying_t}")
        
        for box, (x,y), predicted_label in classifications:
            # Draw the label on the frame
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)  # Green rectangle around the token
            cv2.putText(frame, predicted_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame, classifications

    def identify_token_features(self, hsv_roi,token_roi):
        ''' Extract and return token hsv, and rgb features 
        
            Parameters: 
                hsv_roi: NumPy array (hsv), a bounded section of overall frame, limited to the detected token, to have its features extracted
                token_roi: NumPy array (rgb), a bounded section of overall frame, limited to the detected token, to have its features extracted
                
            Returns:
                features: 2D NumPy array (dimensions (1,6)) containing H,S,V,R,G,B features for detected token

        '''
        #Extract central region (to avoid background influence - also to match feature extractor for classification)
        h, w, _ = token_roi.shape  # Get dimensions of the token ROI

        center_region = token_roi[h//4: 3*h//4, w//4: 3*w//4]
        center_hsv = hsv_roi[h//4: 3*h//4, w//4: 3*w//4]
        
        # Compute the mean HSV and RGB values within the token
        mean_hsv = np.mean(center_hsv, axis=(0,1))
        mean_rgb = np.mean(center_region, axis=(0,1))
        
        print(f"mean hsv: {mean_hsv}")
        print(f"mean_rgb: {mean_rgb}")

        # Create token's feature array: [Hue, Saturation, Value, Red, Green, Blue]
        features = np.hstack((mean_hsv, mean_rgb)).reshape(1, -1)
        print(f"features: {features}")
        
        return features

        
        
