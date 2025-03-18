import os
import sys
import cv2
import time
import numpy as np
from camera_01 import CameraLabelling

module_path = os.path.abspath(r"C:\Users\MaiaRamambason\OneDrive - Imperial College London\Desktop\Year5\UCL_BIP\Object_Detection_Trial\color-detection-opencv\Image_Processing_Improvements\token-detection-01\src")
if module_path not in sys.path:
    print("Adding module path to sys.path")
    sys.path.append(module_path)
    print(f"sys.path: {sys.path}")
    
import white_patch_capture   
from white_patch_capture import WhitePatchCapture

class TokenLabeller:
    
    def __init__(self, color, output_folder):
        self.color = color.lower()
        self.output_folder = output_folder
        self.camera = CameraLabelling()
        self.white_patch_capture = WhitePatchCapture()
        self.color_folder = os.path.join(output_folder, f"{self.color}_tokens")
        os.makedirs(self.color_folder, exist_ok=True)
        
# Proceed with adding images to self.color_folder

    def get_next_filename(self):
        """Get the next available filename for the token image."""
        existing_files = os.listdir(self.color_folder)
        existing_indices = [int(f.split('_')[2].split('.')[0]) for f in existing_files if f.startswith(f"t_{self.color}_")]
        next_index = 1
        while next_index in existing_indices:
            next_index += 1
        return f"t_{self.color}_{next_index:02d}.jpg"

    def process_frame(self, frame, image_patch):
        """Process the frame to isolate tokens. This is a placeholder for your actual processing code."""
        print("Processing frame...")
        
        # Check the number of channels in the input frame
        if len(frame.shape) == 2:
            print("Input frame is grayscale.")
        elif len(frame.shape) == 3:
            print(f"Input frame has {frame.shape[2]} channels.")
        
        #White balancing frame
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
        
        # Testing - draw all Canny contours on frame
        frame_copy = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(frame_copy, contours_canny, -1, (255, 0, 0), 1)
        
        # Find outer rectangle that bounds the board
        roi = self.find_largest_roi(contours_canny)
        
        #Returns list token coordinates identified inside the ROI
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
                    #cv2.drawContours(frame_copy, [box], 0, (0, 0, 255), 2)
                    #cv2.putText(test_frame, f"{area:.0f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                    
                    # Every frame, all tokens of size are found and added to a list
                    # List of tokens is emptied each new frame.
                    isolated_token_rectangles.append(rect)
                    
        return isolated_token_rectangles
        
        
    def save_token_images(self, frame, rectangles):
        """Save the token images based on the rectangles."""
        for rect in rectangles:
            
            box = cv2.boxPoints(rect)  # Get the 4 corner points
            box = np.int32(box)  # Convert to integers
            centre, size, angle = rect
            width, height = size[0], size[1]
            width, height = int(width), int(height)  # Ensure integer values

            # Define the destination points for warping (unrotated rectangle)
            dst_pts = np.array([
                [0, height-1],   # Bottom-Left
                [0, 0],          # Top-Left
                [width-1, 0],    # Top-Right
                [width-1, height-1]  # Bottom-Right
            ], dtype="float32")

            # Get the perspective transform matrix
            matrix = cv2.getPerspectiveTransform(np.float32(box), dst_pts)

            # Warp the image to get the rotated ROI
            isolated_token = cv2.warpPerspective(frame, matrix, (width, height))
            
            filename = self.get_next_filename()
            filepath = os.path.join(self.color_folder, filename)
            cv2.imwrite(filepath, isolated_token)
            print(f"Saved token image: {filepath}")
            
    def whitepatch_balancing(self,frame):
        white_patch_balancer = WhitePatchCapture()
        image_with_rectangles, image_patch = white_patch_balancer.whitepatch_balancing(frame)
        cv2.imshow('White Patch Calibration', image_with_rectangles)
        return image_patch

    def run(self):
        print(f"Starting token labelling for color: {self.color}")
        
        
        #Whitepatch frame for balancing
        print("Capturing preliminary frame for white patch balancing in 2 SECONDS..")
        time.sleep(2)
        try:
            prelim_frame = self.camera.capture_frame()
            if prelim_frame is None:
                print("Error: Preliminary frame capture failed.")
                return
            image_patch = self.whitepatch_balancing(prelim_frame)
            print("White patch balancing applied")
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return
        
        #Delay before capturing the main frame
        print("10 seconds before capturing the main frame...")
        time.sleep(8)
        print("In 2 SECONDS!!")
        time.sleep(2)
        try:
            frame = self.camera.capture_frame()
            if frame is None:
                print("Error: Failed to capture main frame.")
                return
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return
        
        #save the frame to the output folder
        cv2.imwrite(os.path.join(self.output_folder, f"frame_{time.time()}.jpg"), frame)
        if frame is not None:
            rectangles = self.process_frame(frame, image_patch)
            self.save_token_images(frame, rectangles)
        else:
            print("Failed to capture frame.")

if __name__ == "__main__":
    while True:
        color = "orange"
        valid_colors = ["yellow", "cyan", "green", "magenta", "orange", "blue"]
        
        if color not in valid_colors:
            print(f"Invalid color: {color}. Please enter one of the following: {', '.join(valid_colors)}")
        else:
            print(f"Color accepted: {color}")
            
            output_folder = "Image_Processing_Improvements/token-detection-training/labelled_data"  # Change this to your actual output folder path
            labeller = TokenLabeller(color, output_folder)
            print("TokenLabeller instance created.")
            labeller.run()
