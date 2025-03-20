import numpy as np
import cv2
from scipy.spatial import KDTree


class ContourProcessing:
    
    def __init__(self, webcam_index=1):
        self.webcam = cv2.VideoCapture(webcam_index)
        
    def apply_gaussian_blur(self, gray_frame):
        return cv2.GaussianBlur(gray_frame, (5, 5), 0)

    def apply_canny_edge_detection(self, blurred):
        edges = cv2.Canny(blurred, 20, 100)
        cv2.imshow("Canny Edge Detection", edges)
        return edges

    def find_contours(self, edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_largest_roi(self, contours):
        ''' Finds outer most rectangle, defined as being the edges of the "battery"

            Parameters:
                Input: All canny detected contours as list of NumPy array
            
            Returns:
                Output: X,Y,W,H coordinates of the rectangle in a list.
        '''
        largest_area = 0
        roi = None
        for contour in contours:
            _, size, _ = cv2.minAreaRect(contour)
            area = size[0] * size[1]
            if area > 40000 and area > largest_area:
                largest_area = area
                x, y, w, h = cv2.boundingRect(contour)
                roi = (x, y, w, h)
                print(f"ROI: {roi}")
        return roi
    
    def isolate_roi_contours(self, contours: list[np.ndarray], roi: list[int,int,int,int]):
        """Identifies contour coordinates of rectangular tokens within a given ROI
        
            Parameters: 
                Input: All frame contours as a list of NumPy arrays, detected by canny edge detection
            
            Returns:
                Output: Box corner coordinates, as a list of the rectangles identified as tokens within the ROI
        
        """ 
        # Check if ROI is valid
        if roi is None or not isinstance(roi, (list, tuple)) or len(roi) != 4:
            print("Warning: Invalid ROI provided. Skipping contour isolation.")
            return []  # Return an empty list if ROI is invalid

        roi_x, roi_y, roi_w, roi_h = roi
  
            
        roi_x, roi_y, roi_w, roi_h = roi
        
        isolated_token_rectangles = []
        
        for contour in contours:
            # minAreaRect can deal with rotated rectangles
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            x, y = rect[0][0], rect[0][1]
            width, height = rect[1][0], rect[1][1]
            area = width * height
            
            # Skip if one side is more than 3 times the other - avoids slivers of colour from shadows
            if max(width, height) > 3 * min(width, height):
                print(f"Skipping rectangle at ({x}, {y}) due to extreme aspect ratio.")
                continue
             
            if 300 < area <= 900:
                if roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h:
                    print(f"Token detected at ({x}, {y}) with area {area}")
                    if self.non_identical_check(rect, isolated_token_rectangles):
                        isolated_token_rectangles.append(rect)
                    else:
                        print(f"Skipping rectangle at ({x}, {y}) due to proximity to existing token.")
                                
        return isolated_token_rectangles
    
    def non_identical_check(self, rect: list, isolated_token_rectangles: list) -> bool:  
        ''' Checks if rect is at least 5 units away from all stored rectangles using KDTree.
        
        Parameters:
            rect: list: The rectangle to check
            isolated_token_rectangles: list: The list of rectangles to check against
                
        Returns:
            bool: True if rect is at least 5 units away from all stored rectangles, False otherwise
        '''
        if not isolated_token_rectangles:  # No previous rectangles, add directly
            return True
        
        # Extract existing (x, y) points from isolated_token_rectangles
        points = [(r[0][0], r[0][1]) for r in isolated_token_rectangles]
        
        # Create KDTree
        tree = KDTree(points)
        
        # Query the nearest neighbor within radius 5
        x, y = rect[0][0], rect[0][1]
        neighbors = tree.query_ball_point((x, y), r=5)
        
        # If no points found within 5 units, it is isolated
        return len(neighbors) == 0
        

    def show_result(self, frame: np.ndarray , caption: str):
        ''' Displays frame feed along with caption input
        
        Parameters:
            frame (np.ndarray): Input frame
            caption (string): Description of what is shown in the frame
            
        '''
        cv2.imshow(f"{caption}", frame)


    def process_frame(self, frame: np.ndarray, image_patch: np.ndarray) -> list:
        '''Processes camera frame and returns list of isolated rectangular token box coordinates
        
        Parameters:
            frame (np.ndarray): The input frame, a NumPy array, image_patch: white patch used for balancing before edge detection and classification
            
        Returns:
            isolated_token_coords: the list of corner points corresponding to tokens identified as being within the region of interest
        
        '''
        # Perform white patch balancing before any contour detection
        balanced_frame = self.white_balance(frame, image_patch)
        cv2.imshow("balanced frame", balanced_frame)
        # find canny contours from frame
        canny_contours = self.identify_contours(balanced_frame)
        # isolate token coordinates within a roi
        isolated_token_coords = self.identify_token_coords(canny_contours)
        
        return isolated_token_coords
        
        
    
    def identify_contours(self, frame: np.ndarray) -> list[np.ndarray]:
        '''Processes white patch balanced camera frame and identifies canny edges, returns canny contours
        
        Parameters:
            frame (np.ndarray): The input frame, a NumPy array
            
        Returns:
            contours of the detected Canny edges as a list of NumPy arrays'''
            
        #Convert input frame to grayscale    
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #generates gaussian blur on grayscale image
        blurred = self.apply_gaussian_blur(gray_frame)
        #generates canny edge detection on blurred image
        edges = self.apply_canny_edge_detection(blurred)
        #finds contours on canny edge detection
        contours_canny = self.find_contours(edges)
        
        return contours_canny
    
    def identify_token_coords(self, contours: list[np.ndarray]) -> list:
        '''Processes canny contours and filters them to retrive isolated coordinates of token corner points
        
        Parameters:
            contours:  list[np.ndarray] The list of NumPy arrays details all found Canny contours to be filtered 
            so that tokens can be identified
            
        Returns:
            contours of the detected Canny edges as a list of NumPy arrays'''
        # defines the eges of the roi from the frame
        roi = self.find_largest_roi(contours)
        # identifies tokens within that roi and returns the list of their box coordinates
        token_rect_coords = self.isolate_roi_contours(contours, roi)
        print(f"Token coordinates: {token_rect_coords}")
        
        return token_rect_coords
             
    def white_balance(self,frame, image_patch):
        ''' Applies whitepatch balancing to input frame based on image patch from calibration phase
        
        Parameter: 
            frame: np.ndarray: image frame to be balanced
            image_patch: np.ndarray : small chunk of image (ideally white background)
            
        Returns:
            frame: np.ndarray: whitepatch balanced frame
            
        '''
        # Perform white patch balancing, using the frame and patch
        image_max = (frame * 1.0 / image_patch.max(axis=(0, 1))).clip(0, 1)
        image_max = (image_max * 255).astype(np.uint8)   
        
        return image_max
