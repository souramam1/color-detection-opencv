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

    def find_largest_roi(self, contours, display_frame):
        ''' Finds the largest rectangle, defined as being the edges of the "battery"

            Parameters:
                contours (list): All canny detected contours as list of NumPy arrays
                display_frame (np.ndarray): The frame to draw the ROI on
                
            Returns:
                np.ndarray: The largest rotated rectangle.
        '''
        largest_area = 0
        roi_game = None
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            area = rect[1][0] * rect[1][1]
            if area > 40000 and area > largest_area:
                largest_area = area
                roi_game = box
                # Draw the ROI on the frame
                cv2.drawContours(display_frame, [box], 0, (0, 255, 0), 2)
        
        
        if roi_game is not None:
            # Calculate the top 57% and bottom 43% of the ROI
            roi_batt = np.array([
                roi_game[0],
                roi_game[1],
                roi_game[1] + 0.57 * (roi_game[2] - roi_game[1]),
                roi_game[0] + 0.57 * (roi_game[3] - roi_game[0])
            ], dtype=np.int32)
            
            roi_time = np.array([
                roi_game[0] + 0.57 * (roi_game[3] - roi_game[0]),
                roi_game[1] + 0.57 * (roi_game[2] - roi_game[1]),
                roi_game[2],
                roi_game[3]
            ], dtype=np.int32)
            
            # Draw the top 57% (roi_batt) in blue
            cv2.drawContours(display_frame, [roi_batt], 0, (255, 0, 0), 2)
            
            # Draw the bottom 43% (roi_time) in red
            cv2.drawContours(display_frame, [roi_time], 0, (0, 0, 255), 2)
        
        cv2.imshow("ROI", display_frame)

        return roi_game, roi_batt, roi_time
    
    def isolate_roi_contours(self, contours: list[np.ndarray], roi: np.ndarray):
        """Identifies contour coordinates of rectangular tokens within a given ROI
        
            Parameters: 
                contours (list): All frame contours as a list of NumPy arrays, detected by canny edge detection
                roi (np.ndarray): The rotated rectangle defining the ROI
                
            Returns:
                list: Box corner coordinates, as a list of the rectangles identified as tokens within the ROI
        """ 
        # Check if ROI is valid
        if roi is None or not isinstance(roi, np.ndarray) or roi.shape != (4, 2):
            print("Warning: Invalid ROI provided. Skipping contour isolation.")
            return []  # Return an empty list if ROI is invalid

        isolated_token_rectangles = []
        
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            x, y = rect[0][0], rect[0][1]
            width, height = rect[1][0], rect[1][1]
            area = width * height
            
            # Skip if one side is more than 3 times the other - avoids slivers of colour from shadows
            if max(width, height) > 3 * min(width, height):
                continue
             
            if 300 < area <= 900:
                # Check if the center of the rectangle is within the ROI
                if cv2.pointPolygonTest(roi, (x, y), False) >= 0:
                    if self.non_identical_check(rect, isolated_token_rectangles):
                        isolated_token_rectangles.append(rect)
                    else:
                        continue
                                
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
        isolated_token_coords = self.identify_token_coords(canny_contours,balanced_frame)
        
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
    
    def identify_token_coords(self, contours: list[np.ndarray],display_frame) -> list:
        '''Processes canny contours and filters them to retrive isolated coordinates of token corner points
        
        Parameters:
            contours:  list[np.ndarray] The list of NumPy arrays details all found Canny contours to be filtered 
            so that tokens can be identified
            display_frame: passed in so that the roi can be displayed on the frame
            
        Returns:
            contours of the detected Canny edges as a list of NumPy arrays'''
        # defines the eges of the roi from the frame
        roi_game, roi_batt, roi_time = self.find_largest_roi(contours, display_frame)
        # identifies tokens within that roi and returns the list of their box coordinates
        token_rect_coords = self.isolate_roi_contours(contours, roi_game)
        #print(f"Token coordinates: {token_rect_coords}")
        
        return token_rect_coords
             
    def display_smoothed_count(self, frame, smoothed_count: dict) -> np.ndarray:
        ''' Displays the smoothed token counts on a frame - currently unused in code but can be incorporated if count is passed in
        
        Parameters:
            smoothed_count: dict: The smoothed counts of tokens for each color
        
        Returns:
            frame: np.ndarray: The frame with the smoothed counts displayed on it
        '''
        y_offset = 30
        color_map = {
            "yellow": (0, 255, 255),
            "magenta": (255, 0, 255),
            "cyan": (255, 255, 0),
            "green": (0, 255, 0)
        }
        for color in ["yellow", "magenta", "cyan", "green"]:
            count = smoothed_count.get(color, 0)
            cv2.putText(frame, f"{color}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color_map[color], 2)
            y_offset += 30
        return frame
    
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
