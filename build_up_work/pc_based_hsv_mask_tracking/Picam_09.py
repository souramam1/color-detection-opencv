import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

class ContourDetection:
    
    def __init__(self, webcam_index=1):
        self.webcam = cv2.VideoCapture(webcam_index)
        self.detected_token_contours = []
        self.color_ranges = {
            'red' : ((0, 50, 50), (8, 255, 255)),
            'orange': ((0, 0, 50), (16, 255, 255)),
            'yellow': ((16, 50, 50), (39, 255, 255)),
            'magenta': ((116, 52, 50), (179, 255, 255))
        }
        
        # Define BGR color values for text
        self.color_bgr = {
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'yellow': (0, 255, 255),
            'magenta': (255, 0, 255),
            'unknown': (255, 255, 255)  # Default to white for unknown colors
        }
        
    def classify_contour(self, contour, hsv_frame):
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(hsv_frame, mask=mask)[:3]
        
        for color, (lower, upper) in self.color_ranges.items():
            if cv2.inRange(np.uint8([[mean_val]]), np.array(lower), np.array(upper)):
                return color
        return 'unknown'

    def draw_contours(self, frame, contours, hsv_frame):
        for contour in contours:
            color = self.classify_contour(contour, hsv_frame)
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            # I want the text to be in the same colour as the idenity of the object
            text_color = self.color_bgr[color]
            cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        

    

    def capture_frame(self):
        """Capture a frame from the webcam."""
        ret, frame = self.webcam.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame



    def detect_and_draw_contours(self, gray_frame):
        """Detects contours using Canny edge detection and draws them on the image."""
        
        # Apply a gaussian blurr to gray_scale image
        blurred = self.apply_gaussian_blur(gray_frame)
        
        # Apply adaptive thresholding to reduce noise
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
        cv2.imshow("Thresholded", thresh)
        
        #Perform Canny edge detection on the blurred image - DOES NOT USE ADAPTIVE THRESHOLDING
        edges = self.apply_canny_edge_detection(blurred)
        
        #Find the contours once with the thresholded image and once with the canny edge detected image
        contours_thresh = self.find_contours(thresh)
        contours_canny = self.find_contours(edges)
        
        # Draw initial blue outlines of detected contours for both thresholded and canny edge detected images
        frame_copy = self.draw_initial_contours(gray_frame, contours_thresh)
        cv2.imshow("frame copy thresholded contours", frame_copy)
        
        
        frame_copy = self.draw_initial_contours(gray_frame, contours_canny)
        cv2.imshow("frame copy canny contours", frame_copy)
        
        # Use the canny contour to find the ROI (works better than thresholding)
        roi = self.find_largest_roi(contours_canny)
        if roi:
            # Identify tokens using the .draw_roi_contours method.
            # Which ever set of contours is passed into the method will be drawn on the image
            frame_copy = self.draw_roi_contours(frame_copy, contours_canny, roi)
        
        #self.draw_colored_contours(frame_copy, contours)
        
        return frame_copy

    def apply_gaussian_blur(self, gray_frame):
        """Apply GaussianBlur to reduce noise."""
        return cv2.GaussianBlur(gray_frame, (5, 5), 0)

    def apply_canny_edge_detection(self, blurred):
        """Use Canny edge detection to find edges."""
        edges = cv2.Canny(blurred, 20, 100)
        cv2.imshow("Canny Edge Detection", edges)
        return edges

    def find_contours(self, edges):
        """Find contours in the edge-detected image."""
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_initial_contours(self, gray_frame, contours):
        """Make a copy of the original grayscale image to draw on it."""
        frame_copy = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(frame_copy, contours, -1, (255, 0, 0), 1)
        return frame_copy

    def find_largest_roi(self, contours):
        """Find the largest rectangular contour and define it as the region of interest (ROI)."""
        largest_area = 0
        roi = None
        for contour in contours:
            _, size, _ = cv2.minAreaRect(contour)
            area = size[0] * size[1]
            if area > 90000 and area > largest_area:
                largest_area = area
                x, y, w, h = cv2.boundingRect(contour)
                roi = (x, y, w, h)
        return roi

    def draw_roi_contours(self, frame_copy, contours, roi):
        """Draw contours within the ROI and highlight nested contours."""
        # Takes the ROI and draws a rectangle around it
        roi_x, roi_y, roi_w, roi_h = roi
        cv2.rectangle(frame_copy, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
        
        # Empty list in which to store detected token contours
        self.detected_token_contours = []
        
        # Iterate through contours and draw them if they are within the ROI
        for contour in contours:
            # minAreaRect is better than cv2.rectangle for rotated rectangles
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            x,y = rect[0][0], rect[0][1]
            width,height = rect[1][0], rect[1][1]
            area = width * height
            
            # write width and height of rectangle as variables instead of indexing rect all the time.
            # if area is within a certain range, draw the contour and write the area on the image
            if 300 < area <= 800:
                if roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h:
                    print(f"width is {width} and height is {height}")
                    cv2.drawContours(frame_copy, [box], 0, (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"{area:.0f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                    
                    self.detected_token_contours.append(contour)
                    
                    
        print(f"frame type is: {type(frame_copy)}")
        return frame_copy

    def draw_colored_contours(self, frame_copy, contours):
        """Draw colored contours on the image."""
        colours = [
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        for i, c in enumerate(contours):
            col = colours[i % 3]
            cv2.drawContours(frame_copy, contours, i, col, 1)
    
    def show_result(self, frame):
        cv2.imshow("Contour detection with Canny", frame)

    def process_frame(self):
        """Main pipeline: Detect ROI, detect objects, classify colors."""
        
        # Capture a frame from the webcam
        frame = self.capture_frame()
        if frame is None:
            return None, None, None
        
        # Convert the frame to grayscale and HSV
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect contours using Canny edge detection and draw them on the image
        frame_with_canny_contours = self.detect_and_draw_contours(gray_frame)
        cv2.imshow("contours detected", frame_with_canny_contours)
        return frame_with_canny_contours, hsv_frame
    
    def show_thresholding(self):
        
        """TESTING METHOD --> TO DEMONSTRATE OUTCOME OF DIFFERENT THRESHOLDING TECHNIQUES"""
        frame = self.capture_frame()
        if frame is None:
            return None, None, None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Reduce noise
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
        otsu_threshold_val , otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        normalised = cv2.subtract(blurred, otsu_threshold_val)
        otsu_adaptive_thresh = cv2.adaptiveThreshold(normalised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        local_otsu_threshold = self.local_otsu_thresholding(blurred, tile_size=10)

        contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        
        cv2.imshow("adaptive threshold contours", frame)
        cv2.imshow("simple greyscale", gray_frame)
        cv2.imshow("gaussian blurred", blurred)
        cv2.imshow("adaptive thresholded after gauss", thresh)
        cv2.imshow("otsu thresholding after gauss", otsu_thresh)
        cv2.imshow("otsu normalisation and adaptive thresholding", otsu_adaptive_thresh)
        cv2.imshow("local otsu thresholding", local_otsu_threshold)
        self.process_frame()

        
        # RESULTS - ADAPTIVE THRESHOLDING DEALS WITH SHADOW VARIATION BEST
        # Then Second to that OTSU NORMALISATION for ADAPTIVE IS OK BUT deals poorly with shadows.
    
        return frame
    
    def local_otsu_thresholding(self, frame, tile_size=64):
        """USED IN THRESHOLDING DEMONSTRATION ONLY"""
        
        h, w = frame.shape  # Get frame dimensions
        locally_thresholded = np.zeros_like(frame)  # Output image

        # Loop over the image in tiles
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                
                y_end = min(y+ tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = frame[y:y_end, x:x_end]  # Extract tile

                if tile.size > 0:  # Ensure the tile is not empty
                    _, tile_thresh = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    locally_thresholded[y:y+tile_size, x:x+tile_size] = tile_thresh  # Replace with thresholded tile

        return locally_thresholded


    def run(self):
        """Run the real-time object detection and classification pipeline."""
        try:
            while True:
                # ORIGINAL CODE - UNCOMMENT
                frame, hsv_frame = self.process_frame()
                self.draw_contours(frame, self.detected_token_contours, hsv_frame)
            
                

                # TESTING CODE
                #frame = self.show_thresholding()
                
                
                if frame is not None:
                    # ORIGINAL CODE  - UNCOMMENT
                    self.show_result(frame)
                    
                    
                    #TESTING CODE
                    #print("test running") 
                                      
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        """Release the webcam and close all OpenCV windows."""
        self.webcam.release()
        cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    color_detection = ContourDetection()
    color_detection.run()
