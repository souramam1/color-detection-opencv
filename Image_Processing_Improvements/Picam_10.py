import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

class ColourDetectionWithKNN:
    
    def __init__(self, k_neighbors=3, webcam_index=1):
        self.webcam = cv2.VideoCapture(webcam_index)

    def capture_frame(self):
        """Capture a frame from the webcam."""
        ret, frame = self.webcam.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame

    def preprocess_and_segment_with_canny(self, frame):
        """Applies Watershed algorithm to separate touching tokens, starting with a Canny edge image."""
        # Ensure the input frame is binary (Canny result)
        if len(frame.shape) > 2:
            print("Converting to grayscale")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
        else:
            edges = frame  # If already a binary image (Canny result)

        cv2.imshow("Canny Edges", edges)  # Display Canny edges

        # Morphological opening to remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imshow("Morphological Opening", opening)  # Display morphological opening

        # Background detection
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        cv2.imshow("Sure Background", sure_bg)  # Display sure background

        # Distance transform to find sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        cv2.imshow("Distance Transform", dist_transform)  # Display distance transform
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        cv2.imshow("Sure Foreground", sure_fg)  # Display sure foreground

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        cv2.imshow("Unknown Region", unknown)  # Display unknown region

        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1  # Ensure background is not zero
        markers[unknown == 255] = 0  # Mark unknown region

        # Ensure markers are of type CV_32SC1 (32-bit integer array)
        markers = np.int32(markers)

        # Ensure frame_copy is a color image (3-channel) for watershed to work
        if len(frame.shape) == 2:  # If input is grayscale, convert to 3-channel BGR
            frame_copy = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_copy = frame.copy()

        # Apply Watershed algorithm
        cv2.watershed(frame_copy, markers)
        frame_copy[markers == -1] = [0, 0, 255]  # Mark boundaries in red

        # Extract refined contours
        contours, _ = cv2.findContours((markers > 1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Show a frame named watershed with the frame_copy and detected contours
        cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 3)
        cv2.imshow("Watershed", frame_copy)

        return contours



    def detect_and_draw_contours(self, gray_frame):
        """Detects contours using Canny edge detection and draws them on the image."""
        
        blurred = self.apply_gaussian_blur(gray_frame)
        edges = self.apply_canny_edge_detection(blurred)
        thresh = self.apply_adaptive_thresholding_detection(blurred)
        contours = self.find_contours(edges)
        frame_copy = self.draw_initial_contours(gray_frame, contours)
        roi = self.find_largest_roi(contours)
        
        
        if roi:
            print("ROI DETECTED")
            contours = self.preprocess_and_segment_with_canny(thresh) #this would use a canny edge detection as passed into the segment
            filtered_contours = self.filter_contours_by_size(contours)
            frame_copy = self.draw_roi_contours(frame_copy, filtered_contours, roi)
        
        #self.draw_colored_contours(frame_copy, contours)
        
        return frame_copy

    def apply_gaussian_blur(self, gray_frame):
        """Apply GaussianBlur to reduce noise."""
        return cv2.GaussianBlur(gray_frame, (5, 5), 0)

    def apply_canny_edge_detection(self, blurred):
        """Use Canny edge detection to find edges."""
        edges = cv2.Canny(blurred, 80, 100)
        cv2.imshow("Canny Edge Detection", edges)
        return edges
    
    def apply_adaptive_thresholding_detection(self, blurred):
        """Use Canny edge detection to find edges."""
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)  # Adaptive 
        cv2.imshow("Canny Edge Detection", thresh)
        return thresh

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
        """Find the largest rectangulaedr contour and define it as the region of interest (ROI)."""
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
    
    def filter_contours_by_size(self, contours, min_size=300, max_size=9000, width=28, height=55, tolerance=10):
        """Filters contours based on known token size and aspect ratio."""
        filtered_contours = []
        aspect_ratio_range = (width / height, height / width)  # Account for rotation
        
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (w, h) = rect[1]
            
            if w == 0 or h == 0:
                continue  # Ignore invalid contours
            
            area = w * h
            aspect_ratio = min(w/h, h/w)
            
            if min_size <= area <= max_size and aspect_ratio_range[0] - 0.1 <= aspect_ratio <= aspect_ratio_range[1] + 0.1:
                filtered_contours.append(contour)
        
        return filtered_contours

    def draw_roi_contours(self, frame_copy, contours, roi):
        """Draws rotated bounding boxes within the ROI."""
        roi_x, roi_y, roi_w, roi_h = roi
        cv2.rectangle(frame_copy, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
        
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Check if contour is inside ROI
            if roi_x <= rect[0][0] <= roi_x + roi_w and roi_y <= rect[0][1] <= roi_y + roi_h:
                cv2.drawContours(frame_copy, [box], 0, (0, 255, 0), 2)
                
                # Display area for debugging
                area = rect[1][0] * rect[1][1]
                cv2.putText(frame_copy, f"{area:.0f}", (int(rect[0][0]), int(rect[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
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
        frame = self.capture_frame()
        if frame is None:
            return None, None, None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_with_canny_contours = self.detect_and_draw_contours(gray_frame)
        #cv2.imshow("contours detected", frame_with_contours)
        return frame_with_canny_contours

    
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
                frame = self.process_frame()
                
                
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
    color_detection = ColourDetectionWithKNN(k_neighbors=3)
    color_detection.run()
