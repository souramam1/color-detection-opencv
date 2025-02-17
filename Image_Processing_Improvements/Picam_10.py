import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

class ColourDetectionWithKNN:
    
    def __init__(self, k_neighbors=3, webcam_index=1):
        self.webcam = cv2.VideoCapture(webcam_index)

        # Predefined HSV color samples for classification
        self.color_samples = {
            "orange": (10, 255, 255),
            "yellow": (30, 255, 255),
            "magenta": (160, 255, 255),
            "teal": (90, 255, 255)
        }

        # Train k-NN classifier
        self.knn = self.train_knn_classifier(k_neighbors)

    def train_knn_classifier(self, k_neighbors):
        """Train a k-NN classifier using predefined HSV colors."""
        color_values = np.array(list(self.color_samples.values()))
        color_labels = list(self.color_samples.keys())
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
        knn.fit(color_values, color_labels)
        return knn

    def capture_frame(self):
        """Capture a frame from the webcam."""
        ret, frame = self.webcam.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame

    def detect_roi(self, gray_frame):
        """Detect the largest rectangular ROI using Canny edge detection."""
        
        # Step 1: Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Step 2: Use Canny edge detection to find edges
        edges = cv2.Canny(blurred, 50, 150)  # Thresholds for edge detection
        cv2.imshow("Canny Edges", edges)  # Show the edge-detected image

        # Step 3: Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_rect = None
        min_area = 80000  # Minimum area threshold to filter out small contours
        
        frame_copy = gray_frame.copy()  # Copy to draw contours

        # Step 4: Iterate through contours to find the largest rectangular contour
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Approximation factor
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Step 5: Check if the contour has 4 points (suggesting it's a rectangle)
            if len(approx) == 4:  # It is a quadrilateral
                area = cv2.contourArea(approx)
                
                # Filter for the largest area rectangle
                if area > min_area:
                    min_area = area
                    largest_rect = approx  # Store the largest rectangular contour
        
        # Step 6: Draw the largest rectangle (if found)
        if largest_rect is not None:
            x, y, w, h = cv2.boundingRect(largest_rect)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the ROI
            cv2.imshow("ROI Detection", frame_copy)  # Display the frame with detected ROI
            
            return (x, y, w, h)  # Return the bounding box of the ROI
        
        return None  # If no ROI found, return None

    def detect_and_draw_contours(self, gray_frame, roi=None):
        """Detects contours using Canny edge detection and draws them on the image, only within the ROI."""

        # Step 1: Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Step 2: Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)  # Canny edge detection
        
        # Step 3: Find contours in the edge-detected image (on the full frame)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_copy = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Convert gray to BGR for drawing
        
        contour_count = 0  # Count of detected contours

        # Step 4: Draw contours on the frame, but only those within the ROI
        if roi:
            roi_x, roi_y, roi_w, roi_h = roi  # Extract ROI coordinates
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Draw contours only if their area is above a certain threshold
                if 50 < area <= 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if the contour is inside the ROI
                    if roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h:
                        # Draw green rectangles for contours inside the ROI
                        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        contour_count += 1  # Increment contour count
                
                # Optionally draw larger contours in a different color (e.g., red)
                if area > 90000:
                    x, y, w, h = cv2.boundingRect(contour)
                    if roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h:
                        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangles
        
        # Step 5: Show the number of contours detected
        cv2.putText(frame_copy, f"Contours Detected: {contour_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow("Contours in ROI", frame_copy)  # Show the final image with contours drawn
        return frame_copy  # Return the frame with contours drawn

    def process_frame(self):
        """Main pipeline: Detect ROI, detect objects, classify colors."""
        frame = self.capture_frame()
        if frame is None:
            return None, None, None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        roi = self.detect_roi(gray_frame)  # Detect the ROI (using Canny)
        frame_with_contours = self.detect_and_draw_contours(gray_frame, roi)  # Detect and draw contours within ROI
        
        return frame_with_contours

    def show_thresholding(self):
        """Testing will show outcome of different thresholding techniques"""
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

        return frame
    
    def local_otsu_thresholding(self, frame, tile_size=64):
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
                # Process the frame (detect ROI and contours within ROI)
                frame = self.process_frame()
                
                if frame is not None:
                    # Display the result (running count of contours)
                    print("Test running") 

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
