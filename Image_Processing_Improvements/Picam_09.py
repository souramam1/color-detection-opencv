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
        """Detects a rectangular ROI using contour detection."""
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Reduce noise
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
        cv2.imshow("adaptive thresholding ROI", thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_rect = None
        min_area = 80000

        # Make a copy of the original grayscale frame to draw contours on it
        frame_copy = gray_frame.copy()

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Approximate contour
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Draw contours on the frame_copy (not on the thresholded image)
            cv2.drawContours(frame_copy, [approx], -1, (0, 255, 0), 2)  # Draw contour in green

            if len(approx) == 4:  # Looking for a rectangular ROI
                area = cv2.contourArea(approx)
                if area > min_area:
                    min_area = area
                    largest_rect = approx

        # After detecting the largest rectangle, draw it on the copy of the frame
        if largest_rect is not None:
            x, y, w, h = cv2.boundingRect(largest_rect)
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw the bounding box for the ROI
            cv2.imshow("ROI Detection", frame_copy)  # Show the result with the detected ROI

            return (x, y, w, h)

        return None
    

    def detect_and_draw_contours(self, gray_frame):
        """Detects contours using Canny edge detection and draws them on the image."""
        
        # Step 1: Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Step 2: Use Canny edge detection to find edges (Experiment with thresholds)
        edges = cv2.Canny(blurred, 80, 100)  # Lower threshold values to detect finer edges
        cv2.imshow("Canny Edge Detection", edges)

        # Step 3: Find contours in the edge-detected image (Try using RETR_TREE for hierarchical contours)
        # contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 4: Make a copy of the original grayscale image to draw on it
        frame_copy = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(frame_copy, contours, -1, (255,0,0), 1)

        # Step 5: Draw each contour
        # This will draw the contours themselves, not bounding boxes
        #cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 3)  # Drawing all contours in green

        # Step 6: Show the result with all contours drawn
        
        
        # for (contour, topology) in zip(contours,hierarchy):
        for contour in contours:
            # centre, size, rotation = cv2.minAreaRect(contour)
            _, size, _ = cv2.minAreaRect(contour)
            area = size[0] * size[1]
            print(f"heigh and width: {size[0], size[1]}")
            print(f"rect: {area}")
            if 300 < area <= 9000:
                x, y, w, h = cv2.boundingRect(contour)
                # Draw the contour in green
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"{area:.0f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                
            if area > 90000:
                x, y, w, h = cv2.boundingRect(contour)
                # Draw the contour in red
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
       
            # print(f"area : {area}")

        colours = [
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        for i, c in enumerate(contours):
            col = colours[i%3]
            cv2.drawContours(frame_copy, contours, i, col, 1)
            
        

        cv2.imshow("Contours Detected with Canny", frame_copy)
        return frame_copy

    def process_frame(self):
        """Main pipeline: Detect ROI, detect objects, classify colors."""
        frame = self.capture_frame()
        if frame is None:
            return None, None, None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_with_contours = self.detect_and_draw_contours(gray_frame)
        #cv2.imshow("contours detected", frame_with_contours)
        return frame_with_contours

    
    def show_thresholding(self):
        """Testing will show outcome of different thresholding techniques"""
        frame = self.capture_frame()
        if frame is None:
            return None, None, None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Reduce noise
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
        # otsu_threshold_val , otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # normalised = cv2.subtract(blurred, otsu_threshold_val)
        # otsu_adaptive_thresh = cv2.adaptiveThreshold(normalised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # local_otsu_threshold = self.local_otsu_thresholding(blurred, tile_size=10)


        contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        


        
        

        cv2.imshow("adaptive threshold contours", frame)
        cv2.imshow("simple greyscale", gray_frame)
        cv2.imshow("gaussian blurred", blurred)
        cv2.imshow("adaptive thresholded after gauss", thresh)
        # cv2.imshow("otsu thresholding after gauss", otsu_thresh)
        # cv2.imshow("otsu normalisation and adaptive thresholding", otsu_adaptive_thresh)
        # cv2.imshow("local otsu thresholding", local_otsu_threshold)
        self.process_frame()

        
        # RESULTS - ADAPTIVE THRESHOLDING DEALS WITH SHADOW VARIATION BEST
        # Then Second to that OTSU NORMALISATION for ADAPTIVE IS OK BUT deals poorly with shadows.
    
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
                # ORIGINAL CODE - UNCOMMENT
                frame = self.process_frame()
                # END OF ORIGINAL
                
                # TESTING CODE
                #frame = self.show_thresholding()
                #END OF TESTING CODE
                
                if frame is not None:
                    # ORIGINAL CODE  - UNCOMMENT
                    #self.show_result(frame, object_counts)
                    # END OF ORIGINAL
                    
                    #TESTING CODE
                    print("test running") 
                    # END OF TESTING CODE                   
                
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
