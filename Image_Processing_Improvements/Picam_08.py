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
        """Detects contours and draws bounding boxes around them."""
        # Apply adaptive thresholding to the grayscale image
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Reduce noise
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
        cv2.imshow("Thresholded Image", thresh)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Make a copy of the original grayscale frame to draw on it
        frame_copy = gray_frame.copy()

        # Loop through the contours
        for contour in contours:
            # Get the bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw a green rectangle around each detected contour
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

        # Show the result with bounding boxes drawn
        cv2.imshow("Contours Detected", frame_copy)

        return frame_copy



    def detect_objects(self, gray_frame, roi):
        """Detect objects inside the ROI using adaptive thresholding."""
        x, y, w, h = roi
        roi_gray = gray_frame[y:y+h, x:x+w]  # Extract ROI
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)  # Reduce noise
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
        

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    

    def classify_object_color(self, roi_frame, contour):
        """Classify the color of an object using k-NN with HSV colors."""
        hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        mean_hsv = cv2.mean(hsv_frame, mask=mask)[:3]

        color_label = self.knn.predict([mean_hsv])
        return color_label[0]

    def process_frame(self):
        """Main pipeline: Detect ROI, detect objects, classify colors."""
        frame = self.capture_frame()
        if frame is None:
            return None, None, None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        roi = self.detect_roi(gray_frame)

        if roi:
            print("ROI DETECTED!")
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]
            contours = self.detect_objects(gray_frame, roi)

            object_counts = {color: 0 for color in self.color_samples.keys()}
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Ignore small noise
                    color = self.classify_object_color(roi_frame, contour)
                    object_counts[color] += 1

                    # Draw contours
                    cx, cy, cw, ch = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x+cx, y+cy), (x+cx+cw, y+cy+ch), self.get_color_for_display(color), 2)
                    cv2.putText(frame, color.capitalize(), (x+cx, y+cy-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.get_color_for_display(color), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "ROI", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            object_counts = None

        return frame, object_counts, roi
    
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

        cv2.imshow("simple greyscale", gray_frame)
        cv2.imshow("gaussian blurred", blurred)
        cv2.imshow("adaptive thresholded after gauss", thresh)
        cv2.imshow("otsu thresholding after gauss", otsu_thresh)
        cv2.imshow("otsu normalisation and adaptive thresholding", otsu_adaptive_thresh)
        cv2.imshow("local otsu thresholding", local_otsu_threshold)
        
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

    def get_color_for_display(self, color):
        """Map color name to display color in BGR for visualization."""
        color_map = {
            "orange": (0, 165, 255),
            "yellow": (0, 255, 255),
            "magenta": (255, 0, 255),
            "teal": (255, 128, 0)
        }
        return color_map.get(color, (255, 255, 255))

    def show_result(self, frame, object_counts):
        """Display the processed frame and object count."""
        y_offset = 30
        if object_counts:
            for color, count in object_counts.items():
                cv2.putText(frame, f"{color.capitalize()}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.get_color_for_display(color), 3)
                y_offset += 40

        cv2.imshow("Object Detection with k-NN", frame)

    def run(self):
        """Run the real-time object detection and classification pipeline."""
        try:
            while True:
                # ORIGINAL CODE - UNCOMMENT
                frame, object_counts, _ = self.process_frame()
                # END OF ORIGINAL
                
                # TESTING CODE
                #frame = self.show_thresholding()
                #END OF TESTING CODE
                
                if frame is not None:
                    # ORIGINAL CODE  - UNCOMMENT
                    self.show_result(frame, object_counts)
                    # END OF ORIGINAL
                    
                    #TESTING CODE
                    #print("test running") 
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
