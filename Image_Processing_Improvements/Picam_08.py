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
        _, thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Approximate contour
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:  # Looking for a rectangular ROI
                x, y, w, h = cv2.boundingRect(approx)
                return (x, y, w, h)
        
        return None

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
                frame, object_counts, _ = self.process_frame()
                if frame is not None:
                    self.show_result(frame, object_counts)
                
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
