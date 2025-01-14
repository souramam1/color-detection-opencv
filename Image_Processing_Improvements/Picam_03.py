import numpy as np
import cv2
from picamera2 import Picamera2
from collections import deque

class ColorDetectionWithROI:
    def __init__(self, smoothing_window_size=5, resolution=(640, 480), format="RGB888"):
        # Initialize camera and set parameters
        self.picam2 = Picamera2()
        self.resolution = resolution
        self.format = format
        self.configure_camera()
        
        # Define HSV range for green color (used to detect markers)
        self.green_range = ((40, 50, 50), (80, 255, 255))
        
        # Define HSV ranges for other colors
        self.color_ranges = {
            "orange": ((0, 50, 50), (10, 255, 255)),
            "yellow": ((15, 50, 50), (40, 255, 255)),
            "magenta": ((140, 50, 50), (170, 255, 255)),
            "teal": ((85, 50, 50), (100, 255, 255))
        }
        
        self.kernel = np.ones((5, 5), "uint8")
        self.smoothing_window_size = smoothing_window_size
        self.object_counts = {color: deque(maxlen=smoothing_window_size) for color in self.color_ranges}
    
    def configure_camera(self):
        """Configure the camera with specified resolution and format."""
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": self.format, "size": self.resolution}))
        self.picam2.start()
    
    def capture_frame(self):
        """Capture a frame from the Picamera2."""
        return self.picam2.capture_array()
    
    def detect_green_markers(self, hsv_frame):
        """Detect the positions of green markers in the frame."""
        mask = cv2.inRange(hsv_frame, *self.green_range)
        mask = cv2.dilate(mask, self.kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        marker_centers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum size for a marker
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    marker_centers.append((cx, cy))
        
        return marker_centers
    
    def calculate_roi(self, markers):
        """Calculate the bounding box of the ROI based on green markers."""
        if len(markers) >= 4:
            xs, ys = zip(*markers)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            return x_min, y_min, x_max - x_min, y_max - y_min
        return None
    
    def process_frame(self):
        """Capture and process the current frame."""
        image_frame = self.capture_frame()
        hsv_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)
        
        # Detect green markers
        markers = self.detect_green_markers(hsv_frame)
        roi = self.calculate_roi(markers)
        
        masks = {}
        
        if roi:
            x, y, w, h = roi
            roi_hsv = hsv_frame[y:y+h, x:x+w]
            
            # Create masks for colors within the ROI
            for color, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(roi_hsv, lower, upper)
                mask = cv2.dilate(mask, self.kernel)
                masks[color] = mask
            
            return image_frame, masks, roi
        else:
            return image_frame, {}, None
    
    def detect_and_draw_contours(self, image_frame, masks, roi):
        """Detect contours within the ROI and draw them."""
        if roi:
            x, y, w, h = roi
            for color, mask in masks.items():
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                object_count = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Minimum contour area threshold
                        object_count += 1
                        cx, cy, cw, ch = cv2.boundingRect(contour)
                        cv2.rectangle(image_frame, (x+cx, y+cy), (x+cx+cw, y+cy+ch), self.get_color_for_display(color), 2)
                        cv2.putText(image_frame, f"{color.capitalize()}", (x+cx, y+cy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.get_color_for_display(color), 2)
                
                self.object_counts[color].append(object_count)
            
            # Draw ROI rectangle
            cv2.rectangle(image_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_frame, "ROI", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_frame
    
    def get_color_for_display(self, color):
        """Map color name to display color in BGR."""
        color_map = {
            "orange": (0, 165, 255),
            "yellow": (0, 255, 255),
            "magenta": (255, 0, 255),
            "teal": (255, 128, 0)
        }
        return color_map.get(color, (255, 255, 255))
    
    def show_result(self, image_frame):
        """Display the result frame."""
        cv2.imshow("Color Detection with ROI", image_frame)
    
    def run(self):
        """Main loop for detection."""
        try:
            while True:
                image_frame, masks, roi = self.process_frame()
                image_frame = self.detect_and_draw_contours(image_frame, masks, roi)
                self.show_result(image_frame)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Stop the camera and close windows."""
        self.picam2.stop()
        self.picam2.close()
        cv2.destroyAllWindows()


# Run the program
if __name__ == "__main__":
    color_detection = ColorDetectionWithROI(smoothing_window_size=10)
    color_detection.run()
