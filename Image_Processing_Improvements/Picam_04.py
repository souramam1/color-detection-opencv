import numpy as np
import cv2
from picamera2 import Picamera2
from collections import deque


#This script detects the bounds of a region of interest - defined as the largest area
# bound by the colour green - and then only identifies colour blobs within those bounds.
# It does not count those blobs over time.

class ColorDetectionWithROI:
    def __init__(self, smoothing_window_size=5, resolution=(640, 480), format="RGB888"):
        self.picam2 = Picamera2()
        self.resolution = resolution
        self.format = format
        self.configure_camera()
        
        # Define HSV range for green color (adjust if needed)
        self.green_range = ((40, 50, 50), (80, 255, 255))  # HSV range for green
        
        # Define color ranges for contour detection
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
        """Capture a frame from the Picamera2 and return it."""
        frame = self.picam2.capture_array()
        return frame
    
    def detect_green_roi(self, hsv_frame):
        """Detect the green ROI in the frame."""
        mask = cv2.inRange(hsv_frame, *self.green_range)
        mask = cv2.dilate(mask, self.kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Approximation accuracy
            approx = cv2.approxPolyDP(contour, epsilon, True)  
            print(f"measured contour length is {approx} !")
            
            if len(approx) == 4:
                area = cv2. contourArea(contour)  
            
                if area > 3000:  # Threshold for green area size
                    x, y, w, h = cv2.boundingRect(contour)
                    return (x, y, w, h)  # Return the bounding box of the largest green area
        return None  # No ROI detected
    
    def process_frame(self):
        # Capture the current frame and convert to HSV
        image_frame = self.capture_frame()
        hsv_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)
        
        # Detect green ROI
        roi = self.detect_green_roi(hsv_frame)
        masks = {}
        
        # If a green ROI is detected, process colors within the ROI
        if roi:
            x, y, w, h = roi
            roi_hsv = hsv_frame[y:y+h, x:x+w]  # Crop HSV frame to ROI
            
            # Generate masks for the defined colors within the ROI
            for color, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(roi_hsv, lower, upper)
                mask = cv2.dilate(mask, self.kernel)
                masks[color] = mask
            
            return image_frame, masks, roi
        else:
            return image_frame, {}, None
    
    def detect_and_draw_contours(self, image_frame, masks, roi):
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
                        cv2.putText(image_frame, f"{color.capitalize()} Colour", (x+cx, y+cy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.get_color_for_display(color), 2)
                
                self.object_counts[color].append(object_count)
            
            # Draw the green ROI box
            cv2.rectangle(image_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_frame, "ROI", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_frame
    
    def get_smoothed_count(self, color):
        # Apply moving average smoothing to the counts for each color
        
        if not self.object_counts[color]:  # Check if the list/array is empty
            return 0  # Or another default value
        return int(np.mean(self.object_counts[color]))
    
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
        
        # Display the image with detected color regions
        y_offset = 30  # Starting y position for displaying tally
        
        # Loop over all colors and their smoothed counts to display the tally
        for color in self.color_ranges:
            smoothed_count = self.get_smoothed_count(color)
            cv2.putText(image_frame, f"{color.capitalize()} Objects: {smoothed_count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.get_color_for_display(color), 3)  # Adjusted font size
            y_offset += 40  # Move down for the next color tally
        """Display the result frame."""
        cv2.imshow("Color Detection with ROI", image_frame)
    
    def run(self):
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
        self.picam2.stop()
        self.picam2.close()
        cv2.destroyAllWindows()


# Run the program
if __name__ == "__main__":
    color_detection = ColorDetectionWithROI(smoothing_window_size=10)
    color_detection.run()
