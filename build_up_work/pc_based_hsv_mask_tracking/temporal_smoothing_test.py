import numpy as np
import cv2
from collections import deque

#NEXT smooth bounding box positions.
#NEXT load this onto the PI by merging v1r6 type code with this.

class ColorDetection:
    def __init__(self, webcam_index=0, smoothing_window_size=5):
        # Initialize webcam and set color ranges for Red, Green, and Blue
        self.webcam = cv2.VideoCapture(webcam_index)
        
        # Define lower and upper ranges for Red, Green, and Blue in HSV space
        self.color_ranges = {
            "black": ((0, 0, 0), (180, 255, 50)),          # Adjust for your black detection
            "orange": ((10, 100, 100), (25, 255, 255)),   # Hue for orange
            "yellow": ((25, 100, 100), (35, 255, 255)),   # Hue for yellow
            "magenta": ((140, 50, 50), (170, 255, 255)),  # Hue for magenta
            "teal": ((85, 50, 50), (100, 255, 255))
        }
        
        self.kernel = np.ones((5, 5), "uint8")  # Kernel for dilation
        self.smoothing_window_size = smoothing_window_size  # Number of frames to smooth over
        
        # Initialize object counts and deque (for history)
        self.object_counts = {color: deque(maxlen=smoothing_window_size) for color in self.color_ranges}
    
    def process_frame(self):
        # Read the current frame from the webcam
        ret, imageFrame = self.webcam.read()
        if not ret:
            print("Failed to grab frame")
            return None

        # Convert the frame to HSV (Hue, Saturation, Value) color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        masks = {}

        # Generate masks for red, green, and blue colors based on the color ranges
        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsvFrame, lower, upper)
            mask = cv2.dilate(mask, self.kernel)  # Dilate the mask
            masks[color] = mask

        return imageFrame, masks
    
    def detect_and_draw_contours(self, imageFrame, masks):
        # Process the masks and draw contours around detected colors
        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            object_count = 0  # Count objects in this frame
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300:  # Only consider large enough contours
                    object_count += 1  # Increase the count for this color
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(imageFrame, (x, y), (x + w, y + h), self.get_color_for_display(color), 2)
                    cv2.putText(imageFrame, f"{color.capitalize()} Colour", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.get_color_for_display(color))

            # Store the count for smoothing
            self.object_counts[color].append(object_count)
        
        return imageFrame
    
    def get_smoothed_count(self, color):
        # Apply moving average smoothing to the counts for each color
        return int(np.mean(self.object_counts[color]))  # Return smoothed count as integer
    
    def get_color_for_display(self, color):
        # Map color name to display color in BGR (OpenCV color format)
        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0)
        }
        return color_map.get(color, (255, 255, 255))
    
    def show_result(self, imageFrame):
        # Display the image with detected color regions
        y_offset = 30  # Starting y position for displaying tally
        
        # Loop over all colors and their smoothed counts to display the tally
        for color in self.color_ranges:
            smoothed_count = self.get_smoothed_count(color)
            cv2.putText(imageFrame, f"{color.capitalize()} Objects: {smoothed_count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.get_color_for_display(color), 3)  # Adjusted font size
            y_offset += 40  # Move down for the next color tally
        
        # Show the result window
        cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)
    
    def run(self):
        # Main loop to continuously capture frames, detect colors, and display results
        while True:
            imageFrame, masks = self.process_frame()
            if imageFrame is None:
                break

            # Detect and draw contours on the image
            imageFrame = self.detect_and_draw_contours(imageFrame, masks)
            
            # Show the result
            self.show_result(imageFrame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        # Release the webcam and close all OpenCV windows
        self.webcam.release()
        cv2.destroyAllWindows()


# Create a ColorDetection object and run it
if __name__ == "__main__":
     # Use a smoothing window size of 5 frames --> 100 frames (to bear in mind: average is 30fps)
    color_detection = ColorDetection(smoothing_window_size=60) 
    print("here 1")
    color_detection.run()
