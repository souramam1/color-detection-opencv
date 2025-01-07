import numpy as np
import cv2

class ColorDetection:
    def __init__(self, webcam_index=0):
        # Initialize webcam and set color ranges for Red, Green, and Blue
        self.webcam = cv2.VideoCapture(webcam_index)
        
        # Define lower and upper ranges for Red, Green, and Blue in HSV space
        self.color_ranges = {
            "red": (np.array([136, 87, 111], np.uint8), np.array([180, 255, 255], np.uint8)),
            "green": (np.array([25, 52, 72], np.uint8), np.array([102, 255, 255], np.uint8)),
            "blue": (np.array([94, 80, 2], np.uint8), np.array([120, 255, 255], np.uint8))
        }
        
        self.kernel = np.ones((5, 5), "uint8")  # Kernel for dilation
        self.object_counts = {color: 0 for color in self.color_ranges}  # Initialize object counts for each color
        self.previous_contours = {color: [] for color in self.color_ranges}  # To store previous contours
    
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
            self.object_counts[color] = 0  # Reset the count for each color
            current_contours = []  # List to store the contours detected in the current frame

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300:  # Only consider large enough contours
                    self.object_counts[color] += 1  # Increase the count for this color
                    x, y, w, h = cv2.boundingRect(contour)
                    current_contours.append((x, y, w, h))  # Store the bounding box for the current contour
                    cv2.rectangle(imageFrame, (x, y), (x + w, y + h), self.get_color_for_display(color), 2)
                    cv2.putText(imageFrame, f"{color.capitalize()} Colour", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.get_color_for_display(color))
            
            # Keep track of which contours are missing by comparing to the previous frame
            self.track_missing_objects(color, current_contours)
            self.previous_contours[color] = current_contours  # Update the previous contours

        return imageFrame
    
    def track_missing_objects(self, color, current_contours):
        # Compare current contours with previous contours to detect missing objects
        prev_contours = self.previous_contours[color]
        missing_objects = len(prev_contours) - len(current_contours)

        if missing_objects > 0:
            self.object_counts[color] = max(0, self.object_counts[color] - missing_objects)  # Decrease count if objects disappear
    
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
        cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)
        
        # Display the tally of objects for each color
        y_offset = 30  # Starting y position for displaying tally
        for color, count in self.object_counts.items():
            cv2.putText(imageFrame, f"{color.capitalize()} Objects: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.get_color_for_display(color), 2)
            y_offset += 40  # Move down for the next color tally

    def run(self):
        # Main loop to continuously capture frames, detect colors, and display results
        while True:
            imageFrame, masks = self.process_frame()
            if imageFrame is None:
                break

            # Detect and draw contours on the image
            imageFrame = self.detect_and_draw_contours(imageFrame, masks)
            
            # Show the result (image with detected colors and tally)
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
    color_detection = ColorDetection()
    color_detection.run()
