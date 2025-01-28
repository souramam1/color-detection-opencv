import numpy as np
import cv2
from picamera2 import Picamera2
from collections import deque, Counter


# This will be the same as PICAM 02 but adapted to count the most recent colour and total tokens in time along a time bar
# This is for the clock!

class ColorDetectionWithROI:
    def __init__(self, smoothing_window_size=5,transition_window_size=10,stability_threshold = 5, resolution=(640, 480), format="RGB888"):
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
        
        self.smaller_y_over_time = deque(maxlen=transition_window_size)
        self.max_colour_over_time = deque(maxlen=transition_window_size)
        
        self.last_player_colour = None
        self.last_smaller_y = None
        
        self.stability_threshold = stability_threshold
        self.stable_count = 0
        self.previous_stable_values = (None,None)
    
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
            area = cv2.contourArea(contour)
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
        if not roi:
            return image_frame

        x, y, w, h = roi
        max_y, max_y_contour, max_y_color, smaller_y_count = self.find_contours_with_y_and_color(masks, x, y)
        
        smoothed_max_y_color, smoothed_smaller_y_count = self.get_smoothed_color_time()
        
        

        # Check if the player or time has changed
        if (smoothed_max_y_color, smoothed_smaller_y_count) != self.previous_stable_values:
            self.stable_count = 0  # Reset stability counter (values have changed)
        else:
            self.stable_count += 1  # Increment stability counter (values are the same)

        # If stable for enough frames, update and send
        if self.stable_count >= self.stability_threshold:
            print("stable count has reached the threshold")
            if (smoothed_max_y_color, smoothed_smaller_y_count) != self.previous_stable_values:
                print(f"STABLE UPDATE TO SEND: {(smoothed_max_y_color, smoothed_smaller_y_count)}")
                self.previous_stable_values = (smoothed_max_y_color, smoothed_smaller_y_count)
                self.stable_count = 0  # Reset stability counter after sending data


        # Display count of smaller-y-value contours
        
        self.display_time_of_day(image_frame, smoothed_smaller_y_count)

        # Draw the green ROI box
        self.draw_roi_box(image_frame, x, y, w, h)

        # Add arrow to indicate direction of increasing y
        self.draw_y_axis_arrow(image_frame)

        return image_frame
    
    def find_contours_with_y_and_color(self, masks, x_offset, y_offset):
        """
        Find all valid contours and calculate the one with the maximum y-coordinate,
        its color, and the count of other contours with a smaller y-coordinate.
        """
        contours_info = []  # List to store info about valid contours: (bottom_y, color, contour)
        
        
        # Step 1: Collect all valid contours
        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 3000:  # Contour area thresholds
                    cx, cy, cw, ch = cv2.boundingRect(contour)
                    bottom_y = y_offset + cy + ch
                    contour_loop = (cx, cy, cw, ch)
                    contours_info.append((bottom_y, color, contour_loop))  # Add contour info as well
                    self.max_colour_over_time.append(color)

            
        # Step 2: Sort contours by bottom_y (largest first)
        contours_info.sort(key=lambda x: x[0], reverse=True)
        
        # Step 3: Identify max y contour and count others with smaller y
        # TO ADD: to smooth this! Make a dequeue containing the length of the contours_info file and smooth it
        
        if contours_info:
            max_y = contours_info[0][0]
            max_y_color = contours_info[0][1]
            max_y_contour = contours_info[0][2]
            smaller_y_count = len(contours_info) - 1
            self.smaller_y_over_time.append(smaller_y_count)
            #print(f"contours info is {contours_info}")
        else:
            max_y = -1
            max_y_color = None
            max_y_contour = None
            smaller_y_count = 0
            
        
        # Step 4: Optional: Store the color of the max y contour
        if max_y_color:
            self.player_colour = max_y_color

        return max_y, max_y_contour, max_y_color, smaller_y_count



    def draw_max_y_contour(self, image_frame, max_y_contour, max_y_color, x_offset, y_offset):
        """Draw the contour with the largest y-coordinate."""
        cx, cy, cw, ch = max_y_contour
        cv2.rectangle(image_frame, (x_offset + cx, y_offset + cy), (x_offset + cx + cw, y_offset + cy + ch),
                    self.get_color_for_display(max_y_color), 2)
        cv2.putText(image_frame, f"{max_y_color.capitalize()} (Player Turn)", (x_offset + cx, y_offset + cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.get_color_for_display(max_y_color), 2)
        


    def display_time_of_day(self, image_frame, smoothed_smaller_y_count):
        """Display the count of contours with smaller y-values."""
        
        if smoothed_smaller_y_count is None:
            return 0
        display_time = smoothed_smaller_y_count + 6
        suffix = "pm"
        if display_time > 12:
                display_time -= 12
        else: 
            suffix=  "am"
        cv2.putText(image_frame, f"Time of Day: {display_time} {suffix}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    def draw_roi_box(self, image_frame, x, y, w, h):
        """Draw the green ROI box."""
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_frame, "ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    def draw_y_axis_arrow(self, image_frame):
        """Draw an arrow indicating the direction of the y-axis."""
        height, width, _ = image_frame.shape
        cv2.arrowedLine(image_frame, (width - 50, 50), (width - 50, 150), (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(image_frame, "+Y", (width - 70, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def weighted_smoothing(self, deque_list):
        """Apply weighted smoothing to prioritize recent values."""
        weights = np.linspace(1, len(deque_list), len(deque_list))
        smoothed_value = np.average(deque_list, weights=weights)
        return int(smoothed_value)

    
    def get_smoothed_color_time(self):
        
        smoothed_max_y_colour = None
        smoothed_smaller_y_count = None
        # Apply moving average smoothing to the counts for each color
        if not self.smaller_y_over_time or not self.max_colour_over_time:  # Check if the list/array is empty
            return smoothed_max_y_colour, smoothed_smaller_y_count  # Or another default value
        
        counter = Counter(self.max_colour_over_time)
        most_common = counter.most_common(1)
        if most_common:
            smoothed_max_y_colour = most_common[0][0]  # Extract the string
            print(f"Most present string: {smoothed_max_y_colour}")
        else:
            print("Deque is empty")
            
        #Calculate smoothed smaller y count
        smoothed_smaller_y_count = self.weighted_smoothing(self.smaller_y_over_time)
            
        return smoothed_max_y_colour, smoothed_smaller_y_count
    
        
    
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
        """Display the processed frame."""
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
    color_detection = ColorDetectionWithROI(smoothing_window_size=10,transition_window_size=10,stability_threshold=5)
    color_detection.run()
