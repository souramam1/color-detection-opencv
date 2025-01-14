import numpy as np
import cv2
from picamera2 import Picamera2

class ColorDetectionWithMarkers:
    def __init__(self, smoothing_window_size=5, resolution=(640, 480), format="RGB888"):
        self.picam2 = Picamera2()
        self.resolution = resolution
        self.format = format
        self.configure_camera()
        
        # Define HSV range for green markers (adjust if needed)
        self.green_range = ((40, 50, 50), (80, 255, 255))  # HSV range for green
        
        # Define HSV ranges for other colors to detect
        self.color_ranges = {
            "orange": ((0, 50, 50), (10, 255, 255)),
            "yellow": ((15, 50, 50), (40, 255, 255)),
            "magenta": ((140, 50, 50), (170, 255, 255)),
            "teal": ((85, 50, 50), (100, 255, 255))
        }
        
        self.kernel = np.ones((5, 5), "uint8")
        self.smoothing_window_size = smoothing_window_size

    def configure_camera(self):
        """Configure the camera with specified resolution and format."""
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": self.format, "size": self.resolution}))
        self.picam2.start()
    
    def capture_frame(self):
        """Capture a frame from the Picamera2."""
        frame = self.picam2.capture_array()
        return frame
    
    def detect_green_markers(self, hsv_frame):
        """Detect four green markers and return their coordinates."""
        mask = cv2.inRange(hsv_frame, *self.green_range)
        mask = cv2.dilate(mask, self.kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        marker_positions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Adjust threshold for marker size
                # Approximate the contour to a polygon
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Looking for rectangular markers
                    (x, y, w, h) = cv2.boundingRect(approx)
                    marker_positions.append((x + w // 2, y + h // 2))  # Use center of marker

        if len(marker_positions) == 4:
            # Sort marker positions to form a quadrilateral
            marker_positions = self.sort_points(marker_positions)
            return marker_positions
        return None
    
    def sort_points(self, points):
        """Sort points to form a bounding quadrilateral."""
        points = sorted(points, key=lambda p: p[0])  # Sort by x-coordinate
        left = sorted(points[:2], key=lambda p: p[1])  # Top-left, bottom-left
        right = sorted(points[2:], key=lambda p: p[1])  # Top-right, bottom-right
        return left + right  # Return in order: TL, BL, TR, BR
    
    def get_roi_mask(self, frame, markers):
        """Create a mask for the ROI based on marker positions."""
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        pts = np.array(markers, dtype="int32")
        cv2.fillPoly(mask, [pts], 255)  # Fill the quadrilateral
        return mask

    def process_frame(self):
        """Capture the frame and process it for markers and ROI."""
        image_frame = self.capture_frame()
        hsv_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)

        # Detect the four green markers
        markers = self.detect_green_markers(hsv_frame)
        if markers:
            # Create ROI mask from markers
            roi_mask = self.get_roi_mask(image_frame, markers)

            # Detect other colors within the ROI
            masks = {}
            for color, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(hsv_frame, lower, upper)
                mask = cv2.bitwise_and(mask, roi_mask)  # Apply ROI mask
                masks[color] = mask

            return image_frame, masks, markers
        return image_frame, {}, None

    def detect_and_draw_contours(self, image_frame, masks, markers):
        """Detect and draw contours inside the ROI."""
        if markers:
            # Draw the ROI quadrilateral
            for i in range(len(markers)):
                cv2.line(image_frame, markers[i], markers[(i + 1) % 4], (0, 255, 0), 2)
            
            for color, mask in masks.items():
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Minimum contour area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(image_frame, (x, y), (x + w, y + h), self.get_color_for_display(color), 2)
                        cv2.putText(image_frame, f"{color.capitalize()}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.get_color_for_display(color), 2)
        
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
        cv2.imshow("Color Detection with Markers", image_frame)
    
    def run(self):
        """Main loop to detect and display contours within the ROI."""
        try:
            while True:
                image_frame, masks, markers = self.process_frame()
                image_frame = self.detect_and_draw_contours(image_frame, masks, markers)
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
    color_detection = ColorDetectionWithMarkers(smoothing_window_size=10)
    color_detection.run()
