import numpy as np
import cv2
from skimage import img_as_ubyte


class ColorDetection:
    
    init_hsv_calib_val = {
        "red": ((0, 100, 100), (10, 255, 255)),
        "orange": ((0, 0, 50), (16, 255, 255)),
        "yellow": ((16, 50, 50), (39, 255, 255)),
        "magenta": ((116, 52, 50), (179, 255, 255)),   
}
    
    def __init__(self, final_hsv_calib = init_hsv_calib_val):
        self.color_ranges = final_hsv_calib
        print(f"color range in hsv calibrated to : {self.color_ranges}")

        self.color_bgr = {
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'yellow': (0, 255, 255),
            'magenta': (255, 0, 255),
            'unknown': (255, 255, 255)
        }
        
    def classify_contour(self, contour, bgr_frame):
        # convert cv2 bgr to hsv
        hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(hsv_frame, mask=mask)[:3]
        
        for color, (lower, upper) in self.color_ranges.items():
            if cv2.inRange(np.uint8([[mean_val]]), np.array(lower), np.array(upper)):
                return color
        return 'unknown'

    def draw_contours(self, frame,roi, contours, bgr_frame):
        for contour in contours:
            _ , bgr_white_balanced_frame = self.whitepatch_balancing(bgr_frame, roi)
            cv2.imshow("white balanced frame", bgr_white_balanced_frame)
            color = self.classify_contour(contour, bgr_white_balanced_frame)
            if color != 'unknown':
                print(f"Color: {self.color_bgr[color]}")
                # Draw the coloured contour but with the contour having the same colour as the detected colour
                cv2.drawContours(frame, [contour], -1, (255,0,0), 2)
                x, y, w, h = cv2.boundingRect(contour)
                text_color = self.color_bgr[color]
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            else:
                print("Unknown colour detected")
      
                
    def whitepatch_balancing(self,frame, roi):
        
        x, y, w, h = roi
        patch_size = 60
        
        # Calculate the coordinates for the patch in the bottom left-hand corner of the ROI
        from_row = y + h - 20 - patch_size
        from_column = x + 20
        
        # Ensure the patch is within the bounds of the frame
        from_row = min(from_row, frame.shape[0] - patch_size)
        from_column = min(from_column, frame.shape[1] - patch_size)
        
        # Draw the ROI and patch rectangles on the original frame
        image_with_rectangles = frame.copy()
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle for ROI
        cv2.rectangle(image_with_rectangles, (from_column, from_row), (from_column + patch_size, from_row + patch_size), (255, 0, 0), 2)  # Blue rectangle for patch
        
        # Extract the patch and perform white patch balancing
        image_patch = frame[from_row:from_row + patch_size, from_column:from_column + patch_size]
        image_max = (frame * 1.0 / image_patch.max(axis=(0, 1))).clip(0, 1)
        image_max = (image_max * 255).astype(np.uint8)
            
        #     # Convert images from RGB to BGR for OpenCV display
        # image_with_rectangles = cv2.cvtColor(image_with_rectangles, cv2.COLOR_RGB2BGR)
        # image_max = cv2.cvtColor(image_max, cv2.COLOR_RGB2BGR)
        
        return image_with_rectangles, image_max

