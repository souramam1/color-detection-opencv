import numpy as np
import cv2


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

    def draw_contours(self, frame, contours, bgr_frame):
        for contour in contours:
            color = self.classify_contour(contour, bgr_frame)
            if color != 'unknown':
                print(f"Color: {self.color_bgr[color]}")
                # Draw the coloured contour but with the contour having the same colour as the detected colour
                cv2.drawContours(frame, [contour], -1, (255,0,0), 2)
                x, y, w, h = cv2.boundingRect(contour)
                text_color = self.color_bgr[color]
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            else:
                print("Unknown colour detected")

