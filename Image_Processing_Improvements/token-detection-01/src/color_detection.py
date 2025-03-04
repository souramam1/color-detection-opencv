import numpy as np
import cv2
from skimage import img_as_ubyte


class ColorDetection:
    
    init_hsv_calib_val = {
        # Define the HSV ranges for each colour - disctinct Hues are preferred
        "yellow": ((16, 50, 50), (39, 255, 255)),
        "green": ((71, 87, 42), (91, 255, 255)),
        "cyan" : ((93, 186, 61), (108, 255, 255)),
        "magenta": ((116, 52, 50), (179, 255, 255))      
    }
    
    def __init__(self, final_hsv_calib = init_hsv_calib_val):
        
        self.color_ranges = final_hsv_calib
        print(f"color range in hsv calibrated to : {self.color_ranges}")

        self.color_bgr = {
            'yellow': (0, 255, 255),
            'green': (0, 128, 0),
            'cyan': (255, 255, 0),
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
                print(f"Mean val of {mean_val} generates color detected: {color}")
                return color
        return 'unknown'

    def draw_contours(self, frame, contours, bgr_frame, image_patch):
        for contour in contours:
            bgr_white_balanced_frame = self.whitepatch_balancing(bgr_frame, image_patch)
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
      
                
    def whitepatch_balancing(self,frame, image_patch):
        
        # Perform white patch balancing, using the frame and patch
        image_max = (frame * 1.0 / image_patch.max(axis=(0, 1))).clip(0, 1)
        image_max = (image_max * 255).astype(np.uint8)
            

        return image_max

