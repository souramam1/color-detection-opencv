import numpy as np
import cv2


class ContourProcessing:
    
    def __init__(self, webcam_index=1):
        self.webcam = cv2.VideoCapture(webcam_index)
        self.detected_token_contours = []
        
    
    def detect_and_draw_contours(self, gray_frame, hsv_frame):
        
        #generates gaussian blur on grayscale image
        blurred = self.apply_gaussian_blur(gray_frame)
        #generates canny edge detection on blurred image
        edges = self.apply_canny_edge_detection(blurred)
        #finds contours on canny edge detection
        contours_canny = self.find_contours(edges)
        #draws all canny edge detected contours on frame
        frame_copy = self.draw_initial_contours(gray_frame, contours_canny)
        cv2.imshow("gray frame copy canny contours", frame_copy)
        
        #finds largest region of interest
        roi = self.find_largest_roi(contours_canny)
        if roi:
            #goes through all contours and identifies ones in the roi of right size
            frame_with_contours = self.draw_roi_contours(hsv_frame, contours_canny, roi)
        return frame_with_contours, roi

    def apply_gaussian_blur(self, gray_frame):
        return cv2.GaussianBlur(gray_frame, (5, 5), 0)

    def apply_canny_edge_detection(self, blurred):
        edges = cv2.Canny(blurred, 20, 100)
        cv2.imshow("Canny Edge Detection", edges)
        return edges

    def find_contours(self, edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_initial_contours(self, gray_frame, contours):
        frame_copy = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(frame_copy, contours, -1, (255, 0, 0), 1)
        return frame_copy

    def find_largest_roi(self, contours):
        largest_area = 0
        roi = None
        for contour in contours:
            _, size, _ = cv2.minAreaRect(contour)
            area = size[0] * size[1]
            if area > 90000 and area > largest_area:
                largest_area = area
                x, y, w, h = cv2.boundingRect(contour)
                roi = (x, y, w, h)
                print(f"ROI: {roi}")
        return roi

    def draw_roi_contours(self, frame_copy, contours, roi):
        
        roi_x, roi_y, roi_w, roi_h = roi
        cv2.rectangle(frame_copy, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
        
        self.detected_token_contours = []
        
        for contour in contours:
            # minAreaRect can deal with rotated rectangles
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            x, y = rect[0][0], rect[0][1]
            width, height = rect[1][0], rect[1][1]
            area = width * height
            
            if 300 < area <= 800:
                if roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h:
                    #cv2.drawContours(frame_copy, [box], 0, (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"{area:.0f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                    
                    # Every frame, all tokens of size are found and added to a list
                    # List of tokens is emptied each new frame.
                    self.detected_token_contours.append(contour)
        return frame_copy

    def show_result(self, frame):
        cv2.imshow("Contour detection with Canny", frame)

    def process_frame_old(self, frame):
    
        if frame is None:
            return None, None, None
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bgr_frame = frame
        hsv_frame_with_canny_contours, roi = self.detect_and_draw_contours(gray_frame,bgr_frame)
        
        cv2.imshow("Bounded contours detected", hsv_frame_with_canny_contours)
        return hsv_frame_with_canny_contours, bgr_frame, roi
    
    def process_frame(self, frame: np.ndarray) -> list:
        '''Processes camera frame and returns list of isolated rectangular token box coordinates
        
        Parameters:
            frame (np.ndarray): The input frame, a NumPy array
            
        Returns:
            isolated_token_coords: the list of corner points corresponding to tokens identified as being within the region of interest
        
        '''
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        
        
        

    def run(self):
        try:
            while True:
                frame = self.process_frame()
                if frame is not None:
                    self.show_result(frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            print("Cleaning up")


        
if __name__ == "__main__":
    contour_detection = ContourProcessing()
    contour_detection.run()