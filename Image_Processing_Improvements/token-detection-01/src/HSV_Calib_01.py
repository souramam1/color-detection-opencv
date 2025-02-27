import cv2
import numpy as np

class HSVCalibrator:
    def __init__(self, camera_index=1):
        self.cap = cv2.VideoCapture(camera_index)
        self.calibrated_values = {}
        self.colors = ["red", "orange", "yellow", "magenta"]
    
    def nothing(self, x):
        pass
    
    def create_trackbar_window(self, window_name):
        cv2.namedWindow(window_name)
        cv2.createTrackbar("Low H", window_name, 0, 179, self.nothing)
        cv2.createTrackbar("High H", window_name, 179, 179, self.nothing)
        cv2.createTrackbar("Low S", window_name, 0, 255, self.nothing)
        cv2.createTrackbar("High S", window_name, 255, 255, self.nothing)
        cv2.createTrackbar("Low V", window_name, 0, 255, self.nothing)
        cv2.createTrackbar("High V", window_name, 255, 255, self.nothing)
    
    def get_trackbar_values(self, window_name):
        low_h = cv2.getTrackbarPos("Low H", window_name)
        high_h = cv2.getTrackbarPos("High H", window_name)
        low_s = cv2.getTrackbarPos("Low S", window_name)
        high_s = cv2.getTrackbarPos("High S", window_name)
        low_v = cv2.getTrackbarPos("Low V", window_name)
        high_v = cv2.getTrackbarPos("High V", window_name)
        return (low_h, low_s, low_v), (high_h, high_s, high_v)
    
    def calibrate_color(self, colour):
        print(f"Calibrating {colour}... Adjust the sliders and press 'c' to confirm.")
        window_name = f"HSV Calibration - {colour}"
        self.create_trackbar_window(window_name)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower, upper = self.get_trackbar_values(window_name)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            cv2.imshow("Original", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Filtered", result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.calibrated_values[colour] = (lower, upper)
                print(f"Saved {colour} values: Lower HSV {lower}, Upper HSV {upper}")
                cv2.destroyWindow(window_name)
                break
    
    def run(self):
        for colour in self.colors:
            self.calibrate_color(colour)
        
        print("Final calibrated values:")
        for colour, (lower, upper) in self.calibrated_values.items():
            print(f"{colour}: Lower {lower}, Upper {upper}")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = HSVCalibrator()
    calibrator.run()
