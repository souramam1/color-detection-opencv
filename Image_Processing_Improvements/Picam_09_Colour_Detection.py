import cv2
import numpy as np
from Picam_09 import ContourDetection

class ColorClassifier:
    def __init__(self):
        # Define color ranges in HSV
        self.color_ranges = {
            'orange': ((5, 50, 50), (15, 255, 255)),
            'yellow': ((25, 50, 50), (35, 255, 255)),
            'magenta': ((140, 50, 50), (160, 255, 255))
        }

    def classify_contour(self, contour, frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(frame, mask=mask)[:3]
        hsv_mean = cv2.cvtColor(np.uint8([[mean_val]]), cv2.COLOR_BGR2HSV)[0][0]

        for color, (lower, upper) in self.color_ranges.items():
            if cv2.inRange(np.uint8([[hsv_mean]]), np.array(lower), np.array(upper)):
                return color
        return 'unknown'

    def draw_contours(self, frame, contours):
        for contour in contours:
            color = self.classify_contour(contour, frame)
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    # Load your frame here
    frame = cv2.imread('path_to_your_image.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    classifier = ColorClassifier()
    classifier.draw_contours(frame, contours)

    cv2.imshow('Classified Contours', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    contour_detection = ContourDetection()
    contour_detection.run()
    main()