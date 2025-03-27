#  coding: utf-8

import sys
import os

# # Add the parent directory to the system path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import cv2
    from total_ellipse_detection_class import TotalEllipseDetector
    
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import sure the 'total_ellipse_detection_class' ")
    sys.exit(1)


def main():
    image_path = '../images/493.jpg'
    image = cv2.imread(image_path, 1)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ellipse_detector = ed.EllipseDetector()
    ellipses = ellipse_detector.detect(image_gray)

    for ellipse in ellipses:
        image_ellipse = image.copy()
        ellipse.draw(image_ellipse)
        cv2.imshow('ellipse', image_ellipse)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
