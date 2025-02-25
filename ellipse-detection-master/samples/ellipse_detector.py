#  coding: utf-8
print("Hello, World!")
import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(parent_dir)
print("Python path:", sys.path)

import cv2

# Try to import the ellipse_detection module and catch any errors
try:
    import ellipse_detection as ed
    print("ellipse_detection module imported successfully.")
except ImportError as e:
    print("Failed to import ellipse_detection module:", e)
    sys.exit(1)
    
# Check if the imports are successful
print("Checking imports...")
assert 'Ellipse' in dir(ed), "Failed to import Ellipse"
assert 'Segment' in dir(ed), "Failed to import Segment"
assert 'SegmentPair' in dir(ed), "Failed to import SegmentPair"
assert 'EllipseCandidate' in dir(ed), "Failed to import EllipseCandidate"
assert 'SegmentDetector' in dir(ed), "Failed to import SegmentDetector"
assert 'EllipseCenterEstimator' in dir(ed), "Failed to import EllipseCenterEstimator"
assert 'EllipseCandidateMaker' in dir(ed), "Failed to import EllipseCandidateMaker"
assert 'EllipseEstimator' in dir(ed), "Failed to import EllipseEstimator"
assert 'EllipseMerger' in dir(ed), "Failed to import EllipseMerger"
assert 'EllipseDetector' in dir(ed), "Failed to import EllipseDetector"
print("All imports are successful.")


def main():
    image = cv2.imread('../images/493.jpg', 1)
    # image = cv2.imread('../images/sP1010080.jpg', 1)
    # image = cv2.imread('../images/20091031193703238.jpg', 1)
    # image = cv2.imread('../images/KM-612.jpg', 1)
    # image = cv2.imread('../images/49days-301.jpg', 1)
    # image = cv2.imread('../images/ellipse.png', 1)
    # image = cv2.imread('../images/83I83Z838D82P89F196DA81B.JPG', 1)
    # image = cv2.imread('../images/20120827-003.jpg', 1)
    # image = cv2.imread('../images/ellipse2.png', 1)
    # mage = cv2.imread('../images/rot-ellipse.png', 1)
    # image = cv2.imread('../images/rot-ellipse-2.png', 1)

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ellipse_detector = ed.EllipseDetector()
    ellipses = ellipse_detector.detect(image_gray)

    for ellipse in ellipses:
        image_ellipse = image.copy()
        ellipse.draw(image_ellipse)
        cv2.imshow('ellipse', image_ellipse)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
