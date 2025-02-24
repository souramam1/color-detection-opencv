import cv2
import numpy as np
from skimage import color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# Load picture using OpenCV
image_path = r"C:\Users\MaiaRamambason\OneDrive - Imperial College London\Desktop\Year5\UCL_BIP\Object_Detection_Trial\color-detection-opencv\Image_Processing_Improvements\captured_image.jpg"
image_rgb = cv2.imread(image_path)

if image_rgb is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully.")

# Convert to grayscale
cv2.imshow('Original Image', image_rgb)
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 80, 100)

# Convert edges to a format suitable for skimage
edges_skimage = img_as_ubyte(edges)

# Perform a Hough Transform
result = hough_ellipse(edges_skimage, accuracy=20, threshold=250, min_size=100, max_size=120)
result.size = 0

# Check if any ellipses were found
if result.size > 0:
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = (int(round(x)) for x in best[1:5])
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_rgb[cy, cx] = (0, 0, 255)

    # Concatenate images for side-by-side display
    combined_image = np.hstack((image_rgb, edges_rgb))

    # Display the images using OpenCV
    cv2.imshow('Original and Edges with Ellipse', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No ellipses were found.")
    
    # Concatenate images for side-by-side display
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined_image = np.hstack((image_rgb, edges_rgb))

    # Display the images using OpenCV
    cv2.imshow('Original and Canny Edges', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()