import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_ubyte
import cv2

def grayworld_balancing(img):
    img = img.astype(np.float32)
    avgR = np.mean(img[:, :, 0])
    avgG = np.mean(img[:, :, 1])
    avgB = np.mean(img[:, :, 2])
    
    avgGray = (avgR + avgG + avgB) / 3
    
    img[:, :, 0] = np.minimum(img[:, :, 0] * (avgGray / avgR), 255)
    img[:, :, 1] = np.minimum(img[:, :, 1] * (avgGray / avgG), 255)
    img[:, :, 2] = np.minimum(img[:, :, 2] * (avgGray / avgB), 255)
    
    return img.astype(np.uint8)

# Load your image
image_path = r'Image_Processing_Improvements\captured_image.jpg'
dinner = imread(image_path)
dinner = img_as_ubyte(dinner)

# Apply grayworld balancing
dinner_balanced = grayworld_balancing(dinner)

# Concatenate the original and balanced images side by side
combined_image = cv2.hconcat([dinner, dinner_balanced])

# Convert RGB to BGR for OpenCV display
combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

# Display the combined image
cv2.imshow('Original and Grayworld Balanced Image', combined_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

