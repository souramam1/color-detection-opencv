import numpy as np
import cv2
from skimage.io import imread
from skimage import img_as_ubyte

def whitepatch_balancing(image, roi):
    
    x, y, w, h = roi
    patch_size = 60
    
    # Calculate the coordinates for the patch in the bottom left-hand corner of the ROI
    from_row = y + h - 20 - patch_size
    from_column = x + 20
    
    # Ensure the patch is within the bounds of the image
    from_row = min(from_row, image.shape[0] - patch_size)
    from_column = min(from_column, image.shape[1] - patch_size)
    
    # Draw the ROI and patch rectangles on the original image
    image_with_rectangles = image.copy()
    cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle for ROI
    cv2.rectangle(image_with_rectangles, (from_column, from_row), (from_column + patch_size, from_row + patch_size), (255, 0, 0), 2)  # Blue rectangle for patch
    
    # Extract the patch and perform white patch balancing
    image_patch = image[from_row:from_row + patch_size, from_column:from_column + patch_size]
    image_max = (image * 1.0 / image_patch.max(axis=(0, 1))).clip(0, 1)
    image_max = (image_max * 255).astype(np.uint8)
        
        # Convert images from RGB to BGR for OpenCV display
    image_with_rectangles = cv2.cvtColor(image_with_rectangles, cv2.COLOR_RGB2BGR)
    image_max = cv2.cvtColor(image_max, cv2.COLOR_RGB2BGR)
    
    return image_with_rectangles, image_max

# Load your image
image_path = r'Image_Processing_Improvements\captured_image.jpg'
dinner = imread(image_path)
#dinner = img_as_ubyte(dinner)
roi = (57, 49, 467, 355)

# Apply white patch balancing
image_with_rectangles, whitebalanced_image = whitepatch_balancing(dinner, roi)

# Concatenate the original and whitebalanced images side by side
combined_image = cv2.hconcat([image_with_rectangles, whitebalanced_image])

# Display the combined image
cv2.imshow('Original and Whitebalanced Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()