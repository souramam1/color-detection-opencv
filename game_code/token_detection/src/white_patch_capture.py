import numpy as np
import cv2

class WhitePatchCapture:
    def __init__(self):
        self.image_patch = None

    def select_image_patch(self, frame):
        patch_size = 60
        center_y, center_x = frame.shape[0] // 2, frame.shape[1] // 2
        
        # Calculate the coordinates for the patch in the center of the frame
        from_row = center_y - patch_size // 2
        from_column = center_x - patch_size // 2
        
        # Ensure the patch is within the bounds of the frame
        from_row = max(0, min(from_row, frame.shape[0] - patch_size))
        from_column = max(0, min(from_column, frame.shape[1] - patch_size))
        
        # Draw the patch rectangle on the original frame
        image_with_rectangles = frame.copy()
        cv2.rectangle(image_with_rectangles, (from_column, from_row), (from_column + patch_size, from_row + patch_size), (255, 0, 0), 2)  # Blue rectangle for patch
        
        # Extract the patch
        image_patch = frame[from_row:from_row + patch_size, from_column:from_column + patch_size]
        self.image_patch = image_patch
        
        # Show the image with rectangles
        cv2.imshow("Image with rectangles", image_with_rectangles)
        
        return image_with_rectangles, image_patch
    
    def calculate_image_max(self, frame, image_patch):
        ''' Applies whitepatch balancing to input frame based on image patch from calibration phase
        
        Parameter: 
            frame: np.ndarray: image frame to be balanced
            image_patch: np.ndarray : small chunk of image (ideally white background)
            
        Returns:
            frame: np.ndarray: whitepatch balanced frame
            
        '''
        # Perform white patch balancing, using the frame and patch
        image_max = (frame * 1.0 / image_patch.max(axis=(0, 1))).clip(0, 1)
        image_max = (image_max * 255).astype(np.uint8)   
        
        return image_max

