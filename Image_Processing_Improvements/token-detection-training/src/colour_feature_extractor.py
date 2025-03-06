import cv2
import numpy as np
import os
import pandas as pd
import re
from datetime import datetime

class ColorFeatureExtractor:
    """Extract mean HSV and RGB values from a folder of images and save to a CSV file.
    These will be used to train a KNN classifier for color recognition."""
    
    def __init__(self, base_folder_path):
        self.base_folder_path = base_folder_path
        self.data = []

    def extract_features(self, image_path):
        """Extract mean HSV and RGB values from an image."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not open {image_path}")
            return None
        
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract central region (to avoid background influence)
        h, w, _ = image.shape
        center_region = image[h//4: 3*h//4, w//4: 3*w//4]
        center_hsv = hsv_image[h//4: 3*h//4, w//4: 3*w//4]

        # Compute mean HSV and RGB values
        mean_hsv = np.mean(center_hsv, axis=(0, 1))
        mean_rgb = np.mean(center_region, axis=(0, 1))

        return mean_hsv, mean_rgb

    def get_label_from_filename(self, filename):
        """Extract the color label from the filename."""
        print(f"filename: {filename}")
        match = re.search(r"(yellow|magenta|cyan|green)", filename, re.IGNORECASE)
        return match.group(1).lower() if match else "unknown"

    def process_images_in_folder(self, folder_path):
        """Process all images in a specific folder and extract features."""
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(folder_path, filename)
                mean_hsv, mean_rgb = self.extract_features(image_path)

                if mean_hsv is not None:
                    label = self.get_label_from_filename(filename)
                    self.data.append([
                        mean_hsv[0], mean_hsv[1], mean_hsv[2],  # HSV values
                        mean_rgb[0], mean_rgb[1], mean_rgb[2],  # RGB values
                        label
                    ])
        print(f"Processed {len(self.data)} images in {folder_path}.")

    def process_all_folders(self):
        """Process all color folders and extract features."""
        color_folders = ["yellow_tokens", "green_tokens", "magenta_tokens", "cyan_tokens"]
        for color_folder in color_folders:
            folder_path = os.path.join(self.base_folder_path, color_folder)
            if os.path.exists(folder_path):
                self.process_images_in_folder(folder_path)
            else:
                print(f"Folder {folder_path} does not exist.")

    def save_to_csv(self):
        """Save extracted features to a CSV file inside the folder."""
        csv_folder = os.path.join(self.base_folder_path, "CSV_files")
        os.makedirs(csv_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"Labels_HSV_RGB_{timestamp}.csv"
        csv_path = os.path.join(csv_folder, csv_filename)
        df = pd.DataFrame(self.data, columns=["Hue", "Saturation", "Value", "Red", "Green", "Blue", "Label"])
        df.to_csv(csv_path, index=False)
        print(f"CSV file saved to {csv_path}")

    def run(self):
        """Run the full pipeline."""
        self.process_all_folders()
        self.save_to_csv()

# Example usage
if __name__ == "__main__":
    base_folder = "Image_Processing_Improvements/token-detection-training/labelled_data"  # Change this to your actual base folder path
    extractor = ColorFeatureExtractor(base_folder)
    extractor.run()
