import os
import csv
import cv2
import time
import threading
from datetime import datetime

class DataWriter:
    def __init__(self, folder_name="data_files"):
        self.folder_name = folder_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.folder_name, f"data_{self.timestamp}.csv")
        self.image_folder = os.path.join(self.folder_name, f"images_{self.timestamp}")

        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        # Initialize CSV with headers for color counts
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["orange", "yellow", "magenta", "teal"])
            writer.writeheader()

        print(f"Data will be saved to: {self.file_path}")
        print(f"Images will be saved to: {self.image_folder}")
        
    
    def write_data(self, color_counts):
        """Write color counts to the CSV file."""
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["orange", "yellow", "magenta", "teal"])
            writer.writerow(color_counts)
        print(f"Data written to file: {color_counts}")
        
    def capture_and_save_image(self,camera):
        image_array = camera.capture_array()
        image_name = os.path.join(self.image_folder, f"image_{self.timestamp}.png")
        cv2.imwrite(image_name, image_array)
        print(f"Image captured and saved as: {image_name}")
        

    def start_timer_tocsv(self, color_detection):
        """Start a timer to write data every 2 seconds."""
        def write_periodically():
            color_counts = {color: color_detection.get_smoothed_count(color) for color in color_detection.color_ranges}
            self.write_data(color_counts)
            threading.Timer(2, write_periodically).start()  # Re-run every 2 seconds
        
        write_periodically()
        
    def start_timer_toimage(self, camera):
        """Start a timer to capture image every 15 seconds."""
        def save_image_periodically():
            self.capture_and_save_image(camera)
            threading.Timer(15, save_image_periodically).start()
        save_image_periodically()
