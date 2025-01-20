import os
import csv
import time
import threading
from datetime import datetime

class DataWriter:
    def __init__(self, folder_name="data_files"):
        self.folder_name = folder_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.folder_name, f"data_{self.timestamp}.csv")

        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        # Initialize CSV with headers for color counts
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["orange", "yellow", "magenta", "teal"])
            writer.writeheader()

        print(f"Data will be saved to: {self.file_path}")
    
    def write_data(self, color_counts):
        """Write color counts to the CSV file."""
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["orange", "yellow", "magenta", "teal"])
            writer.writerow(color_counts)
        print(f"Data written to file: {color_counts}")

    def start_timer(self, color_detection):
        """Start a timer to write data every 2 seconds."""
        def write_periodically():
            color_counts = {color: color_detection.get_smoothed_count(color) for color in color_detection.color_ranges}
            self.write_data(color_counts)
            threading.Timer(2, write_periodically).start()  # Re-run every 2 seconds
        
        write_periodically()
