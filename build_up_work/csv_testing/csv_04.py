import os
import csv
import time
from datetime import datetime
import threading

# A VERSION WITH 1 THREAD RUNNING AN INTERRUPT SO THAT OPERATION CAN CONTINUE ON THE MAIN THREAD (i.e image processing)

# Create a new folder within the current working directory
folder_name = "data_files"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Generate a unique file name based on the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = os.path.join(folder_name, f"data_{timestamp}.csv")

# Create and initialize the CSV file with headers
with open(file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["orange", "yellow", "magenta", "teal"])
    writer.writeheader()

print(f"Data will be saved to: {file_path}")
print("Writing data every 2 seconds. Press Ctrl+C to stop.")

# Function to save data to the CSV file
def save_data():
    data = {
        "orange": 10,
        "yellow": 20,
        "magenta": 30,
        "teal": 40,
    }
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["orange", "yellow", "magenta", "teal"])
        writer.writerow(data)
    print("Data written to file:", data)
    
    # Schedule the next call to `save_data` in 2 seconds
    threading.Timer(2, save_data).start()

# Start the initial call to save data
save_data()

# Keep the main thread alive (this prevents the program from exiting immediately)
try:
    while True:
        time.sleep(1)  # Sleep just to keep the program running
except KeyboardInterrupt:
    print("\nProgram stopped.")
