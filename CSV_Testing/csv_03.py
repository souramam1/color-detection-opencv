import os
import csv
import time
from datetime import datetime

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

# Write data to the file every 2 seconds
try:
    while True:
        save_data()
        time.sleep(2)  # Wait for 2 seconds before the next write
except KeyboardInterrupt:
    print("\nProgram stopped.")
