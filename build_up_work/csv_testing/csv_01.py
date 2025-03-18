import os
import csv
import time
from datetime import datetime
import keyboard

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
print("Press 'W' to save data. Press 'Esc' to exit.")

# Function to save data to the CSV file
def save_data():
    data = {
        "orange": 10,  # Replace with your desired integer value
        "yellow": 20,  # Replace with your desired integer value
        "magenta": 30,  # Replace with your desired integer value
        "teal": 40,  # Replace with your desired integer value
    }
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["orange", "yellow", "magenta", "teal"])
        writer.writerow(data)
    print("Data saved:", data)

# Listen for key press events
try:
    while True:
        if keyboard.is_pressed('w'):  # Press 'W' to save data
            save_data()
            time.sleep(0.2)  # Small delay to prevent duplicate writes
