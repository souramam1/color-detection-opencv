import os
import csv
from datetime import datetime
from pynput import keyboard

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
        "orange": 10,
        "yellow": 20,
        "magenta": 30,
        "teal": 40,
    }
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["orange", "yellow", "magenta", "teal"])
        writer.writerow(data)
    print("Data saved:", data)

# Define the key listener
def on_press(key):
    try:
        if key.char == 'w':  # Press 'W' to save data
            save_data()
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:  # Press 'Esc' to exit
        print("Exiting program.")
        return False

# Start listening to the keyboard
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
