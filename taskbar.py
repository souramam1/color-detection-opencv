import numpy as np
import cv2

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Default HSV ranges for red, green, and blue
red_lower, red_upper = np.array([136, 87, 111]), np.array([180, 255, 255])
green_lower, green_upper = np.array([25, 52, 72]), np.array([102, 255, 255])
blue_lower, blue_upper = np.array([94, 80, 2]), np.array([120, 255, 255])

# Calibration mode toggle
calibration_mode = False
color_ranges = {"red": [red_lower, red_upper],
                "green": [green_lower, green_upper],
                "blue": [blue_lower, blue_upper]}

def nothing(x):
    pass

# Create trackbars for calibration
def create_calibration_window():
    cv2.namedWindow("Calibration")
    for color in color_ranges:
        cv2.createTrackbar(f"{color} Low H", "Calibration", 0, 179, nothing)
        cv2.createTrackbar(f"{color} High H", "Calibration", 179, 179, nothing)
        cv2.createTrackbar(f"{color} Low S", "Calibration", 0, 255, nothing)
        cv2.createTrackbar(f"{color} High S", "Calibration", 255, 255, nothing)
        cv2.createTrackbar(f"{color} Low V", "Calibration", 0, 255, nothing)
        cv2.createTrackbar(f"{color} High V", "Calibration", 255, 255, nothing)

def update_color_ranges():
    for color in color_ranges:
        low_h = cv2.getTrackbarPos(f"{color} Low H", "Calibration")
        high_h = cv2.getTrackbarPos(f"{color} High H", "Calibration")
        low_s = cv2.getTrackbarPos(f"{color} Low S", "Calibration")
        high_s = cv2.getTrackbarPos(f"{color} High S", "Calibration")
        low_v = cv2.getTrackbarPos(f"{color} Low V", "Calibration")
        high_v = cv2.getTrackbarPos(f"{color} High V", "Calibration")
        color_ranges[color] = [np.array([low_h, low_s, low_v]), np.array([high_h, high_s, high_v])]

# Create calibration window
create_calibration_window()

# Main loop
while True:
    _, imageFrame = webcam.read()

    # Convert to HSV
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    masks = {}  # Store masks for display
    if calibration_mode:
        # Update HSV ranges dynamically
        update_color_ranges()

    # Apply masks for each color
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsvFrame, lower, upper)
        kernel = np.ones((5, 5), "uint8")
        mask = cv2.dilate(mask, kernel)
        masks[color] = mask  # Store mask for display

        if not calibration_mode:
            # Draw contours and labels on the main feed
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    color_map = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
                    cv2.rectangle(imageFrame, (x, y), (x + w, y + h), color_map[color], 2)
                    cv2.putText(imageFrame, f"{color.capitalize()} Colour", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_map[color], 2)

    # Show the main detection feed
    cv2.imshow("Webcam Feed", imageFrame)

    # Show individual color masks in separate windows
    for color, mask in masks.items():
        cv2.imshow(f"{color.capitalize()} Mask", mask)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        calibration_mode = not calibration_mode  # Toggle calibration mode
        if calibration_mode:
            print("Entered Calibration Mode. Adjust HSV ranges using trackbars.")
        else:
            print("Exited Calibration Mode.")
    elif key == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
