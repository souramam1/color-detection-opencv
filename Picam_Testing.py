import cv2
import numpy as np

def nothing(x):
    pass

# Open webcam
cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow("HSV Calibration")

# Create trackbars for lower HSV values
cv2.createTrackbar("Lower H", "HSV Calibration", 40, 180, nothing)
cv2.createTrackbar("Lower S", "HSV Calibration", 50, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Calibration", 50, 255, nothing)

# Create trackbars for upper HSV values
cv2.createTrackbar("Upper H", "HSV Calibration", 80, 180, nothing)
cv2.createTrackbar("Upper S", "HSV Calibration", 255, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Calibration", 255, 255, nothing)

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of trackbars
    l_h = cv2.getTrackbarPos("Lower H", "HSV Calibration")
    l_s = cv2.getTrackbarPos("Lower S", "HSV Calibration")
    l_v = cv2.getTrackbarPos("Lower V", "HSV Calibration")
    u_h = cv2.getTrackbarPos("Upper H", "HSV Calibration")
    u_s = cv2.getTrackbarPos("Upper S", "HSV Calibration")
    u_v = cv2.getTrackbarPos("Upper V", "HSV Calibration")

    # Define HSV range for green
    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Show mask
    cv2.imshow("Mask", mask)
    cv2.imshow("Original", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
