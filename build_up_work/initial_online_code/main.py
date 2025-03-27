import cv2             #open cv lib
from PIL import Image  # Pillow library image class for processing of image boxes here

from util import get_limits # import the get limits function from util file - this allows for HSV thresholds to be calculated dynamically


yellow = [0, 255, 255]  # yellow in BGR 
red = [0,0,255] # red in BGR
blue = [255,0,0] # blue in BGR
cap = cv2.VideoCapture(0) #the index of the webcam you want to connect to (if attaching an overhead will be 1,2 etc.)
while True:
    ret, frame = cap.read()  # ret is a Boolean that returns True or False (i.e false would be no frames are captured) , frame is the actual frame

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #converting from BGR colourspace to HSV (hue, saturation, value) - better suited to object identification, more similar to how we see objects.

    lowerLimit, upperLimit = get_limits(color=blue) #

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

