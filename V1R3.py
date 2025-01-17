import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"size":(1640,1232)},main={"format":'RGB888', "size": (640,480)}))


try:
    picam2.start()
    time.sleep(2)
    img = picam2.capture_array()
    cv2.imwrite("output.jpg", img)
    #cv2.imshow("output",img)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    picam2.stop()
    picam2.close()
