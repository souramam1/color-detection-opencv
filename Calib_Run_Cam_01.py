from HSV_Calib_01 import HSVCalibrator
from Picam_07 import ColorDetectionWithROI

# This will run a calibration FIRST and then directly input those values into the colour detection file.
# In PICAM_07, lines 14 to 20 contain a generic calibration: line 21 defines the class --> here a different calibration can be passed in
# Line 28 uses the value of final_calib to continue with object detection.
# TESTED.

class CalibRun:
    
    def __init__(self):
        self.hsv_calib = HSVCalibrator()
        self.hsv_calib.run()
        
        self.color_detector = ColorDetectionWithROI(final_hsv_calib=self.hsv_calib.calibrated_values, smoothing_window_size=10)
        self.color_detector.run()
        
    
if __name__ == "__main__":
    calibrate_and_run = CalibRun()