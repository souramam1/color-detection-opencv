from colour_detection import ColorDetectionWithROI
from data_writer import DataWriter

def main():
    # Initialize color detection with ROI
    colour_detection = ColorDetectionWithROI(smoothing_window_size=5)
    
    # Initialize data writer and start timer for saving color counts
    data_writer = DataWriter()
    data_writer.start_timer(colour_detection)

    # Run color detection
    colour_detection.run()

if __name__ == "__main__":
    main()
