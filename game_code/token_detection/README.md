# Color Detection and Contour Detection Project

This project implements a real-time object detection and classification system using color detection and contour detection techniques. It utilizes OpenCV for image processing and NumPy for numerical operations.

## Project Structure

```
color-detection-opencv
├── src
│   ├── color_detection.py      # Contains the ColorDetection class for color classification
│   ├── contour_detection.py     # Contains the ContourDetection class for contour detection
│   ├── main.py                  # Integrates ColorDetection and ContourDetection classes
├── requirements.txt             # Lists the project dependencies
└── README.md                    # Project documentation
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd color-detection-opencv
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the object detection and classification pipeline, execute the following command:

```
python src/main.py
```

Ensure that your webcam is connected and accessible. The program will open a window displaying the video feed with detected contours and classified colors.

## Dependencies

This project requires the following Python packages:

- OpenCV
- NumPy

You can install these packages using pip, as mentioned in the installation section.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.