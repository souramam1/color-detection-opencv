import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
from skimage import color, img_as_ubyte
from skimage.measure import EllipseModel, ransac

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_rgb = self.load_image()
        self.image_gray = None
        self.image_blurred_gray = None
        self.edges = None
        self.contours = None
        self.points = None
        self.roi = None
        self.ellipses = []

    def load_image(self):
        """Load the image from the specified path."""
        image = cv2.imread(self.image_path)
        if image is None:
            print("Error: Could not load image.")
            return None
        else:
            print("Image loaded successfully.")
            return image

    def preprocess(self):
        """Convert image to grayscale and apply Gaussian blur."""
        self.image_gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_BGR2GRAY)
        self.image_blurred_gray = cv2.GaussianBlur(self.image_gray, (5, 5), 0)  # Reduce noise
        cv2.imshow("Blurred Image", self.image_blurred_gray)
        cv2.waitKey(0)

    def detect_edges(self):
        """Perform Canny edge detection."""
        start_time = time.time()
        self.edges = cv2.Canny(self.image_blurred_gray, 50, 200)
        end_time = time.time()
        print(f"Edge detection took {end_time - start_time:.4f} seconds")
        cv2.imshow("Edges", self.edges)
        cv2.waitKey(0)

    def convert_edges(self):
        """Convert edges to uint8 for visualization."""
        conv_start_time = time.time()
        edges_uint8 = img_as_ubyte(self.edges)
        conv_end_time = time.time()
        print(f"Conversion took {conv_end_time - conv_start_time:.4f} seconds")
        return edges_uint8

    def extract_edge_points(self):
        """Extract edge points for RANSAC."""
        y, x = np.nonzero(self.edges)  # Get edge coordinates
        self.points = np.column_stack((x, y))
        print(f"Size of points: {self.points.shape}")  # Print number of edge points

        if len(self.points) == 0:
            print("No edge points detected. Adjust the Canny parameters.")
            
    def find_contours(self):
        """Find contours in the edge-detected image."""
        self.contours, _ = cv2.findContours(self.edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    
    def find_largest_roi(self):
        """Find the largest rectangular contour and define it as the region of interest (ROI)."""
        largest_area = 0
        
        for contour in self.contours:
            _, size, _ = cv2.minAreaRect(contour)
            area = size[0] * size[1]
            if area > 90000 and area > largest_area:
                largest_area = area
                x, y, w, h = cv2.boundingRect(contour)
                self.roi = (x, y, w, h)
        

    def crop_roi(self):
        """Crop the image to the region of interest (ROI) and display it."""
        if self.roi is not None:
            x, y, w, h = self.roi
            constant = 20
            self.image_rgb = self.image_rgb[y+constant:y+h-constant, x+constant:x+w-constant]
            self.image_gray = self.image_gray[y+constant:y+h-constant, x+constant:x+w-constant]
            self.image_blurred_gray = self.image_blurred_gray[y+constant:y+h-constant, x+constant:x+w-constant]
            self.edges = self.edges[y+constant:y+h-constant, x+constant:x+w-constant]
            self.extract_edge_points()  # Re-extract edge points within the cropped ROI
            
            # Display the cropped area
            cv2.imshow("Cropped Image", self.image_rgb)
            cv2.waitKey(0)
        else:
            print("No ROI defined. Cannot crop the image.")
        

    def fit_ellipses(self, max_ellipses=2):
        """Fit ellipses to the edge points using RANSAC."""
        for _ in range(max_ellipses):
            # RANSAC Ellipse Fitting
            ransac_s_time = time.time()
            ransac_model, inliers = ransac(self.points, EllipseModel, min_samples=5, residual_threshold=2, max_trials=100)
            ransac_e_time = time.time()
            print(f"RANSAC took {ransac_e_time - ransac_s_time:.4f} seconds")

            # If an ellipse was detected, store it and remove inliers
            if ransac_model:
                self.ellipses.append(ransac_model.params)
                self.points = self.points[~inliers]  # Remove inliers from points
            else:
                break


    def draw_roi(self):
        """Draw contours within the ROI and highlight nested contours."""
        roi_x, roi_y, roi_w, roi_h = self.roi
        cv2.rectangle(self.image_rgb, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
        
    def draw_ellipses(self, edges_uint8):
        """Draw detected ellipses on the original image and edge image."""
        for params in self.ellipses:
            xc, yc, a, b, theta = params
            print(f"Ellipse found: Center=({xc}, {yc}), Axes=({a}, {b}), Angle={theta}")
            cv2.ellipse(self.image_rgb, (int(xc), int(yc)), (int(a), int(b)), np.degrees(theta), 0, 360, (0, 0, 255), 2)
            cv2.ellipse(edges_uint8, (int(xc), int(yc)), (int(a), int(b)), np.degrees(theta), 0, 360, (255, 0, 0), 2)

    def display_results(self, edges_uint8):
        """Display the results with detected ellipses."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.set_title("Original Image with Detected Ellipses")
        ax1.imshow(cv2.cvtColor(self.image_rgb, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying

        ax2.set_title("Edges and Detected Ellipses")
        ax2.imshow(edges_uint8, cmap='gray')

        plt.show()

def main():
    # Load picture and process it
    image_path = r"C:\Users\MaiaRamambason\OneDrive - Imperial College London\Desktop\Year5\UCL_BIP\Object_Detection_Trial\color-detection-opencv\Image_Processing_Improvements\captured_image.jpg"
    processor = ImageProcessor(image_path)
    processor.preprocess()
    processor.detect_edges()
    processor.find_contours()
    processor.find_largest_roi()
    processor.crop_roi()
    edges_uint8 = processor.convert_edges()
    processor.extract_edge_points()
    processor.fit_ellipses()
    processor.draw_roi()
    processor.draw_ellipses(edges_uint8)
    processor.display_results(edges_uint8)

if __name__ == "__main__":
    main()
