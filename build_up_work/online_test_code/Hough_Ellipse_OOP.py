import matplotlib.pyplot as plt
import time
import cv2
from skimage import color, data, img_as_ubyte
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.feature import canny


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_rgb = self.load_image()
        self.image_gray = None
        self.image_blurred_gray = None
        self.edges = None
        self.ellipses = []
        self.roi = None
    def load_image(self):
        """Load the image from the specified path."""
        self.image_rgb = cv2.imread(self.image_path)
        #self.image_rgb = data.coffee()[0:220, 160:420]
        if self.image_rgb is None:
            print("Error: Could not load image.")
            return None
        else:
            print("Image loaded successfully.")
            return self.image_rgb
        
    def oldprocess(self):
        
        self.image_gray = color.rgb2gray(self.image_rgb)
        self.edges = canny(self.image_gray, sigma=2.0, low_threshold=0.3, high_threshold=0.8)
        edges_uint8 = img_as_ubyte(self.edges)
        cv2.imshow("Edges", edges_uint8)
        cv2.waitKey(0)

    def preprocess(self):
        """Convert image to grayscale and apply Gaussian blur."""
        self.image_gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_BGR2GRAY)
        self.image_blurred_gray = cv2.GaussianBlur(self.image_gray, (5, 5), 0)  # Reduce noise
        cv2.imshow("Blurred Image", self.image_blurred_gray)
        cv2.waitKey(0)

    def detect_edges(self):
        """Perform Canny edge detection."""
        start_time = time.time()
        self.edges = cv2.Canny(self.image_blurred_gray, 10, 250)
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
            
            
            # Display the cropped area
            cv2.imshow("Cropped Image", self.image_rgb)
            
            cv2.imshow("Cropped Edges", self.edges)
            cv2.waitKey(0)
        else:
            print("No ROI defined. Cannot crop the image.")

    def fit_ellipses(self, min_size=10, max_size=400):
        """Fit ellipses to the edge points using Hough Transform."""
        print("Fitting ellipses to the edge points...")
        hough_s_time = time.time()
        result = hough_ellipse(self.edges, accuracy=25, threshold=500, min_size=min_size, max_size=max_size)
        result.sort(order='accumulator')
        hough_e_time = time.time()
        print(f"Hough transform took {hough_e_time - hough_s_time:.4f} seconds")

        if len(result) == 0:
            print("No ellipse was found")
            return

        # Estimated parameters for the ellipse
        best = list(result[-1])
        yc, xc, a, b = (int(round(x)) for x in best[1:5])
        orientation = best[5]

        # Print the dimensions of the identified ellipse
        print(f"Identified ellipse parameters: Center=({xc}, {yc}), Major axis length={a}, Minor axis length={b}, Orientation={orientation}")

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        self.image_rgb[cy, cx] = (0, 0, 255)
        self.ellipses.append((xc, yc, a, b, orientation))

    def display_results(self):
        """Display the results with detected ellipses."""
        edges_uint8 = self.convert_edges()
        edges_rgb = color.gray2rgb(edges_uint8)
        for xc, yc, a, b, orientation in self.ellipses:
            cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
            edges_rgb[cy, cx] = (250, 0, 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.set_title("Original Image with Detected Ellipses")
        ax1.imshow(cv2.cvtColor(self.image_rgb, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying

        ax2.set_title("Edges and Detected Ellipses")
        ax2.imshow(edges_rgb)

        plt.show()

def main():
    # Load picture and process it
    image_path = r"C:\Users\MaiaRamambason\OneDrive - Imperial College London\Desktop\Year5\UCL_BIP\Object_Detection_Trial\color-detection-opencv\Image_Processing_Improvements\captured_image.jpg"
    processor = ImageProcessor(image_path)
    # processor.preprocess()
    # processor.detect_edges()
    processor.oldprocess()
    # processor.find_contours()
    # processor.find_largest_roi()
    # processor.crop_roi()
    processor.fit_ellipses()
    processor.display_results()

if __name__ == "__main__":
    main()