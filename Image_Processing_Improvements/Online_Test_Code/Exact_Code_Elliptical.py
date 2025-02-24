import matplotlib.pyplot as plt
import time

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import cv2

# Load picture, convert to grayscale and detect edges

image_rgb = data.coffee()[0:220, 160:420]
image_gray = color.rgb2gray(image_rgb)
image_blurred_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

# image_gray_uint8 = img_as_ubyte(image_gray)
# edges = cv2.Canny(image_gray_uint8, threshold1=80, threshold2 = 100)
start_time = time.time()

edges = canny(image_blurred_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

end_time = time.time()
edge_detection_time = end_time - start_time
print(f"Edge detection took {edge_detection_time:.4f} seconds")

conv_start_time = time.time()
edges_uint8 = img_as_ubyte(edges)
conv_end_time = time.time()
conversion_time = conv_end_time - conv_start_time
print(f"Conversion took {conversion_time:.4f} seconds")

im_show_time = time.time()
cv2.imshow('Edges', edges_uint8)
im_show_end_time = time.time()
imshow_time = im_show_end_time - im_show_time
print(f"imshow took {imshow_time:.4f} seconds")
# Perform a Hough Transform

# The accuracy corresponds to the bin size of the histogram for minor axis lengths.
# A higher `accuracy` value will lead to more ellipses being found, at the
# cost of a lower precision on the minor axis length estimation.
# A higher `threshold` will lead to less ellipses being found, filtering out those
# with fewer edge points (as found above by the Canny detector) on their perimeter.
hough_s_time = time.time()
result = hough_ellipse(edges, accuracy=25, threshold=500, min_size=80, max_size=120)
result.sort(order='accumulator')
hough_e_time = time.time()
hough_time = hough_e_time - hough_s_time
print(f"Hough transform took {hough_time:.4f} seconds")


# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = (int(round(x)) for x in best[1:5])
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(
    ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True
)

line_s_time = time.time()
ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)
line_e_time = time.time()
line_time = line_e_time - line_s_time
print(f"Line plotting took {line_time:.4f} seconds")

plt.show()