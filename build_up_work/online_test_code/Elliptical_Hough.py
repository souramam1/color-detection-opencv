import matplotlib.pyplot as plt

import time
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# Load picture, convert to grayscale and detect edges
image_rgb = data.coffee()[0:220, 160:420]
image_gray = color.rgb2gray(image_rgb)
edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

# Perform a Hough Transform
# The accuracy corresponds to the bin size of the histogram for minor axis lengths.
# A higher `accuracy` value will lead to more ellipses being found, at the
# cost of a lower precision on the minor axis length estimation.
# A higher `threshold` will lead to less ellipses being found, filtering out those
# with fewer edge points (as found above by the Canny detector) on their perimeter.
hough_s_time = time.time()
result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
hough_e_time =  time.time()
print(f"Hough transform took {hough_e_time - hough_s_time:.4f} seconds")
result.sort(order='accumulator')

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

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()