import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread('images/butterfly.jpg', cv2.IMREAD_GRAYSCALE)

# Resize the image to 256x256
img_resized = cv2.resize(img, (256, 256))

# Apply Gaussian smoothing
img_blurred = cv2.GaussianBlur(img_resized, (5, 5), sigmaX=1.4)

# Compute gradients using Sobel operator
Ix = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=3)

# Compute gradient magnitude
grad_img = np.hypot(Ix, Iy)
grad_img = (grad_img / grad_img.max()) * 255  # Normalize to 0-255
grad_img = grad_img.astype(np.uint8)

# Save gradient images
cv2.imwrite('Ix.jpg', np.abs(Ix).astype(np.uint8))
cv2.imwrite('Iy.jpg', np.abs(Iy).astype(np.uint8))
cv2.imwrite('grad_img.jpg', grad_img)

# Calculate gradient direction in degrees
theta = np.rad2deg(np.arctan2(Iy, Ix)) % 180  # Angle between 0 and 180

# Quantize angles into four directions
q_theta = np.zeros(theta.shape, dtype=np.uint8)
q_theta[(theta >= 0) & (theta < 22.5)] = 0
q_theta[(theta >= 157.5) & (theta <= 180)] = 0
q_theta[(theta >= 22.5) & (theta < 67.5)] = 1
q_theta[(theta >= 67.5) & (theta < 112.5)] = 2
q_theta[(theta >= 112.5) & (theta < 157.5)] = 3

# Perform Non-Maxima Suppression
M, N = grad_img.shape
non_max_supp = np.zeros((M, N), dtype=np.uint8)

for i in range(1, M - 1):
    for j in range(1, N - 1):
        angle = q_theta[i, j]
        mag = grad_img[i, j]
        
        if angle == 0:
            neighbors = [grad_img[i, j + 1], grad_img[i, j - 1]]
        elif angle == 1:
            neighbors = [grad_img[i - 1, j + 1], grad_img[i + 1, j - 1]]
        elif angle == 2:
            neighbors = [grad_img[i - 1, j], grad_img[i + 1, j]]
        elif angle == 3:
            neighbors = [grad_img[i - 1, j - 1], grad_img[i + 1, j + 1]]
        
        if mag >= max(neighbors):
            non_max_supp[i, j] = mag
        else:
            non_max_supp[i, j] = 0

# Save the result
cv2.imwrite('non_maxima_supp.jpg', non_max_supp)

# Define thresholds
T_1 = non_max_supp.max() * 0.1  # Lower threshold
T_2 = non_max_supp.max() * 0.3  # Higher threshold

# Double thresholding
strong_edges = (non_max_supp >= T_2)
weak_edges = ((non_max_supp >= T_1) & (non_max_supp < T_2))
edges = np.zeros(non_max_supp.shape, dtype=np.uint8)
edges[strong_edges] = 255

# Edge linking
M, N = non_max_supp.shape
for i in range(1, M - 1):
    for j in range(1, N - 1):
        if weak_edges[i, j]:
            angle = q_theta[i, j]
            if angle == 0:
                neighbors = [edges[i, j + 1], edges[i, j - 1]]
            elif angle == 1:
                neighbors = [edges[i - 1, j + 1], edges[i + 1, j - 1]]
            elif angle == 2:
                neighbors = [edges[i - 1, j], edges[i + 1, j]]
            elif angle == 3:
                neighbors = [edges[i - 1, j - 1], edges[i + 1, j + 1]]
            
            if any(n == 255 for n in neighbors):
                edges[i, j] = 255

# Save the final edge image
cv2.imwrite('butterfly_edges.png', edges)
