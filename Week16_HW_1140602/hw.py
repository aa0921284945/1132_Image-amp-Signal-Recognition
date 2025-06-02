import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('images/coin.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (9, 9), 0)

# Use Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the original image
output = img.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Count the coins based on contours
coin_count = len(contours)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blur, cmap='gray')
plt.title('Gaussian Blurred')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Coins: {coin_count}')
plt.axis('off')

plt.tight_layout()
plt.show()
