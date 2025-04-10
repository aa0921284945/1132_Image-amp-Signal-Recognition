import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- 1. Read Image & Preprocessing ----------
image = cv2.imread('images/coins.png')
b, g, r = cv2.split(image)
gray = b 

# Threshold on blue channel
_, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded Image")
plt.axis('off')
plt.savefig('thresholded_image.png')
plt.show()

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
plt.imshow(thresh_cleaned, cmap='gray')
plt.title("Morphological Operations")
plt.axis('off')
plt.savefig('morphological_operations.png')
plt.show()

# ---------- 2. Blob Detection ----------
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 255

params.filterByArea = True
params.minArea = 9000
params.maxArea = 100000

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByConvexity = True
params.minConvexity = 0.8

params.filterByInertia = True
params.minInertiaRatio = 0.3

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(thresh_cleaned)

output = image.copy()
for kp in keypoints:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    r = int(kp.size / 2)
    cv2.circle(output, (x, y), r, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Coins with Blobs")
plt.axis('off')
plt.savefig('blob_detection.png')
plt.show()

# ---------- 3. Connected Component Analysis ----------
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_cleaned)

areas = stats[1:, cv2.CC_STAT_AREA]
sorted_indices = np.argsort(areas)[::-1]
top10_indices = sorted_indices[:10] + 1

cols = 5
rows = 2
plt.figure(figsize=(15, 6))

for i, idx in enumerate(top10_indices):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(labels == idx, cmap='gray')
    plt.title(f"Component ID: {idx}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('connected_components_analysis.png')
plt.show()

# ---------- 4. Contour Detection & Circle Fitting ----------
image_contour = image.copy()
contours, hierarchy = cv2.findContours(thresh_cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image_contour, contours, -1, (0, 255, 0), 3)

for index, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if area < 1000:
        continue

    M = cv2.moments(cnt)
    if M['m00'] == 0:
        continue

    cx = int(round(M['m10'] / M['m00']))
    cy = int(round(M['m01'] / M['m00']))
    
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image_contour, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv2.circle(image_contour, (cx, cy), 10, (255, 0, 0), -1)
    cv2.putText(image_contour, f"{index+1}", (x + 40, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print(f"Contour #{index+1} has area = {area:.2f} and perimeter = {perimeter:.2f}")

plt.imshow(cv2.cvtColor(image_contour, cv2.COLOR_BGR2RGB))
plt.title("Contour Detection + Fitted Circles")
plt.axis('off')
plt.savefig('contour_fitting.png')
plt.show()
