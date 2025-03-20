import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/hw1.png')

scalingFactor = 1/255.0

# # Convert unsigned int to float
# image = np.float32(image)
# #Scale the values so that they lie between [0,1]
# image = image * scalingFactor

# #Convert back to unsigned int
# image = image * (1.0/scalingFactor)
# image = np.uint8(image)

contrastPercentage = 30

#Clip the values to [0, 255] and change it back to uint8 for display
contrastImage = image * (1 + contrastPercentage/100)
clippedContrastImage = np.clip(contrastImage, 0, 255)
contrastHighClippedUint8 = np.uint8(clippedContrastImage)

#Convert range to [0, 1] and keep it in float format
contrastHighNormalized = (image * (1 + contrastPercentage/100))/255
contrastHighNormalied01Clipped = np.clip(contrastHighNormalized, 0, 1)

plt.figure(figsize=[10,5])
plt.subplot(131);plt.imshow(image[...,::-1]);plt.title("Original Image");
plt.subplot(132);plt.imshow(contrastHighClippedUint8[...,::-1]);plt.title("Converted back to uint8");
plt.subplot(133);plt.imshow(contrastHighNormalied01Clipped[...,::-1]);plt.title("Normalized float to [0,1]");
plt.savefig('contrast_comparison.png', bbox_inches='tight', dpi=300)
plt.show()