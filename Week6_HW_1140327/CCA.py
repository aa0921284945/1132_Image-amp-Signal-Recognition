import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/truth.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.show()

#Threshold Image
th, imThresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
#Find connected components
_, imLabels = cv2.connectedComponents(imThresh)
plt.imshow(imLabels, cmap='gray')
plt.colorbar()
plt.show()
#Display the Labels
nComponents = imLabels.max()
displayRows = np.ceil(nComponents/3.0)
plt.figure(figsize=[8,5])
#print the truth worlds
plt.subplot(2,3,1)
plt.imshow(imLabels==0, cmap='gray')
plt.title("Background, Component ID : {}".format(0))
#print the Letter T
plt.subplot(2,3,2)
plt.imshow(imLabels==1, cmap='gray')
plt.title("Component ID : {}".format(1))
#print the Letter R
plt.subplot(2,3,3)
plt.imshow(imLabels==2, cmap='gray')
plt.title("Component ID : {}".format(2))
#print the Letter U
plt.subplot(2,3,4)
plt.imshow(imLabels==3, cmap='gray')
plt.title("Component ID : {}".format(3))
#print the Letter T
plt.subplot(2,3,5)
plt.imshow(imLabels==4, cmap='gray')
plt.title("Component ID : {}".format(4))
#print the Letter H
plt.subplot(2,3,6)
plt.imshow(imLabels==5, cmap='gray')
plt.title("Component ID : {}".format(5))
plt.show()