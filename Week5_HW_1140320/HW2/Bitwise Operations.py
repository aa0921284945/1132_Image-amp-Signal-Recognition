import cv2
import numpy as np
import matplotlib.pyplot as plt

tigerImage = cv2.imread('images/hw2_1.png')
plt.imshow(tigerImage[...,::-1])
plt.show()

#Make a copy
tigerWithHatBitwise = tigerImage.copy()

#Load the sunglasses image with Alpha channel
hatImagePath = 'images/hw2_2.png'
hatPNG = cv2.imread(hatImagePath,-1)
print("Original Hat Dimension = {}".format(hatPNG.shape))

#Resize the image to fit over the head region
hatPNG = cv2.resize(hatPNG,(170,90))
print("Image Dimension = {}".format(hatPNG.shape))

# Separate the Color and alpha channels
hatRGB = hatPNG[:,:,0:3]
hatMask1 = hatPNG[:,:,3]

# Display the images for clarity
plt.figure(figsize=[10,5])
plt.subplot(121);plt.imshow(hatRGB[...,::-1]);plt.title("Color channels");
plt.subplot(122);plt.imshow(hatMask1,cmap='gray');plt.title("Alpha channel");
plt.show()

# Get the head region from the face image
headROI= tigerWithHatBitwise[0:90,37:207]

#Make the dimensions of the mask same as the input image.
#Since Face Image is a 3 channel image, we create a 3 channel image for the mask
hatMask = cv2.merge((hatMask1,hatMask1,hatMask1))

# Use the mask to create the masked head region
head = cv2.bitwise_and(headROI,cv2.bitwise_not(hatMask))

# Use the mask to create the masked sunglass region
sunglass = cv2.bitwise_and(hatRGB,hatMask)

# Combine the Sunglass in the head Region to get the augmented image
headROIFinal = cv2.add(head,sunglass)

# Display the intermediate results
plt.figure(figsize=[10,5])
plt.subplot(131);plt.imshow(head[...,::-1]);plt.title("Masked head Region");
plt.subplot(132);plt.imshow(sunglass[...,::-1]);plt.title("Masked Hat");
plt.subplot(133);plt.imshow(headROIFinal[...,::-1]);plt.title("Combined head Region");
plt.savefig('hat_on_tiger.png', bbox_inches='tight', dpi=300)
plt.show()

# Replace the head ROI with the output from the previous section
tigerWithHatBitwise[0:90,37:207] = headROIFinal

# Display the final result
plt.figure(figsize=[7,7])
plt.subplot(121);plt.imshow(tigerImage[...,::-1]);plt.title("Original Image");
plt.subplot(122);plt.imshow(tigerWithHatBitwise[...,::-1]);plt.title("With Hat");
plt.savefig('tiger_with_hat.png', bbox_inches='tight', dpi=300)
plt.show()