import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/hw1.png')

brightnessOffset = 60
#Add the offset for increasing brightness
brightImageOpenCV = cv2.add(image,np.ones(image.shape,dtype='uint8')*brightnessOffset)

brightHighInt32 = np.int32(image) + brightnessOffset
brightHighInt32Clipped  = np.clip(brightHighInt32,0,255)

plt.figure(figsize=[10,5])
plt.subplot(131);plt.imshow(image[...,::-1]);plt.title("Original Image")
plt.subplot(132);plt.imshow(brightImageOpenCV[...,::-1]);plt.title("Using cv2.add function")
plt.subplot(133);plt.imshow(brightHighInt32Clipped[...,::-1]);plt.title("Using numpy and clipping")
plt.savefig('brightness_comparison.png', bbox_inches='tight', dpi=300)
plt.show()

#Add the offset for increasing brightness
brightHighFloat32 = np.float32(image) + brightnessOffset
brightHighFloat32NormalizedClipped  = np.clip(brightHighFloat32/255,0,1)

brightHighFloat32ClippedUint8 = np.uint8(brightHighFloat32NormalizedClipped*255)

#Display the images
plt.figure(figsize=[10,5])
plt.subplot(131);plt.imshow(image[...,::-1]);plt.title("Original Image")
plt.subplot(132);plt.imshow(brightHighFloat32NormalizedClipped[...,::-1]);plt.title("Using np.float32 and clipping")
plt.subplot(133);plt.imshow(brightHighFloat32ClippedUint8[...,::-1]);plt.title("Using int->float->int and clipping")
plt.savefig('brightness_comparison_float.png', bbox_inches='tight', dpi=300)
plt.show()
