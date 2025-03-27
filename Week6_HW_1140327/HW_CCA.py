import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 讀入灰階圖
image = cv2.imread('images/carlicense.png', cv2.IMREAD_GRAYSCALE)

# 顯示原始圖
plt.figure(figsize=(6, 4))
plt.title("Original Grayscale Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# 二值化處理
_, imThresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
imThresh_inv = cv2.bitwise_not(imThresh)
plt.figure()
plt.title("Inverted Threshold Image")
plt.imshow(imThresh_inv, cmap='gray')
plt.axis('off')
plt.show()

# 找到連通元件
num_labels, imLabels = cv2.connectedComponents(imThresh_inv)

# 根據 num_labels 依序顯示各個連通元件
# 設定每行顯示3個元件，根據總元件數決定列數
num_labels = num_labels - 2
cols = 3
rows = math.ceil(num_labels / cols)
plt.figure(figsize=(15, 5 * rows))

for i in range(num_labels):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(imLabels == i, cmap='gray')
    plt.title("Component ID: {}".format(i))
    plt.axis('off')

plt.tight_layout()
plt.savefig('CCA.png')
plt.show()
