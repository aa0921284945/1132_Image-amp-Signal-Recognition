import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('images/hw02.png')

# 調整亮度與對比
alpha = 1.2  # 對比
beta = 30    # 亮度
im_bright = cv2.convertScaleAbs(im, alpha=alpha, beta=beta)

# 顯示原圖與調整後圖像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(im[:, :, ::-1])
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(im_bright[:, :, ::-1])
plt.title(f'Brightened (α={alpha}, β={beta})')
plt.axis('off')
plt.show()
