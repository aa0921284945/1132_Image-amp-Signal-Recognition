import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'
image = cv2.imread('images/number_zero.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

# 將除了白色以外的像素區域修改為紅色
lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255])
mask = cv2.inRange(image, lower_white, upper_white)
plt.imshow(mask, cmap='gray')
plt.show()
image[mask == 0] = [255, 0, 0]

# 顯示修改後的圖像
plt.imshow(image)
plt.show()

# 儲存修改後的圖像
cv2.imwrite('test_number_zero.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))