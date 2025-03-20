import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 設定 matplotlib 的圖像大小和色彩映射
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# 讀取圖像並轉換為 RGB 格式
image = cv2.imread('images/hw3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 顯示原始圖像
plt.imshow(image)
plt.axis('off')
plt.show()

# 設定紅色範圍
lower_red = np.array([150, 0, 0])
upper_red = np.array([255, 100, 100])
mask_red = cv2.inRange(image, lower_red, upper_red)

# 設定紅色2範圍
lower_red2 = np.array([249, 0, 0])
upper_red2 = np.array([255, 100, 100])
mask_red2 = cv2.inRange(image, lower_red2, upper_red2)

# 設定紅色3範圍
lower_red3 = np.array([233, 0, 0])
upper_red3 = np.array([255, 100, 100])
mask_red3 = cv2.inRange(image, lower_red3, upper_red3)

# 設定藍色範圍
lower_blue = np.array([0, 0, 150])
upper_blue = np.array([100, 100, 255])
mask_blue = cv2.inRange(image, lower_blue, upper_blue)

# 設定藍色2範圍
lower_blue2 = np.array([0, 0, 240])
upper_blue2 = np.array([100, 100, 255])
mask_blue2 = cv2.inRange(image, lower_blue2, upper_blue2)

# 設定藍色3範圍
lower_blue3 = np.array([0, 0, 220])
upper_blue3 = np.array([100, 100, 255])
mask_blue3 = cv2.inRange(image, lower_blue3, upper_blue3)

# 顯示紅色和藍色遮罩
fig, axs = plt.subplots(2, 3, figsize=(12, 12))
axs[0, 0].imshow(mask_red, cmap='gray')
axs[0, 0].axis('off')
axs[0, 0].set_title('Red Mask')

axs[0, 1].imshow(mask_red2, cmap='gray')
axs[0, 1].axis('off')
axs[0, 1].set_title('Red Mask 2')

axs[0, 2].imshow(mask_red3, cmap='gray')
axs[0, 2].axis('off')
axs[0, 2].set_title('Red Mask 3')

axs[1, 0].imshow(mask_blue, cmap='gray')
axs[1, 0].axis('off')
axs[1, 0].set_title('Blue Mask')

axs[1, 1].imshow(mask_blue2, cmap='gray')
axs[1, 1].axis('off')
axs[1, 1].set_title('Blue Mask 2')

axs[1, 2].imshow(mask_blue3, cmap='gray')
axs[1, 2].axis('off')
axs[1, 2].set_title('Blue Mask 3')

plt.show()

# 將遮罩擴展到與圖像相同的形狀
mask_red = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2RGB)
mask_red2 = cv2.cvtColor(mask_red2, cv2.COLOR_GRAY2RGB)
mask_red3 = cv2.cvtColor(mask_red3, cv2.COLOR_GRAY2RGB)
mask_blue = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2RGB)
mask_blue2 = cv2.cvtColor(mask_blue2, cv2.COLOR_GRAY2RGB)
mask_blue3 = cv2.cvtColor(mask_blue3, cv2.COLOR_GRAY2RGB)

# 使用遮罩修改圖像
imageRed = np.where(mask_red == 0, [255, 255, 255], image)
imageRed2 = np.where(mask_red2 == 0, [255, 255, 255], image)
imageRed3 = np.where(mask_red3 == 0, [255, 255, 255], image)

imageBlue = np.where(mask_blue == 0, [255, 255, 255], image)
imageBlue2 = np.where(mask_blue2 == 0, [255, 255, 255], image)
imageBlue3 = np.where(mask_blue3 == 0, [255, 255, 255], image)

# 合併修改後的圖像
combined_image_top = np.hstack((imageRed, imageRed2, imageRed3))
combined_image_bottom = np.hstack((imageBlue, imageBlue2, imageBlue3))
combined_image = np.vstack((combined_image_top, combined_image_bottom))

# 合併遮罩
combined_mask_top = np.hstack((mask_red, mask_red2, mask_red3))
combined_mask_bottom = np.hstack((mask_blue, mask_blue2, mask_blue3))
combined_mask = np.vstack((combined_mask_top, combined_mask_bottom))

# 確保合併後的圖像是 uint8 型態
combined_image = combined_image.astype(np.uint8)
combined_mask = combined_mask.astype(np.uint8)

# 儲存合併後的圖像和遮罩
cv2.imwrite('group_image.jpg', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('group_mask.jpg', combined_mask)

# 顯示修改後的圖像
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs[0, 0].imshow(imageRed)
axs[0, 0].axis('off')
axs[0, 0].set_title('Image with Red Mask')

axs[0, 1].imshow(imageRed2)
axs[0, 1].axis('off')
axs[0, 1].set_title('Image with Red Mask 2')

axs[0, 2].imshow(imageRed3)
axs[0, 2].axis('off')
axs[0, 2].set_title('Image with Red Mask 3')

axs[1, 0].imshow(imageBlue)
axs[1, 0].axis('off')
axs[1, 0].set_title('Image with Blue Mask')

axs[1, 1].imshow(imageBlue2)
axs[1, 1].axis('off')
axs[1, 1].set_title('Image with Blue Mask 2')

axs[1, 2].imshow(imageBlue3)
axs[1, 2].axis('off')
axs[1, 2].set_title('Image with Blue Mask 3')

plt.show()