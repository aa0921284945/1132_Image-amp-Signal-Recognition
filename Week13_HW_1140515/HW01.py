import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

# make fonts readable in matplotlib (in case the notebook default is small)
rcParams["font.size"] = 12

# --- Load the test image -------------------------------------------------
img_bgr = cv2.imread("images/banana.png")
if img_bgr is None:
    raise FileNotFoundError("Could not read images/banana.png")

# --- 1. Convert to HSV & build foreground mask ---------------------------
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
S, V = img_hsv[..., 1], img_hsv[..., 2]

# simple threshold to drop the white background
fg_mask = ((S > 40) & (V > 60)).astype(np.uint8) * 255

# morphological clean‑up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

# --- 2. Extract banana contours & sort left‑to‑right ----------------------
contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# helper: classify by mean hue (OpenCV range 0‑179)
def classify(mean_h):
    if mean_h > 30:
        return "Unripe (Green)"
    elif mean_h > 23:
        return "Ripe (Yellow)"
    elif mean_h > 20:
        return "Overripe (Spotted)"
    else:
        return "Spoiled (Brown)"

results = []
annotated = img_bgr.copy()

for idx, cnt in enumerate(contours, start=1):
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    
    # skip tiny contours, if any
    if w_box * h_box < 500:  
        continue

    mask_single = np.zeros_like(fg_mask)
    cv2.drawContours(mask_single, [cnt], -1, 255, thickness=-1)
    
    H_vals = img_hsv[..., 0][mask_single == 255]
    if H_vals.size == 0:
        continue
    
    mean_h = np.mean(H_vals)
    label  = classify(mean_h)
    results.append((idx, round(mean_h, 2), label))
    
    # draw rectangle
    cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    
    # draw label inside the bounding box (fallback to below box if not enough room)
    label_y = y + 20 if y < 15 else y - 10
    cv2.putText(annotated, label, (x + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

# --- 3. Show annotated image ---------------------------------------------
annot_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 4))
plt.imshow(annot_rgb)
plt.axis("off")
plt.title("Automatic Banana Ripeness Detection")
plt.show()

# --- 4. Show results as a table ------------------------------------------
df = pd.DataFrame(results, columns=["Banana #", "Mean H (OpenCV)", "Ripeness"])
