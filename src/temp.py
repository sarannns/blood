from typing import List
import dclab
import matplotlib.pyplot as plt
import cv2
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import imutils

ds = dclab.new_dataset("../Data001.rtdc")
for i in range(10):
    img = ds["image"][i]

    (T, threshInv) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Threshold masked regions in the image
    masked = cv2.bitwise_and(img, img, mask=threshInv)

    # Detect edges using Canny
    edged = cv2.Canny(masked, T, T * 2)

    # Detect contours in edged threshold masked image
    cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    closed_contours = []
    open_contours = []
    for i in cnts:
        if cv2.contourArea(i) > cv2.arcLength(i, closed=True):
            closed_contours.append(i)
        else:
            open_contours.append(i)

    d = cv2.drawContours(img.astype(np.uint8), closed_contours, -1, (0, 255, 0), 1)

    plt.imshow(d, cmap="gray")
    plt.show()
