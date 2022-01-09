from typing import List
import dclab
import matplotlib.pyplot as plt
import cv2
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import imutils


class WhiteNeighborCorrector:
    def __init__(self) -> None:
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def erode(self, img: np.array, iterations: int = 1):
        img = cv2.erode(img, self.kernel, iterations=iterations)
        return img

    def dilate(self, img: np.array, iterations: int = 1):
        img = cv2.dilate(img, self.kernel, iterations=iterations)
        return img

    def closing(self, img: np.array):
        img = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, (1, 1))
        return img

    def run(self, img):
        erode = self.erode(img)
        dilate = self.dilate(erode)
        close = self.closing(dilate)
        return close


class Contour:
    def __init__(self) -> None:
        pass

    def get_threshold(self, img):
        (T, threshInv) = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        return T, threshInv

    def create_mask(self, img, threshInv):
        masked = cv2.bitwise_and(img, img, mask=threshInv)

        return masked

    def detect_edge(self, img, T):
        edged = cv2.Canny(img, T, T * 2)
        return edged

    def __call__(self, img):

        thresh, threshInv = self.get_threshold(img)
        masked_img = self.create_mask(img, threshInv)
        edged_img = self.detect_edge(masked_img, thresh)

        cnts = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        closed_contours = []
        for i in cnts:
            if cv2.contourArea(i) > cv2.arcLength(i, closed=True):
                closed_contours.append(i)
            else:
                pass  # Open

        d = cv2.drawContours(img.astype(np.uint8), closed_contours, -1, (0, 255, 0), 1)
        return d


class ContourDetector:
    def __init__(self) -> None:
        self.contour = Contour()

    def read_data(self, data_path):
        ds = dclab.new_dataset(data_path)
        return ds["image"]

    def run(self, dataset):

        contours = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exec:
            for result in exec.map(self.contour, dataset):
                contours.append(result)

        return contours


if __name__ == "__main__":

    ############### TASK 1 ###############

    detector = ContourDetector()
    dataset = detector.read_data(data_path="../Data001.rtdc")
    dataset = dataset[:10]
    contours = detector.run(dataset)
    ml_data = zip(dataset, contours)
    for data, contour in ml_data:
        plt.imshow(data)
        plt.show()
        plt.imshow(contour)
        plt.show()

    ############### TASK 2 ###############

    wnc = WhiteNeighborCorrector()
    wncs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exec:
        for result in exec.map(wnc.run, dataset):
            wncs.append(result)

    for d, c in zip(dataset, wncs):
        plt.imshow(d, cmap="gray")
        plt.show()
        plt.imshow(c, cmap="gray")
        plt.show()
