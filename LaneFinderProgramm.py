import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import cv2
import os
import numpy as np
import math

class LaneFinder:
    def __init__(self, mtx, dist, M, warp_size, unwarp_size, img_size):
        self.mtx = mtx
        self.dist = dist
        self.M = M
        self.warp_size = warp_size
        self.unwarp_size = unwarp_size
        self.img_size = img_size
        self.mask = np.zeros((warp_size[1], warp_size[0], 3), dtype=np.uint8)
        self.roi_mask = np.ones((warp_size[1], warp_size[0], 3), dtype=np.uint8)


    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist)

    def warp(self, img):
        return cv2.warpPerspective(img, self.M, self.unwarp_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)

    def unwarp(self):
        pass

    def equalize_lines(self):
        pass

    def find_lanes(self, img):
        img = self.undistort(img)

        img = self.warp(img)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls = cv2.medianBlur(hls, 5)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

        plt.imshow(img)
        plt.show()
        plt.imshow(hls)
        plt.show()
        plt.imshow(lab)
        plt.show()

        # This gives you the elliptical kernal back.
        big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        # This gives you the rectengular gernal
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        greenery = (lab[:, :, 2].astype(np.uint8) > 145) & cv2.inRange(hls, (0, 0, 50), (35, 190, 255))

        plt.imshow(greenery, cmap = 'gray')
        plt.show()

        mask_road = np.logical_not(greenery).astype(np.uint8) & (hls[:, :, 1] < 250)

        # This function is used to remove the noise of the iage
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, small_kernel)
        plt.imshow(mask_road, cmap = 'gray')
        plt.show()

        windowsimage = self.sliding_window_search(mask_road)
        # This combines the mask_road with the kernel
        plt.imshow(windowsimage)
        plt.show()

        return windowsimage

    def sliding_window_search(self, img):
        out_img = np.dstack((img, img, img)) * 255
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        print(histogram)

    def process_image(self, img):
        pass






def loadCalibrationData():
    dist_pickel = pickle.load(open("./camera_cal/calibration_pickle.p", "rb"))
    mtx = dist_pickel["mtx"]
    dist = dist_pickel["dist"]

    return mtx, dist

def calc_warp_points():
    """
    :return: Source and Destination pointts
    """
    src = np.float32 ([
        [375, 480],
        [905, 480],
        [1811, 685],
        [-531, 685]
    ])

    dst = np.float32 ([
            [0, 0],
            [500, 0],
            [500, 600],
            [0, 600]
        ])
    return src, dst



mtx, dist = loadCalibrationData()
src, dst = calc_warp_points()
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)


output_dir = "output_images"


img = mpimg.imread("test_images/test4.jpg")
# plt.imshow(img)
# plt.show()

img_size = (img.shape[1], img.shape[0])
warp_size = (1280, 720)
unwarp_size = (500, 600)

lf = LaneFinder(mtx, dist, M, warp_size, unwarp_size, img_size)
processedImage = lf.find_lanes((img))
print(lf)
