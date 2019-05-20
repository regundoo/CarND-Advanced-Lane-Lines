import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class PerspectiveTransform:
    def __init__(self):
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)


    @property
    def src(self):
        return np.array([(190, 720), (589, 457), (698, 457), (1145, 720)], np.float32)

    @property
    def dst(self):
        offset_left = 300
        offset_right = 300
        img_shape = (720, 1280)
        img_mid = img_shape[1] / 2
        x1 = img_mid - offset_left
        x2 = img_mid + offset_right
        return np.array([(x1, img_shape[0]), (x2, img_shape[0]), (x2, 0), (x1, 0)], np.float32)

    @property
    def M(self):
        return self.M

    @property
    def M_inv(self):
        return self.M_inv

    def warp(self, img, newdims = None):
        if (newdims is None):
            newdims = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(image, self.M, newdims, flags=cv2.INTER_LINEAR)

    def unwarp(self, img, newdims=None):
        if newdims is None:
            newdims = img.shape[1], img.shape[0]
        return cv2.warpPerspective(img, self.Minv, newdims, flags=cv2.INTER_LINEAR)
