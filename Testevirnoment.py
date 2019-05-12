import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import CameraClaibration


dist_img = mpimg.imread('camera_cal/calibration2.jpg')

img_size = (dist_img.shape[1], dist_img.shape[0])
CameraClaibration.storeCameraCalibration()
undist_img = CameraClaibration.CameraCalibrator.undistort(img_size, dist_img)

plt.figure(figsize=(2, 1))

plt.subplot(1, 2, 1)
plt.title('dist')
plt.imshow(dist_img)
#plt.axis("off")

plt.subplot(1, 2, 2)
plt.title('undist')
plt.imsave('output_images/calibration2.jpg', undist_img)
plt.imshow(undist_img)