# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:37:08 2019

@author: Y1PGLOCK
"""
#import pickle
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
# prepare object points
nx = 9
ny = 6

images_path = 'camera_cal'
os.listdir('camera_cal')

objpoints = [] #3D Points in real world space
imgpoints = [] # 2D points in image plane

#Prepare Object points 
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)



for image_file in os.listdir(images_path):
    if image_file.endswith("jpg"):
        img = mpimg.imread(os.path.join(images_path, image_file))
        #plt.imshow(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret == True:
            #Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            imgpoints.append(corners)
            objpoints.append(objp)
            plt.imshow(img)
            plt.show()
           # undist = cv2.undistort(img, mx, dist, None, mtx)

#print(imgpoints)
# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#print(mtx)

for image_file in os.listdir(images_path):
    if image_file.endswith("jpg"):
        # show distorted images
        img = mpimg.imread(os.path.join(images_path, image_file))
        plt.imshow(cv2.undistort(img, mtx, dist))
        plt.show()