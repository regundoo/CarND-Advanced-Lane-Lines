import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
import pickle

#prepare opject points
nx = 6
ny = 9

CAL_IMAGE_SIZE = (720, 1280, 3)
CALC_CAL_POINTS = True
CalPathImages = 'camera_cal/calibration*.jpg'
CALIBRATION_PATH = 'camera_cal/calibration.p'


def calulateCalibration(CalPathImages, nx, ny):
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)

    images = glob.glob(CalPathImages)
    print(images)

    for fname in images:
        img = mpimg.imread(fname)

        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the Chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print(ret)
        print(corners)

        # if found, draw corners
        if ret == True:
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            imgpoints.append(corners)
            objpoints.append(objp)
            #plt.imshow(img)
            #plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, CAL_IMAGE_SIZE[:-1], None, None)
    calibration = {'objpoints': objpoints,
                       'imgpoints': imgpoints,
                       'cal_images': objp,
                       'mtx': mtx,
                       'dist': dist,
                       'rvecs': rvecs,
                       'tvecs': tvecs}

    return calibration


def storeCameraCalibration():

    if CALC_CAL_POINTS:
        calibration = calulateCalibration(CalPathImages, nx, ny)
        with open(CALIBRATION_PATH, 'wb') as f:
            pickle.dump(calibration, file=f)
    else:
        with open(CALIBRATION_PATH, "rb") as f:
            calibration = pickle.load(f)

    return calibration

class CameraCalibrator:
    def __init__(self, image_size, calibration):

        self.objpoints = calibration['objpoints']
        self.imgpoints = calibration['imgpoints']
        self.image_size = image_size

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, image_size, None, None)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
