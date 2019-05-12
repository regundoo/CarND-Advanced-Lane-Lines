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
CALC_CAL_POINTS = False
CalPathImages = 'camera_cal/calibration*.jpg'
CALIBRATION_PATH = 'camera_cal/calibration.p'

# Make a list of calibration images
#imagelist = os.listdir("camera_cal/")
#dirname = 'camera_cal/'
#print(imagelist)

#img = mpimg.imread(os.path.join(dirname, imagelist[12]))
#img = cv2.imread(fname)


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
    calibration = calulateCalibration(CalPathImages, nx, ny)
    with open(CALIBRATION_PATH, 'wb') as f:
        pickle.dump(calibration, file=f)

    return calibration

