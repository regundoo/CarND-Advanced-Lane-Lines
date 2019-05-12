import pickle
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

Rows = 6
Cols = 9
image_size = (720, 1280, 3)
cal_points = False
imagePath = '/camera_cal/calibration*.jpg'
calOutput = '/camera_cal/calMatrix.p'

def calculateCalibration(pattern, rows, cols):


    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob(pattern)
    cal_images = np.zeros((len(images), *CAL_IMAGE_SIZE), dtype=np.uint8)

    successfull_cnt = 0

    for idx, fname in enumerate(tqdm(images, desc='Processing image')):
        img = imread(fname)
        if img.shape[0] != CAL_IMAGE_SIZE[0] or img.shape[1] != CAL_IMAGE_SIZE[1]:
            img = imresize(img, CAL_IMAGE_SIZE)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            successfull_cnt += 1

            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
            cal_images[idx] = img

    print("%s/%s camera calibration images processed." % (successfull_cnt, len(images)))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size[:-1], None, None)

    calibration = {'objpoints': objpoints,
                   'imgpoints': imgpoints,
                   'cal_images': cal_images,
                   'mtx': mtx,
                   'dist': dist,
                   'rvecs': rvecs,
                   'tvecs': tvecs}

    return calibration

def get_camera_calibration():
    """
    Depending on the constant CALC_CAL_POINTS the camera calibration will be
    calculated and stored on disk or loaded.
    """
    if CALC_CAL_POINTS:
        calibration = calculateCalibration(imagePath, ROWS, COLS)
        with open(calOutput, 'wb') as f:
            pickle.dump(calibration, file=f)
    else:
        with open(calOutput, "rb") as f:
            calibration = pickle.load(f)

    return calibration


class CameraCalibrator:
    def __init__(self, image_shape, calibration):
        """
        Helper class to remove lens distortion from images
        :param image_shape: with and height of the image
        :param calibration: calibration object which can be retrieved from "get_camera_calibration()"
        """
        self.objpoints = calibration['objpoints']
        self.imgpoints = calibration['imgpoints']
        self.image_shape = image_shape

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(self.objpoints, self.imgpoints, image_shape, None, None)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)