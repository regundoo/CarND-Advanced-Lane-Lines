import glob
import pickle

import cv2
import numpy as np


def CalibrateCamera():
    # prepare opject points
    nx = 6
    ny = 9

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (ny, nx), None)

        # If found, add object points, image points
        if ret:
            print('Found object points and image points in: ' + fname)
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (ny, nx), corners, ret)
            output_fname = './camera_cal/found/' + fname.split('\\')[-1]
            print('Write ChessboardCorners to: ' + output_fname)

    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open("./camera_cal/calibration_pickle.p", "wb"))

    print('Camera Calibration done!')

    # dist_img = mpimg.imread('./camera_cal/calibration2.jpg')

    # undist_img = cv2.undistort(dist_img, mtx, dist, None, mtx)

    # plt.figure(figsize=(2, 1))

    # plt.subplot(1, 2, 1)
    # plt.title('dist')
    # plt.imshow(dist_img)
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.title('undist')
    # plt.imsave('output_images/calibration2.jpg', undist_img)
    # plt.imshow(undist_img)
