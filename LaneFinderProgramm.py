import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import cv2
import os
import numpy as np
import math

class LaneFinder:
    def __init__(self, mtx, dist, warp_size, img_size):
        self.mtx = mtx
        self.dist = dist
        self.warp_size = warp_size
        # self.unwarp_size = 
        self.img_size = img_size
        self.mask = np.zeros((warp_size[1], warp_size[0], 3), dtype=np.uint8)
        self.roi_mask = np.ones((warp_size[1], warp_size[0], 3), dtype=np.uint8)


    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist)

    def warp(self, combined_binary, M, plot=True):
        warped_image = cv2.warpPerspective(combined_binary, M, (combined_binary.shape[1], combined_binary.shape[0]), flags=cv2.INTER_LINEAR)  # keep same size as input image
        if(plot):
            # Ploting both images Binary and Warped
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.set_title('Binary/Undistorted and Tresholded')
            ax1.imshow(combined_binary, cmap='gray')
            ax2.set_title('Binary/Undistorted and Warped Image')
            ax2.imshow(warped_image, cmap='gray')
            plt.show()
        
        return warped_image

    def unwarp(self):
        pass

    def equalize_lines(self):
        pass

    def find_lanes(self, img):
        
        src, dst = self.calc_warp_points(img)
        print(src)
        print(dst)
        M = cv2.getPerspectiveTransform(src, dst)

        img = self.undistort(img)

        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls = cv2.medianBlur(hls, 5)
        s_channel = hls[:,:,2]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

        plt.imshow(img)
        plt.show()
        plt.imshow(hls)
        plt.show()
        plt.imshow(lab)
        plt.show()
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        plt.imshow(gray, cmap = 'gray')
        plt.show()
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        plt.imshow(scaled_sobel, cmap = 'gray')
        plt.show()
        
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
 
        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
 
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        
        plt.imshow(combined_binary, cmap = 'gray')
        plt.show()
        
        warpImage = self.warp(combined_binary, M)

        # This gives you the elliptical kernal back.
        # big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        # This gives you the rectengular gernal
        # small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # greenery = (lab[:, :, 2].astype(np.uint8) > 145) & cv2.inRange(hls, (0, 0, 50), (35, 190, 255))

        # plt.imshow(greenery, cmap = 'gray')
        # plt.show()

        # mask_road = np.logical_not(greenery).astype(np.uint8) & (hls[:, :, 1] < 250)

        # This function is used to remove the noise of the iage
        # mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, small_kernel)
        # plt.imshow(mask_road, cmap = 'gray')
        # plt.show()

        windowsimage = self.sliding_window_search(warpImage)
        # This combines the mask_road with the kernel
        


       

    def sliding_window_search(self, img):
        out_img = np.dstack((img, img, img)) * 255
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        plt.plot(histogram)

    def process_image(self, img):
        pass

    def calc_warp_points(self, img):
#        shape = img.shape[::-1] # (width,height)
#        w = shape[0]
#        h = shape[1]
#        """
#        :return: Source and Destination pointts
#        """
#        src = np.float32([ [580,450], [160,h], [1150,h], [740,450]]) 
#    
#        dst = np.float32([ [0,0], [0,h], [w,h], [w,0]])
        
        corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
        top_left=np.array([corners[0,0],0])
        top_right=np.array([corners[3,0],0])
        offset=[150,0]
        
        src = np.float32([corners[0],corners[1],corners[2],corners[3]])
        dst = np.float32([corners[0]+offset,top_left+offset,top_right-offset ,corners[3]-offset])
        
        return src, dst




def loadCalibrationData():
    dist_pickel = pickle.load(open("./camera_cal/calibration_pickle.p", "rb"))
    mtx = dist_pickel["mtx"]
    dist = dist_pickel["dist"]

    return mtx, dist





mtx, dist = loadCalibrationData()
M_inv = cv2.getPerspectiveTransform(dst, src)


output_dir = "output_images"


img = mpimg.imread("test_images/test1.jpg")
# plt.imshow(img)
# plt.show()

img_size = (img.shape[1], img.shape[0])
warp_size = (1280, 720)


lf = LaneFinder(mtx, dist, warp_size, img_size)
processedImage = lf.find_lanes((img))
print(lf)
