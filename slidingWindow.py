# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:54:31 2019

@author: Y1PGLOCK
"""
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class slidingWindowSearch:
    def __init__(self, img):
        self.img = img
        self.out_img = np.dstack((img, img, img))*255
        self.histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        self.midpoint = np.int(self.histogram.shape[0] / 2)
        self.left_bound = 150
        self.right_bound = 1150
        self.leftx_base = np.argmax(self.histogram[self.left_bound:self.midpoint]) + self.left_bound
        self.rightx_base = np.argmax(self.histogram[self.midpoint:self.right_bound]) + self.midpoint
        # print(leftx_base, rightx_base, midpoint)
    
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set height of windows
        self.window_height = np.int(img.shape[0] / self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        self.nonzero = img.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        # Current positions to be updated for each window
    def sliding_window_search(self, img, nwindows, windows_height):
        # Create an output image to draw on and  visualize the result
        #out_img = np.dstack((img, img, img)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            # print(win_y_low, win_y_high)
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            # print(win_xleft_low, win_xleft_high)
    
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
    
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            # print(good_left_inds)
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            else:
                if len(nonzerox[np.concatenate(left_lane_inds)]):
                    leftx_current = np.int(np.mean(nonzerox[np.concatenate(left_lane_inds)]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            else:
                if len(nonzerox[np.concatenate(right_lane_inds)]):
                    rightx_current = np.int(np.mean(nonzerox[np.concatenate(right_lane_inds)]))
            # print(nonzerox[good_right_inds])
            # print(rightx_current)
        # Concatenate the arrays of indices
        # print(left_lane_inds)
        left_lane_inds = np.concatenate(left_lane_inds)
        # print(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
    
        return left_fit, right_fit, out_img