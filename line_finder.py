# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:52:08 2019

@author: Y1PGLOCK
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


#reading in an image
image = mpimg.imread('test_images/test1.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
plt.show()
print("Test")


class LaneFinder:
    def grayscale(img):
        # Nothing changed here
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def canny(img, low_threshold, high_threshold):
        # Nothing changed here
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)


    def gaussian_blur(img, kernel_size):
        # Nothing changed here
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


    def region_of_interest(img, vertices):
        # Nothing changed here
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        # create empty fields for lines, define lenght of the line and cooridnates
        left_lines = []
        left_length = []
        right_lines = []
        right_length = []

        if lines is not None:
            for line in lines:
                # for every line out of the hough line do ...
                for x1, y1, x2, y2 in line:
                    # calculate the slope of every line from hough object
                    slope = ((y2 - y1) / (x2 - x1))
                    # calculate the intersection of every line with the y achsis to complete line formula
                    intercept = -(x2 * slope) + y2
                    # caluclate the distance between the two data points: p1p2=sqrt((x2-x1)^2+(y2-y1)^2)
                    # This will give us the chance to adopt every line to the same scale, it is not needed to plot the lines
                    laenge = int(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))

                    # split the lines in right and left
                    if slope < -0.5:
                        left_lines.append((slope, intercept))
                        left_length.append((laenge))
                    elif slope > 0.5:
                        right_lines.append((slope, intercept))
                        right_length.append((laenge))
                    else:
                        continue

                        # average all the lines and create on lasting line out of the average
            # other method would be to fit the line with slope and intersect with polyfit, average should work fine here
            left_lane = np.dot(left_length, left_lines) / np.sum(left_length)
            right_lane = np.dot(right_length, right_lines) / np.sum(right_length)

            linien = left_lane, right_lane

            for lane in linien:
                slope, intercept = lane
                # find y and x coordinates to shown slope and intersect. y1 will start at the max. value of the image shape
                y1 = int(img.shape[0])
                # y2 will be extedned to the max. value of the trapez shape. For easier adaption, the value is used and not the variable name.
                y2 = int(imshape[0] - imshape[0] * 0.4)
                # calculate the fitting x positions to defined ys and slops
                x1 = int(np.nan_to_num((y1 - intercept) / slope))
                x2 = int(np.nan_to_num((y2 - intercept) / slope))

                # Draw line
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            # z = np.polyfit(x, y, 1)
            # print(z)


    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
        # Nothing changed here
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        draw_lines(line_img, lines)
        return line_img


    # Python 3 has support for cool math symbols.

    def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
        # Nothing changed here
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + γ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, γ)


    os.listdir("test_images/")

    # Build your pipeline that will draw lane lines on the test_images
    # then save them to the test_images_output directory.

    # list pipeline and define dir
    imagelist = os.listdir("test_images/")
    dirname = 'test_images/'
    print(imagelist)

    saveimagelist = 'images_output/'

    # define Global parameters

    # Canny parameters
    low_threshold = 50
    high_threshold = 150

    # Gaussian parameters
    kernel_size = 3

    # Hough parameters
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_len = 10  # min numbers of pixels to define the lane
    max_line_gap = 20  # gap between lanes

    # Mask parameters for trapezoid shape, values are shown in percentage to the image sizes
    trapbottom = 0.85
    traptop = 0.07
    trapheight = 0.4


    def processimages(image):
        # plt.imshow(image)
        # transform image to gray scal eimage
        grayimage = grayscale(image)

        imageblur = gaussian_blur(grayimage, kernel_size)
        cannyimage = canny(imageblur, low_threshold, high_threshold)

        # define mask
        imshape = image.shape
        # vertices = np.array([[150,540],[400,330], [560,330],[810,540]], np.int32)
        # define trapezoid shape out of the variables of the image and the defined sizes
        vertices = np.array([[ \
            ((imshape[1] * (1 - trapbottom)) // 2, imshape[0]), \
            ((imshape[1] * (1 - traptop)) // 2, imshape[0] - imshape[0] * trapheight), \
            (imshape[1] - (imshape[1] * (1 - traptop)) // 2, imshape[0] - imshape[0] * trapheight), \
            (imshape[1] - (imshape[1] * (1 - trapbottom)) // 2, imshape[0])]] \
            , dtype=np.int32)
        cropimage = region_of_interest(cannyimage, vertices)

        line_image = np.copy(cropimage) * 0
        line_image = hough_lines(cropimage, rho, theta, threshold, min_line_len, max_line_gap)

        final_image = weighted_img(line_image, image, α=0.8, β=1., γ=0.)
        # plt.figure(i+1)
        # plt.imshow(line_image, cmap='gray')
        plt.imshow(final_image, cmap='gray')
        path = 'output_images'
        cv2.imwrite(os.path.join(path, 'waka.jpg'), final_image)


i = 0

# for every image in the directory, do the investigation Always the last image will be shown in the output
for i in range(len(imagelist)):
    image = mpimg.imread(os.path.join(dirname, imagelist[i]))
    imshape = image.shape
    processimages(image)
    print('image:',imagelist[i],' is:', type(image), 'with dimensions:', image.shape)


