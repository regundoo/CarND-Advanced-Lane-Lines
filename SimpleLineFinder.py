import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from moviepy.editor import VideoFileClip

# Import camera calibration class and perspective transformation class
from PerTransformer import PerTransformerClass
from CameraCalibration import calibrateCamera


# calculate undistorted image
def undistort_image(dist_img, show=True):
    # Read camera mtx and dist from file and safe in variable, which is safed in the calibration_pickle.p.
    dist_pickel = pickle.load(open("./camera_cal/calibration_pickle.p", "rb"))
    mtx = dist_pickel["mtx"]
    dist = dist_pickel["dist"]

    # undist the given image with this function
    undist_img = cv2.undistort(dist_img, mtx, dist, None, mtx)
    
    # if plot funciton is needed
    if show:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        ax1.imshow(cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB))
        ax2.set_title('Undistorted Image', fontsize=20)
        return undist_img
    else:
        return undist_img

# add gaussian blur to the image. This improves the quality of the lines
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def binary_threshold(img, s_thresh, sx_thresh):
    out_img = np.copy(img)
    # Convert to HLS color space 
    hls = cv2.cvtColor(out_img, cv2.COLOR_RGB2HLS).astype(np.float)
    # Separate the S channel
    s_channel = hls[:, :, 2]

    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the thresholds to the combined binary image and return it
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary


def sliding_window_search(img):
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255
    # Find the peak of the left and right halves of the histogram
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # Find the midle of the histogram
    midpoint = np.int(histogram.shape[0] / 2)
    left_bound = 150
    right_bound = 1150
    leftx_base = np.argmax(histogram[left_bound:midpoint]) + left_bound
    rightx_base = np.argmax(histogram[midpoint:right_bound]) + midpoint
    # print(leftx_base, rightx_base, midpoint)

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
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

    left_lane_inds = np.concatenate(left_lane_inds)
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


def get_curverad(left_fit, right_fit, out_img):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    y_eval = np.max(ploty)

    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radius of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    curverad = (left_curverad + right_curverad) / 2

    return curverad


def get_offset(left_fit, right_fit, img_size):
    # caluclate the offset of the camera between the lines
    lane_width = 3.7  # metres
    h = img_size[1]
    w = img_size[0]

    left_pix = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
    right_pix = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]

    scale = lane_width / np.abs(left_pix - right_pix)

    midpoint = np.mean([left_pix, right_pix])
    # calculate the distance from midpoint in given scale
    offset = (w / 2 - midpoint) * scale
    return offset


def draw_lines(warped, undist, left_fit, right_fit, src, dst, out_img, count, img_size):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Transfrom the image to the given warp
    newwarp = PerTransformerClass(src, dst).inverse_transform(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # caluclate the curvature rad
    curverad = get_curverad(left_fit, right_fit, out_img)
    curvature_text = "Curvature: " + str(np.round(curverad, 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, curvature_text, (30, 60), font, 1, (0, 255, 0), 2)
    # print the calculated values to the image as an output
    offset = get_offset(left_fit, right_fit, img_size)
    offset_text = "Lane offset  = {:.2f} m".format(offset)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, offset_text, (30, 90), font, 1, (0, 255, 0), 2)

    cv2.putText(result, '%4d' % count, (5, 700), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result


last_left_fit = None
last_right_fit = None

# Define src and dst matrix for image warp
def calc_warp_points(img_size):
    src = np.float32([
        [545, 450],
        [742, 450],
        [1602, 720],
        [-261, 720]
    ])

    dst = np.float32([
        [0, 0],
        [img_size[0], 0],
        [img_size[0], img_size[1]],
        [0, img_size[1]]
    ])
    return src, dst


def process_image(image):
    # This function processes all the images
    # It checkes if camera calibration output is available. If not, the calibraiton will be performed first
    global last_left_fit, last_right_fit
    alpha = 0.2

    count = 0
    kernel_size = 3
    img_size = (image.shape[1], image.shape[0])
    # print(img_size)

    exists = os.path.isfile("./camera_cal/calibration_pickle.p")

    if exists:
        # undistored the images
        undistorted = undistort_image(image)
        # add gaussian blur 
        gaussian_blurimage = gaussian_blur(undistorted, kernel_size)

        s_thresh = (150, 255)
        sx_thresh = (20, 100)
        
        # get thoreshold images
        color_binary, combined_binary = binary_threshold(gaussian_blurimage, s_thresh, sx_thresh)
        # get src and dst from def
        src, dst = calc_warp_points(img_size)
        # warp the image with pertransformerclass
        binary_warped = PerTransformerClass(src, dst)
        binary_warped = binary_warped.transform(combined_binary)

        left_fit, right_fit, out_img = sliding_window_search(binary_warped)

        if last_left_fit is not None:
            left_fit = (1 - alpha) * last_left_fit + alpha * left_fit
            right_fit = last_right_fit * (1 - alpha) + right_fit * alpha

        last_left_fit = left_fit
        last_right_fit = right_fit

        result = draw_lines(binary_warped, undistorted, left_fit, right_fit, src, dst, out_img, count, img_size)

        return result
    else:
        # perform camera calibration
        calibrateCamera()
        # undistored the images
        undistorted = undistort_image(image)
        # add gaussian blur 
        gaussian_blurimage = gaussian_blur(undistorted, kernel_size)

        s_thresh = (150, 255)
        sx_thresh = (20, 100)
        # get thoreshold images
        color_binary, combined_binary = binary_threshold(gaussian_blurimage, s_thresh, sx_thresh)
        # get src and dst from def
        src, dst = calc_warp_points(img_size)
        # warp the image with pertransformerclass
        binary_warped = PerTransformerClass(src, dst)
        binary_warped = binary_warped.transform(combined_binary)

        left_fit, right_fit, out_img = sliding_window_search(binary_warped)

        if last_left_fit is not None:
            left_fit = (1 - alpha) * last_left_fit + alpha * left_fit
            right_fit = last_right_fit * (1 - alpha) + right_fit * alpha

        last_left_fit = left_fit
        last_right_fit = right_fit

        result = draw_lines(binary_warped, undistorted, left_fit, right_fit, src, dst, out_img, count, img_size)

        return result

# decide if the test image shall be used or the project video processed
photo = False

if photo:
    # Run on a test image
    img = cv2.imread("test_images/test4.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = process_image(img)

    plt.figure(figsize=(16, 8))
    plt.imshow(result)
    plt.show()
    plt.axis("off")
else:
    video_output = "output_images/project_video.mp4"
    clip1 = VideoFileClip("project_video.mp4", audio=False)
    clip1_output = clip1.fl_image(process_image)
    clip1_output.write_videofile(video_output, audio=False)
