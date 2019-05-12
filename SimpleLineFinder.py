import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from moviepy.editor import VideoFileClip
from IPython.display import HTML



fname = "./test_images/test2.jpg"
image = mpimg.imread(fname)
plt.imshow(image)
plt.show()


def undistort_image(dist_img):
    dist_pickel = pickle.load(open("./camera_cal/calibration_pickle.p", "rb"))
    mtx = dist_pickel["mtx"]
    dist = dist_pickel["dist"]

    undist_img = cv2.undistort(dist_img, mtx, dist, None, mtx)

    return undist_img


undist_img = undistort_image(image)

output_fname = "./output_images/" + fname.split('/')[-1]
output_img = cv2.cvtColor(undist_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_fname, output_img)


def binary_threshold(img, s_thresh=(120, 255), sx_thresh=(20, 100)):
    out_img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(out_img, cv2.COLOR_RGB2HLS).astype(np.float)
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
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary


color_binary, combined_binary = binary_threshold(undist_img)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('stacked thresholded image')
plt.imshow(color_binary)
plt.subplot(1, 2, 2)
plt.title('combined thresholded image')
plt.imshow(combined_binary, cmap='gray')


def warp(img, src, dst, inverse=False):
    if inverse:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    transformed = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    return transformed


src = np.float32([[200, 720], [1125, 720], [685, 450], [595, 450]])
dst = np.float32([[320, 720], [1000, 720], [1000, 0], [320, 0]])

fname = "./output_images/straight_lines1.jpg"
src_img = mpimg.imread(fname)

dst_img = warp(src_img, src, dst, False)

src_pts = src.reshape((-1, 1, 2)).astype("int32")
dst_pts = dst.reshape((-1, 1, 2)).astype("int32")
cv2.polylines(src_img, [src_pts], True, (0, 255, 255), thickness=5)
cv2.polylines(dst_img, [dst_pts], True, (255, 0, 255), thickness=5)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('src image')
plt.imshow(src_img)
plt.subplot(1, 2, 2)
plt.title('warped image')
plt.imshow(dst_img)

binary_warped = warp(combined_binary, src, dst, False)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('thresholded image')
plt.imshow(combined_binary, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('warped thresholded image')
plt.imshow(binary_warped, cmap='gray')
plt.show()

histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
plt.plot(histogram)


def sliding_window_search(img):
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
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


left_fit, right_fit, out_img = sliding_window_search(binary_warped)

# Generate x and y values for plotting
ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)


def get_curverad(left_fit, right_fit):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    y_eval = np.max(ploty)
    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    curverad = (left_curverad + right_curverad) / 2

    return curverad


curverad = get_curverad(left_fit, right_fit)
# Now our radius of curvature is in meters
print(curverad, 'm')


def get_offset(left_fit, right_fit):
    lane_width = 3.7  # metres
    h = 720
    w = 1280

    left_pix = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
    right_pix = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]

    scale = lane_width / np.abs(left_pix - right_pix)

    midpoint = np.mean([left_pix, right_pix])

    offset = (w / 2 - midpoint) * scale
    return offset


offset = get_offset(left_fit, right_fit)
# Now our radius of curvature is in meters
print(offset, 'm')


def draw_lines(warped, undist, left_fit, right_fit):
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

    newwarp = warp(color_warp, src, dst, True)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    curverad = get_curverad(left_fit, right_fit)
    curvature_text = "Curvature: " + str(np.round(curverad, 2))
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result, curvature_text, (30, 60), font, 1, (0, 255, 0), 2)

    offset = get_offset(left_fit, right_fit)
    offset_text = "Lane offset from center = {:.2f} m".format(offset)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result, offset_text, (30, 90), font, 1, (0, 255, 0), 2)

    return result


result = draw_lines(binary_warped, undist_img, left_fit, right_fit)
plt.imshow(result)

last_left_fit = None
last_right_fit = None


def process_image(image):
    """
    Execute our image processing pipeline on the provided image.
    """
    global last_left_fit, last_right_fit
    alpha = 0.2

    undistorted = undistort_image(image)
    _, combined_binary = binary_threshold(undistorted, (150, 255), (20, 100))

    src = np.float32([[200, 720], [1125, 720], [685, 450], [595, 450]])
    dst = np.float32([[320, 720], [1000, 720], [1000, 0], [320, 0]])
    binary_warped = warp(combined_binary, src, dst, False)

    left_fit, right_fit, _ = sliding_window_search(binary_warped)

    if last_left_fit is not None:
        left_fit = (1 - alpha) * last_left_fit + alpha * left_fit
        right_fit = last_right_fit * (1 - alpha) + right_fit * alpha

    last_left_fit = left_fit
    last_right_fit = right_fit

    result = draw_lines(binary_warped, undistorted, left_fit, right_fit)

    return result

# Run on a test image
img = cv2.imread("test_images/test6.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = process_image(img)

plt.figure(figsize=(16,8))
plt.imshow(result)
plt.axis("off");

video_output = "output_images/project_video.mp4"
clip1 = VideoFileClip("project_video.mp4", audio=False)
clip1_output = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clip1_output.write_videofile(video_output, audio=False)