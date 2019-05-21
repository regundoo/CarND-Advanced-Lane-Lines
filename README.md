# Advanced Lane Lines finder
---

The following requirements are given for the advanced lane lines finder project:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./Writeup_images/1.png 
[image2]: ./Writeup_images/2.png 
[image3]: ./Writeup_images/hls.png 
[image4]: Writeup_images/combined2.png 
[image5]: ./Writeup_images/5.png 
[image6]: ./Writeup_images/6.png 
[image7]: ./Writeup_images/7.png 
[image8]: ./Writeup_images/8.png 
[video1]: ./output_images/project_video.mp4 "Video"

[Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---

### Camera Calibration
The first thing that should be done is the camera calibration. The camera calibration is performed by the python file 'CameraCalibration.py' in the main directory.
Camera calibration can be called directly over the script. The Main Lane Finder programm will also check if the camera calibration was done. If not, it will run the calibration in advance.  

The camera calibration works the following: The distorted image will be undistorted to improve the quality of geometrical measurement.
The procedure follows the following steps:
* The 'object points' are prepared, which will be the x, y and z coordinates of the chessboard corners in the real world. 
The chessboard is assumed to be fixed in the z direction and the size of the chessboard is the same for all images. Therefore, the 
object points are the same for each calibration image.
* After transfering the calibration images into grayscale, the chessboard pattern is detected with the cv2.findChessboardCorners. 
The found corners get appended to the list of all image points and the according 'object points' get appended to the list of object points.
* The distortion coefficient and the camera calibration matrix are comupted with the cv2.calibrateCamera() function. The caluclated
parameters are pickled into the CameraCalibration_pickle file for further usage.


![undistortedimage][image1]

### Image Processing

Since we learned in the Camera Calibration that it is nessesary to undistort every incomming image. The first
step with every image from the image pipeline is to run the function of undistort the image. The following picures
shows an example of the original image and the same image undistorted.
For the undistortion the calibration_pickle.p file is loaded to get all the inputs nessesary for the cv2.undistort() function.
![undistortedimageOriginal][image2]


#### Thresholded for the binary image.

Before using the threshold filtering for the images, a gaussian blurring is applied to reduce the image noice. The function used
is the following:

```python
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

The HLS channels of the image are explored in the next step:
![HLStransfer][image3]

It is seen that the L channel is very sensitive to changes in the lighting. Since I don't tune the L channel for every image,
only the S and H channel are used for the filtering. Therefore, the two channels are combined to create the Combined_binary
image filter. The sobel filter is also adopted to the color filtered images. 

![conbinedImage][image4]

#### Perspective transformation

Next step in the image process is the perspective transformation. Therefore, the PerTransformer class is used. 
With the cv2.warp function, the image is transferred to the bird-eye view. The class also inclueds a method for the inverse
matrix transformation to reverse the process later on.
![unwarpedImage][image5]

#### Lane Line detection and sliding window method

With the sliding window method, the lane lines are detected from the transformed bird-eye view.
The histogram chart of the pixel values is created in the x dimention. Only the lower half of the image is used, since
this is the part where the lanes are. The peaks in the histogram are showing the line points. Since there are two peaks,
the left and the right line can be found.
![slidingwindow][image6]

The image is devided into 16 horizontal slices and the histogram is caluclated for each slice. All found points are fitted
with the polyfit function to create an interpolated line that represents the acutall lane on the road.
![slidingwindow][image7]
Note that this image only represents the process and does not show the used 16 slinding windows in the code.

#### Find the curvature of the lane and the position of the car between the lines

The curveture of the fitted functions is caluclated with the following code:
```python
left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    curverad = (left_curverad + right_curverad) / 2

```

The represented curvature on each image is the avarage of both measured curves.
The calculated pixels are fitted to the real world with the pixel per meter mapping function.

#### Create the processed output image

The processed image is warped back to the original geometry with the inverse function. All details about curvature radius and distance
of the vihicle to the center is printed on the image itself.

![perfectimage][image8]

### Video processing

The video processing works with the same code and parameters. Each frame of the video is a single picture and can be handled
the same way as the example pictures. After processing each single picture, they can be put back together to a video. 

The output video can be found here:
![Project Video][video1]

### Discussion

The biggest issue in this project are sudden changes of the light condition. The code uses a lot of parameter tuning with
trail and error. With changing conditions, this tuning is not working any more and lines get either completely lost or images
get filled with noise. A method to compensate for the changing brightness of the images is nessesary to create a robust
code for the changing conditions. The sliding window method works with perfect lines on the road but as soon as there are
two lines very close together or even more lines available, you can't decide anymore which line is the correct one.


