**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort1.png "Undistorted Example 1"
[image2]: ./output_images/undistort2.png "Undistorted Example 2"
[image3]: ./output_images/threshold.png "Binary Example"
[image4]: ./output_images/undist_transform.png "Warp Example"
[image5]: ./output_images/window_fit.png "Fit Visual"
[image6]: ./output_images/result.png "Output"
[video1]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Calculate the camera matrix and distortion coefficients. 

The code for this step is contained in the function get_calibration() (lines 9-87 in `src/calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion correction.

I use the camera matrix and distortion coefficients obtained from the camera calibration step for distortion correction with the `cv2.undistort` function (lines 18 in `src/transform.py`). I then apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Thresholding.

I used a combination of color (lines 103 in `src/threshold.py`) and gradient (lines 107 in `src/threshold.py`) thresholds to generate a combined binary image (lines 110 in `src/threshold.py`). Afterwards, I select the region of interest and apply the morphology operations to remove the noise and fill the holes (lines 113-119 in `src/threshold.py`). Here's an example of my output for this step.

![alt text][image3]

#### 3. Perspective transform.

The code for my perspective transform includes a function called `transform()`, which appears in lines 13 through 47 in the file `src/transform.py` (output_images/examples/example.py).  The `transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the source and destination points manually according to the lane lines on the test image `test_images/straight_lines1.jpg`:

```python
    trans_src = np.float32([[225,697], [1078,697], [705,460], [576,460]])
    offset = 360
    trans_dst = np.float32([[img_W/2-offset,img_H], [img_W/2+offset,img_H], [img_W/2+offset,0], [img_W/2-offset,0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 225, 697      | 380, 720        | 
| 1078, 697      | 1000, 720      |
| 705, 460     | 1000, 0      |
| 576, 460      | 380, 0        |

I verified that my perspective transform was working as expected by checking the warped image to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Finding lane-line pixels and fitting with a polynomial.

I use the convolve-based sliding window searching method to find the lane-line pixels in the function called `sliding_win_search()` which appears in lines 85 through 179 in the file `src/sliding_win_search.py`. Then I fit my lane lines with a 2nd order polynomial (lines 106-107 in `src/sliding_win_search.py`) like this:

![alt text][image5]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to the center.

I did this in lines 122 through 124 in my code in `src/main.py`

#### 6. Plot results back down onto the road.

I implemented this step in lines 103 through 118 in my code in `src/main.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### Final video output.  

Here's a [link to my video result](https://youtu.be/3Z8t-fboAi0)

---

### Discussion

Actually, my pipeline will fail when there is a serious illumination jitter. I have used sanity check (lines 87-88 in `src/main.py`) and smooth strategy to reduce the failure frames. I think using the simple hand-crafted gradient and color feature are not some robust to various scenes, maybe the semantic segmentation method can pursue this project further. 

