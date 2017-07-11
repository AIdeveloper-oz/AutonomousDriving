**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/car_hog.jpg
[image3]: ./output_images/notcar_hog.jpg
[image4]: ./output_images/bbox.png
[image5]: ./output_images/false_pos_filter.jpg
[image6]: ./output_images/result.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. Features extraction.

The code for this step is contained in funtion `single_img_features()` (lines 63-99 in `src/feature.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images in function `prepare_data()` (lines 25-68 in `src/utils.py`).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are examples using the `HSV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]

####2. HOG parameters choice.

I tried various combinations of parameters and finally I use the following settings  (lines 31-42 in `src/main.py`):

```python
param = {
    'color_space' : 'HSV', # Can be RGB, HSV, (LUV, HLS, YUV, YCrCb leads to Nan for PNG image)
    'orient' : 11,  # HOG orientations
    'pix_per_cell' : 8, # HOG pixels per cell
    'cell_per_block' : 2, # HOG cells per block
    'hog_channel' : "ALL", # Can be 0, 1, 2, or "ALL"
    'spatial_size' : (16, 16), # Spatial binning dimensions
    'hist_bins' : 32,    # Number of histogram bins
    'spatial_feat' : True, # Spatial features on or off
    'hist_feat' : True, # Histogram features on or off
    'hog_feat' : True, # HOG features on or off
}
```

####3. Classifier training.

To avoid the problem of time-series data for train/test split, I chose the first 90% images in each folder for training and the last 10% for testing. Besides, I did left-right flip to augment the data.

I used both HOG and color histogram feature mentioned above and trained a linear SVM using different `C` parameter, but the only differenc is the training speed and the test Accuracy is the same.
```
5.92 Seconds to train SVC with C=10.000000...
Test Accuracy of SVC =  0.9902
24.37 Seconds to train SVC with C=1.000000...
Test Accuracy of SVC =  0.9902
24.41 Seconds to train SVC with C=0.100000...
Test Accuracy of SVC =  0.9902
```
Finally, I use `C=1` and all the data to train the classifer.

###Sliding Window Search

####1. Implementation and parameter choice.

The code for this step is contained in funtion `multiscale_search()` (lines 89-102 in `src/fast_multiscale_search.py`).  

To speed up, I first extracted hog features of the whole images and sub-sampled these features to get all of the overlaying windows. I decided to search windows in scale `[1.3, 1.5, 1.8, 2.2]` and limit different earch regions for different scales, since smaller objects appear in farther distance.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Final video output. 
Here's a [link to my video result](https://youtu.be/FHHEedd39wk)


####2. False positives filter and temporal smoothing.

The code of this step is in function 'false_pos_filter()' (lines 59-100 in `src/false_pos_filter.py`).  

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  To make the reulsts more smooth and robust, I also allpy a temporal window of 6 frames to smooth the result (lines 66-73 in `src/false_pos_filter.py`).

### Here is an example result showing the heatmap, the result of `scipy.ndimage.measurements.label()` and the bounding boxes:
![alt text][image5]

### Here the resulting bounding boxes are drawn onto the project_video.mp4:
![alt text][image6]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The `mpimg.imread()` gives different range for png and jpg images. I used `mpimg.imread()*255` for png images but seems incorrect. Since the `cv2.cvtColor()` can rescale the range to the same for png and jpg images, I used the combination of `mpimg.imread()` and `cv2.cvtColor()` to avoid the range problem.

There are still some false positive in the results, especially during serious illumination variation (e.g. shadow). I think more data collection and augmentation can help to aleviate this problem.

