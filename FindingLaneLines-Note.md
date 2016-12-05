Color selection is not reliable, since it is very sensitive to illumination variations. So the standard pipeline of Lane Line Detection is: 
**Grayscale --> Gaussian Blur --> Canny Edge Detection --> ROI --> Hough Transform Line Detection --> Fit Lines With Slope --> Filter wrong lines with Slope --> Adaptive Temporally Smooth**
The fore stages should guarantee that all lines be detected, and the later stages should remove the noise as much as possibile.


###1. Canny Edge Detection using OpenCV function `Canny`
```python
edges = cv2.Canny(gray, low_threshold, high_threshold)
```
Applying Canny to the image **gray** and your output will be another image called **edges**. **low_threshold** and **high_threshold** are your thresholds for edge detection. The algorithm will first detect strong edge (strong gradient) pixels above the high_threshold, and reject pixels below the low_threshold. Next, pixels with values between the **low_threshold** and **high_threshold** will be included as long as they are connected to strong edges. The output edges is a binary image with white pixels tracing out the detected edges and black everywhere else.

<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/CannyDetection.png" width="800"/>
</p>


###2. Hough Transform on Edges to detect lines using an OpenCV function called HoughLinesP
```python
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
```
Operating on the image **edges** (the output from Canny) and the output from `HoughLinesP` will be lines, which will simply be an array containing the endpoints (x1, y1, x2, y2) of all line segments detected by the transform operation. The other parameters define just what kind of line segments we're looking for.

**rho** and **theta** are the distance and angular resolution of our grid in Hough space. Remember that, in Hough space, we have a grid laid out along the (Θ, ρ) axis. You need to specify **rho** in units of pixels and **theta** in units of radians.

**rho** takes a minimum value of 1, and a reasonable starting place for **theta** is 1 degree (pi/180 in radians). Scale these values up to be more flexible in your definition of what constitutes a line.

The **threshold** parameter specifies the minimum number of votes (intersections in a given grid cell) a candidate line needs to have to make it into the output. The empty **np.array([])** is just a placeholder, no need to change it. **min_line_length** is the minimum length of a line (in pixels) that you will accept in the output, and **max_line_gap** is the maximum distance (again, in pixels) between segments that you will allow to be connected into a single line.

<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/HoughTransform.png" width="800"/>
</p>


###3. Basics of Jupyter Notebook and Python
see https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python


###4. Future Work
The framework used here is useful in simple situations. But when in more complex scenes such as occlusions, bent roads, snowy or rainy weather, there will be more noises which may lead to failure. I imagine the temporal infomation can help to reduce the noisy and overcome the occlusions to make the detection more smooth temporally. Besides, there are too many hyperparameters here, which limits the generation ability of the methods. 

I think there are several ways to improve the performance:

1) **HSV Color Selection:**, HSV is insensitive to brightness, but it may cut the generalization of the methods to other videos.

2) **Histogram Equalization:** use cv2.equalizeHist(img) before blurring.

3) **Adaptive Strategy:** I apply a Adaptive Temporally Smooth strategy in the line paramether update, and many other strategy can be applied to make methods adaptive.

4) **Data-driven Machine Learning:** recently, the semantic segmentation is a popular topic, which can be used to find lane lines in uncontrolled situations.


###5. Further reading: Improvement & Resources
Canny Edge Detection - In this portion of the pipeline, there are two main parameters you can tune: lower threshold and higher threshold. Your parameters here seem reasonable. If you want to play around with some more parameters here, check out this [link](http://stackoverflow.com/questions/21324950/how-to-select-the-best-set-of-parameters-in-canny-edge-detection-algorithm-imple). The tutorial in the link will describe a common method for choosing threshold in Canny Edge Detection.

This research [paper](http://airccj.org/CSCP/vol5/csit53211.pdf) goes into details on how to detect curves and faded lanes. It uses an extended version of hough lines algorithm to detect tangents to the curve which can help you detect the curve.
