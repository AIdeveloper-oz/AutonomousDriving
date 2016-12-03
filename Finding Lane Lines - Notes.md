###1. Canny Edge Detection using OpenCV function `Canny`
```python
edges = cv2.Canny(gray, low_threshold, high_threshold)
```
Applying Canny to the image **gray** and your output will be another image called **edges**. **low_threshold** and **high_threshold** are your thresholds for edge detection. The algorithm will first detect strong edge (strong gradient) pixels above the high_threshold, and reject pixels below the low_threshold. Next, pixels with values between the **low_threshold** and **high_threshold** will be included as long as they are connected to strong edges. The output edges is a binary image with white pixels tracing out the detected edges and black everywhere else.

###2. Hough Transform on Edges to detect lines using an OpenCV function called HoughLinesP
```python
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
```
Operating on the image **edges** (the output from Canny) and the output from `HoughLinesP` will be lines, which will simply be an array containing the endpoints (x1, y1, x2, y2) of all line segments detected by the transform operation. The other parameters define just what kind of line segments we're looking for.

**rho** and **theta** are the distance and angular resolution of our grid in Hough space. Remember that, in Hough space, we have a grid laid out along the (Θ, ρ) axis. You need to specify **rho** in units of pixels and **theta** in units of radians.

**rho** takes a minimum value of 1, and a reasonable starting place for **theta** is 1 degree (pi/180 in radians). Scale these values up to be more flexible in your definition of what constitutes a line.

The **threshold** parameter specifies the minimum number of votes (intersections in a given grid cell) a candidate line needs to have to make it into the output. The empty **np.array([])** is just a placeholder, no need to change it. **min_line_length** is the minimum length of a line (in pixels) that you will accept in the output, and **max_line_gap** is the maximum distance (again, in pixels) between segments that you will allow to be connected into a single line.


###3. Basics of Jupyter Notebook and Python
see https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python
