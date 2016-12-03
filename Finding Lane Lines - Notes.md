Canny Edge Detection using OpenCV function `Canny`
```python
edges = cv2.Canny(gray, low_threshold, high_threshold)
```
Applying Canny to the image **gray** and your output will be another image called **edges**. **low_threshold** and **high_threshold** are your thresholds for edge detection. The algorithm will first detect strong edge (strong gradient) pixels above the high_threshold, and reject pixels below the low_threshold. Next, pixels with values between the **low_threshold** and **high_threshold** will be included as long as they are connected to strong edges. The output edges is a binary image with white pixels tracing out the detected edges and black everywhere else. See the OpenCV Canny Docs for more details.
