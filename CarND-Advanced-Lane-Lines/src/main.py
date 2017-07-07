import numpy as np
import cv2
import glob
import os, pdb
import matplotlib.pyplot as plt
import pickle

from calibration import get_calibration
from transform import transform
from threshold import threshold

cal_img_dir = '../camera_cal'
test_img_dir = '../test_images'
# out_dir =  '../output_images'
nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y
DISPLAY = True

######################################################################
### Camera calibration (camera matrix and distortion coefficients) ###
# mtx, dist = get_calibration(cal_img_dir, test_img_dir, nx, ny, display=False)


######################################################################
### Use different thresholds to get mask ###
img_path = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))[0]
img_BGR = cv2.imread(img_path)
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

lane_binary = threshold(img_RGB, display=DISPLAY)


######################################################################
### Undistortion and perspective transform ###
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# img_path = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))[0]
# img_BGR = cv2.imread(img_path)
# img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

## corners for perspective transform choosen mannually (x,y)
lane_RGB= cv2.cvtColor(lane_binary*255, cv2.COLOR_GRAY2RGB)
img_H, img_W = img_RGB.shape[:2]
trans_src = np.float32([[225,697], [1078,697], [673,443], [603,443]])
offset = 330
trans_dst = np.float32([[img_W/2-offset,img_H], [img_W/2+offset,img_H], [img_W/2+offset,0], [img_W/2-offset,0]])
_, M = transform(lane_RGB, trans_src, trans_dst, mtx, dist, display=DISPLAY)

