import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import zipfile
import glob
import os, pdb

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Extract data from compressed files to /tmp and split into train/test set
# Note: train/test set cannot be split randomly, since there are multiple
# images for one car
def prepare_data(data_dir, train_ratio=0.9):
    # Extract data
    fantasy_zip = zipfile.ZipFile(os.path.join(data_dir,'vehicles.zip'))
    fantasy_zip.extractall('/tmp') 
    fantasy_zip.close()
    fantasy_zip = zipfile.ZipFile(os.path.join(data_dir,'non-vehicles.zip'))
    fantasy_zip.extractall('/tmp') 
    fantasy_zip.close()

    # Read in cars and notcars
    cars_train = []
    cars_test = []
    notcars_train = []
    notcars_test = []
    cars_dir = '/tmp/vehicles'
    notcars_dir = '/tmp/non-vehicles'
    for sub_dir in sorted(os.listdir(cars_dir)):
        files_tmp = glob.glob(os.path.join(cars_dir,sub_dir,'*.png'))
        end = int(len(files_tmp)*train_ratio)
        cars_train.extend(files_tmp[:end])
        cars_test.extend(files_tmp[end:])
    for sub_dir in sorted(os.listdir(notcars_dir)):
        files_tmp = glob.glob(os.path.join(notcars_dir,sub_dir,'*.png'))
        end = int(len(files_tmp)*train_ratio)
        notcars_train.extend(files_tmp[:end])
        notcars_test.extend(files_tmp[end:])

    return cars_train, cars_test, notcars_train, notcars_test

