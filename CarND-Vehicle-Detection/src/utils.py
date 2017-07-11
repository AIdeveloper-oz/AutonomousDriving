import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
import zipfile
import glob
import os, pdb


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
def prepare_data(data_dir, train_ratio=0.9, save=True):
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

    if save:
        # fig = plt.figure()
        plt.subplot(141)
        plt.imshow(mpimg.imread(cars_train[0]))
        plt.title('Car train')
        plt.subplot(142)
        plt.imshow(mpimg.imread(cars_test[0]))
        plt.title('Car test')
        plt.subplot(143)
        plt.imshow(mpimg.imread(notcars_train[0]))
        plt.title('notCar train')
        plt.subplot(144)
        plt.imshow(mpimg.imread(notcars_test[0]))
        plt.title('notCar test')
        plt.tight_layout()
        plt.savefig('car_not_car.jpg', bbox_inches='tight', dpi=400)
    return cars_train, cars_test, notcars_train, notcars_test

