import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, pdb
from sklearn.preprocessing import StandardScaler
import scipy.misc as misc
import time
import pickle, glob
from moviepy.editor import VideoFileClip

from feature import extract_features, single_img_features, get_hog_features, color_convert_from_RGB, sample_hog_vis
from utils import prepare_data, draw_boxes
from classification import get_classifer
from fast_multiscale_search import multiscale_search
from false_pos_filter import false_pos_filter


########################################################################
############################# Training #################################

# ################# Prepare data
# print('################# Prepare data ###################')
# data_dir = '../data'
# cars_train, cars_test, notcars_train, notcars_test = prepare_data(data_dir, train_ratio=0.9)


# ################ Extract feature
# print('################ Extract feature #################')
# ## Parameters
# param = {
#     'color_space' : 'HSV', # Can be RGB, HSV, (LUV, HLS, YUV, YCrCb leads to Nan for PNG image)
#     'orient' : 11,  # HOG orientations
#     'pix_per_cell' : 8, # HOG pixels per cell
#     'cell_per_block' : 2, # HOG cells per block
#     'hog_channel' : "ALL", # Can be 0, 1, 2, or "ALL"
#     'spatial_size' : (16, 16), # Spatial binning dimensions
#     'hist_bins' : 32,    # Number of histogram bins
#     'spatial_feat' : True, # Spatial features on or off
#     'hist_feat' : True, # Histogram features on or off
#     'hog_feat' : True, # HOG features on or off
# }

# ## Hog visualization
# sample_hog_vis(cars_train[0], notcars_train[0], param)

# ## Extract features
# cars_train_fea = extract_features(cars_train, param, data_aug=True)
# cars_test_fea = extract_features(cars_test, param, data_aug=True)
# notcars_train_fea = extract_features(notcars_train, param, data_aug=True)
# notcars_test_fea = extract_features(notcars_test, param, data_aug=True)
# #
# x_train = np.vstack((cars_train_fea, notcars_train_fea)).astype(np.float64)  
# x_test = np.vstack((cars_test_fea, notcars_test_fea)).astype(np.float64)             
# # Fit a per-column scaler
# x_scaler = StandardScaler().fit(x_train)
# scaled_x_train = x_scaler.transform(x_train)
# scaled_x_test = x_scaler.transform(x_test)
# # Define the labels vector
# y_train = np.hstack((np.ones(len(cars_train_fea)), np.zeros(len(notcars_train_fea))))
# y_test = np.hstack((np.ones(len(cars_test_fea)), np.zeros(len(notcars_test_fea))))


# ################# Classification
# print('################ Classification #################')
# svc = get_classifer(scaled_x_train, scaled_x_test, y_train, y_test)


# ################# Save
# model_param_pickle = {'svc':svc, 'x_scaler':x_scaler, 'param':param}
# pickle.dump( model_param_pickle, open( "model_param_pickle.p", "wb" ))


########################################################################
############################# Testing ##################################

model_param_pickle = pickle.load( open( "model_param_pickle.p", "rb" ) )
svc = model_param_pickle["svc"]
x_scaler = model_param_pickle["x_scaler"]
param = model_param_pickle["param"]

## For writeup
img_path = sorted(glob.glob('../test_images/*.jpg'))[0]
img_RGB = mpimg.imread(img_path)
# if img_path.endswith('png'):
#     img_RGB = img_RGB.astype(np.float32)*255
# img_RGB = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
bbox_list = multiscale_search(img_RGB, svc, x_scaler, param)
false_pos_filter(img_RGB, bbox_list, save=True)
pdb.set_trace()


heat_list = []

def pipeline_video(img):
    global heat_list
    bbox_list = multiscale_search(img, svc, x_scaler, param)
    after_img, _, heat_list =false_pos_filter(img, bbox_list, threshold=1.5, heat_list=heat_list, smooth=6, save=False)
    return after_img


# white_output = 'result.mp4'
# clip1 = VideoFileClip('../test_video.mp4')
# white_clip = clip1.fl_image(pipeline_video) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

white_output = 'project_video_result.mp4'
clip1 = VideoFileClip('../project_video.mp4')
white_clip = clip1.fl_image(pipeline_video) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)