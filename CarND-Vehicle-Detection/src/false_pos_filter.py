import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
import os, pdb
from utils import draw_boxes

# # Read in a pickle file with bboxes saved
# # Each item in the "all_bboxes" list will contain a 
# # list of boxes for one of the images shown above
# box_list = pickle.load( open( "bbox_pickle.p", "rb" ))

# # Read in image similar to one shown above 
# image = mpimg.imread('test_image.jpg')
# image = cv2.cvtColor(cv2.imread('test_image.jpg'), cv2.COLOR_BGR2RGB)
# heat = np.zeros_like(image[:,:,0]).astype(np.float)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    bbox_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        ## Sanity Check
        # bbox_w = bbox[1][0] - bbox[0][0]
        # bbox_h = bbox[1][1] - bbox[0][1]
        # ratio = float(bbox_w)/float(bbox_h)
        # if ratio<0.4 or ratio >2.5:
        #     continue
        bbox_list.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img, bbox_list

def false_pos_filter(img, bbox_list, threshold=1, heat_list=None, smooth=10, save=False):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,bbox_list)
        
    # Apply threshold to help remove false positives
    if heat_list is not None:
        while len(heat_list)<smooth-1:
            heat_list.insert(0, np.zeros_like(heat))
        if len(heat_list)>=smooth:
            heat_list.pop(0)
        heat_list.append(heat)
        heat = np.array(heat_list).mean(axis=0)
    heat = apply_threshold(heat,threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    after_img, filted_bbox_list = draw_labeled_bboxes(np.copy(img), labels)
    # pdb.set_trace()
    if save:
        # fig = plt.figure()
        plt.subplot(221)
        before_img = draw_boxes(img, bbox_list, color=(0, 0, 255), thick=6)
        plt.imshow(before_img)
        plt.title('Before filtering')
        plt.subplot(222)
        plt.imshow(labels[0], cmap='gray')
        plt.title('Labels')
        plt.subplot(223)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.tight_layout()
        plt.subplot(224)
        plt.imshow(after_img)
        plt.title('After filtering')
        plt.savefig('false_pos_filter.jpg', bbox_inches='tight', dpi=400)

    return after_img, filted_bbox_list, heat_list
