import numpy as np
import cv2
import os, pdb
import matplotlib.pyplot as plt

# Define a function that takes an image, gradient orientation, and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for s,v value of HSV
def color_threshold(img, thresh_s=(170, 225), thresh_v=(170, 225)):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_ch = hls[:,:,2] ## the s channel range of hls is different from that of hsv
    # Threshold color channel
    binary_output = np.zeros_like(s_ch)
    binary_output[(s_ch>=thresh_s[0]) & (s_ch<=thresh_s[1])] = 1

    # Return the binary image
    return binary_output

#region of interest
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img, dtype=np.uint8)
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# def thresholding(img, grad_thx_min =211, grad_thx_max =255,grad_thy_min =0, grad_thy_max = 25, mag_th_min = 150,mag_th_max = 255, dir_th_min  = 0.7, dir_th_max = 1.3, s_threshold_min = 113, s_threshold_max = 255, v_threshold_min = 234, v_threshold_max = 255,  k_size = 15, adp_thr = 250):
def threshold(img_RGB, 
              threshold_dic={'grad_x':(20, 100), 
                             'grad_y':(20, 100), 
                             'mag':(52, 107), 
                             'dir':(0.7, 1.3),
                             'hls_s':(170, 255)}, 
              ksize=15, # Choose a larger odd number to smooth gradient measurements
              display=False):

    ## Apply each of the thresholding functions
    grad_x = abs_sobel_thresh(img_RGB, 'x', ksize, threshold_dic['grad_x'])
    # grad_y = abs_sobel_thresh(img_RGB, 'y', ksize, threshold_dic['grad_y'])
    # mag_binary = mag_thresh(img_RGB, ksize, threshold_dic['mag'])
    # dir_binary = dir_threshold(img_RGB, ksize, threshold_dic['dir'])
    color_binary = color_threshold(img_RGB, threshold_dic['hls_s'])

    combine = np.zeros_like(color_binary)
    combine[(grad_x==1) | color_binary==1] = 1

    ## Post process
    imshape = img_RGB.shape
    vertices = np.array([[(.55*imshape[1], 0.6*imshape[0]), (imshape[1],imshape[0]),
                        (0,imshape[0]),(.45*imshape[1], 0.6*imshape[0])]], dtype=np.int32)
    combine_masked = region_of_interest(combine.astype(np.uint8), vertices)
    kernel = np.ones((4,4),np.uint8)
    combine_closed = cv2.morphologyEx(combine_masked, cv2.MORPH_CLOSE, kernel)
    combine_final = cv2.morphologyEx(combine_closed, cv2.MORPH_OPEN, kernel)

    cv2.imwrite('combine_masked.jpg',combine_masked)
    cv2.imwrite('combine_final.jpg',combine_final)

    if display:
        ax = plt.subplot(2,2,1)
        plt.imshow(grad_x, cmap='gray')
        ax.set_title('grad_x')
        ax = plt.subplot(2,2,2)
        plt.imshow(color_binary, cmap='gray')
        ax.set_title('color_binary')
        ax = plt.subplot(2,2,3)
        plt.imshow(combine_masked, cmap='gray')
        ax.set_title('combine_masked')
        ax = plt.subplot(2,2,4)
        plt.imshow(combine_final, cmap='gray')
        ax.set_title('combine_final')
        plt.show()

    return combine_final


# def threshold_interative(img_RGB, 
#                             grad_x_min =211, 
#                             grad_x_max =255,
#                             grad_y_min =0, 
#                             grad_y_max = 25, 
#                             mag_min = 150,
#                             mag_max = 255, 
#                             dir_min  = 0.7, 
#                             dir_max = 1.3, 
#                             s_min = 113, 
#                             s_max = 255, 
#                             k_size = 15):
#     combine = threshold(img_RGB, 
#                         threshold_dic={'grad_x':(20, 100), 
#                                      'grad_y':(20, 100), 
#                                      'mag':(30, 100), 
#                                      'dir':(0.7, 1.3), 
#                                      'hls_s':(170, 225)}, 
#                         ksize=ksize,
#                         display=False)
#     plt.imshow(combine, cmap = "gray")

# from threshold import threshold_interative
# from ipywidgets import interactive, interact, fixed
# interact (threshold_interative, 
#             img_RGB=fixed(img_RGB), 
#             grad_x_min =(0,255), 
#             grad_x_max =(0,255), 
#             grad_y_min =(0,255), 
#             grad_y_max =(0,255), 
#             mag_min =(0,255), 
#             mag_max =(0,255), 
#             dir_min = (0,np.pi/2.0,0.1), 
#             dir_max = (0,np.pi/2.0,0.1), 
#             s_min =(0,255), 
#             s_max =(0,255),
#             k_size = (1,31,2))