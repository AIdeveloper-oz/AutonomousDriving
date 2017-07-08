import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob, os, pdb
import cv2

# Read in a thresholded image
img = mpimg.imread('warped_img.jpg')

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(img, window_width, window_height, margin, old_win_cent=None):    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(img[int(3*img.shape[0]/4):,:int(img.shape[1]/2)], axis=0)
    l_conv_signal = np.convolve(window,l_sum)
    r_sum = np.sum(img[int(3*img.shape[0]/4):,int(img.shape[1]/2):], axis=0)
    r_conv_signal = np.convolve(window,r_sum)

    l_center = np.argmax(l_conv_signal)-window_width/2
    r_center = np.argmax(r_conv_signal)-window_width/2+int(img.shape[1]/2)
    # if old_win_cent is None:
    #     l_center = np.argmax(l_conv_signal)-window_width/2
    #     r_center = np.argmax(r_conv_signal)-window_width/2+int(img.shape[1]/2)
    # else:
    #     old_l_center, old_r_center = window_centroids[0]
    #     offset = window_width/2
    #     # Finde left
    #     l_min_index = int(max(old_l_center+offset-margin,0))
    #     l_max_index = int(min(old_l_center+offset+margin,img.shape[1]))
    #     l_center = np.argmax(l_conv_signal[l_min_index:l_max_index])+l_min_index-offset
    #     # Finde right
    #     r_min_index = int(max(old_r_center+offset-margin,0))
    #     r_max_index = int(min(old_r_center+offset+margin,img.shape[1]))
    #     r_center = np.argmax(r_conv_signal[r_min_index:r_max_index])+r_min_index-offset
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(img.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(img[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,img.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,img.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids


def measure_curve(leftx, rightx, ploty, display):
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


def sliding_win_search(img, old_left_fit=None, old_right_fit=None, display=False, save=True):
    ## Window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    ## If we obtain confident lane from previous frame, we use it to guide current frame
    if (old_left_fit is not None) and (old_right_fit is not None):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (old_left_fit[0]*(nonzeroy**2) + old_left_fit[1]*nonzeroy + old_left_fit[2] - margin)) & (nonzerox < (old_left_fit[0]*(nonzeroy**2) + old_left_fit[1]*nonzeroy + old_left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (old_right_fit[0]*(nonzeroy**2) + old_right_fit[1]*nonzeroy + old_right_fit[2] - margin)) & (nonzerox < (old_right_fit[0]*(nonzeroy**2) + old_right_fit[1]*nonzeroy + old_right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Calculate culvature
        plot_fity = np.linspace(0, img.shape[0]-1, num=img.shape[0])
        left_fitx = left_fit[0]*plot_fity**2 + left_fit[1]*plot_fity + left_fit[2]
        right_fitx = right_fit[0]*plot_fity**2 + right_fit[1]*plot_fity + right_fit[2]   
        left_curverad, right_curverad = measure_curve(left_fitx, right_fitx, plot_fity, display)
        print(left_curverad, 'm', right_curverad, 'm')
        # # Generate x and y values for plotting
        # ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    else:
        ## Restart window searching
        window_centroids = find_window_centroids(img, window_width, window_height, margin)
        win_cent_array = np.array(window_centroids)
        ## Fit a second order polynomial to left/right line
        if len(window_centroids) > 0:
            leftx, rightx = win_cent_array[:,0], win_cent_array[:,1]
            leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
            rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
            ploty = np.linspace(0, img.shape[0]-1, num=int(img.shape[0]/window_height)) + int(window_height/2)
            left_fit = np.polyfit(ploty, leftx, 2)    
            right_fit = np.polyfit(ploty, rightx, 2)
            ## Curvature check
            left_curverad, right_curverad = measure_curve(leftx, rightx, ploty, display)
            print(left_curverad, 'm', right_curverad, 'm')
            # Example values: 632.1 m    626.2 m

        # If we found any window centers
        left_fitx, right_fitx, plot_fity = None, None, None
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(img)
            r_points = np.zeros_like(img)

            # Go through each level and draw the windows    
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width,window_height,img,window_centroids[level][0],level)
                r_mask = window_mask(window_width,window_height,img,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            ## Draw the window search result
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage = np.array(cv2.merge((img,img,img)),np.uint8) # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 0.5, template, 0.5, 0.0) # overlay the orignal road image with window results
        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((img,img,img)),np.uint8)

    ## Draw the fit result
    plot_fity = np.linspace(0, img.shape[0]-1, num=img.shape[0])
    left_fitx = left_fit[0]*plot_fity**2 + left_fit[1]*plot_fity + left_fit[2]
    right_fitx = right_fit[0]*plot_fity**2 + right_fit[1]*plot_fity + right_fit[2]   

    plt.plot(left_fitx, plot_fity, color='red', linewidth=3)
    plt.plot(right_fitx, plot_fity, color='red', linewidth=3)

    if save:
        plt.savefig('sliding_win_polyfit.png')

    if display:
        # Display the final results
        plt.imshow(output)
        plt.title('window fitting results')
        plt.show()

    return left_curverad, right_curverad, left_fitx, right_fitx, plot_fity, left_fit, right_fit