import numpy as np
import cv2
import glob
import os, pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip

from calibration import get_calibration
from transform import transform
from threshold import threshold
from sliding_win_search import sliding_win_search


######################### For writeup #########################
def pipeline_writeup(display=True, save=True):
    ######################################################################
    ### Camera calibration (camera matrix and distortion coefficients) ###
    cal_img_dir = '../camera_cal'
    test_img_dir = '../test_images'
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y
    mtx, dist = get_calibration(cal_img_dir, test_img_dir, nx, ny, display=display, save=save)
    # dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
    # mtx = dist_pickle["mtx"]
    # dist = dist_pickle["dist"]

    ######################################################################
    ### Read test image ###
    # img_path = sorted(glob.glob(os.path.join(test_img_dir, '*.png')))[3]
    # img_RGB = (mpimg.imread(img_path)*255).astype(np.uint8)
    img_path = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))[0]
    img_RGB = mpimg.imread(img_path)
    if display:
        plt.imshow(img_RGB)
        plt.show()

    ######################################################################
    ### Use different thresholds ###
    lane_binary, lane_gray = threshold(img_RGB, display=display, save=save)

    ######################################################################
    ### Undistortion and perspective transform ###
    ## corners for perspective transform choosen mannually (x,y)
    img_H, img_W = img_RGB.shape[:2]
    trans_src = np.float32([[225,697], [1078,697], [705,460], [576,460]])
    offset = 360
    trans_dst = np.float32([[img_W/2-offset,img_H], [img_W/2+offset,img_H], [img_W/2+offset,0], [img_W/2-offset,0]])

    # trans_src = np.float32([[585,460], [203,720], [1127,720], [695,460]])
    # trans_dst = np.float32([[320,0], [320,720], [960,720], [960,0]])
    warped_img, M, Minv = transform(lane_gray, trans_src, trans_dst, mtx, dist, display=display, save=save)

    ######################################################################
    ### Undistortion and perspective transform ###
    _, _, _, _, _,_,_ = sliding_win_search(warped_img, None, None, display=display, save=save)

## test
pipeline_writeup(save=False)

######################### For video #########################
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
trans_pickle = pickle.load( open( "trans_M_Minv.p", "rb" ) )
M = trans_pickle["M"]
Minv = trans_pickle["Minv"]

left_fitx_list = []
right_fitx_list = []
curve_diff_thx = 400
curve_min = 500
avg_frame_num = 3
old_left_fit = None
old_right_fit = None
def pipeline_video(img_RGB):
    global old_left_fit
    global old_right_fit
    display=False
    save=False
    _, lane_gray = threshold(img_RGB, display=display, save=save)
    warped_img, _, _ = transform(lane_gray, None, None, mtx, dist, M=M, Minv=Minv, display=display, save=save)
    left_cr, right_cr, left_fitx, right_fitx, plot_fity, left_fit, right_fit = sliding_win_search(warped_img, old_left_fit, old_right_fit, display=display, save=save)

    ## Sanity Check
    if abs(left_cr-right_cr) < curve_diff_thx:
        if left_cr>curve_min and right_cr>curve_min:
            if len(left_fitx_list)>avg_frame_num:
                left_fitx_list.pop(0)
                right_fitx_list.pop(0)
            left_fitx_list.append(left_fitx)
            right_fitx_list.append(right_fitx)
            old_left_fit = left_fit
            old_right_fit = right_fit

    left_fitx = np.mean(np.array(left_fitx_list), axis=0)
    left_fitx = (left_fitx + left_fitx_list[-1])/2
    right_fitx = np.mean(np.array(right_fitx_list), axis=0)
    right_fitx = (right_fitx + right_fitx_list[-1])/2

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, plot_fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_RGB.shape[1], img_RGB.shape[0])) 
    # Combine the result with the original image
    undist = cv2.undistort(img_RGB, mtx, dist, None, mtx)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)



    curvature = "Estimated lane curvature %.2fm" % ((left_cr+right_cr)/2)
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    dist_centre = "Estimated offset from lane center %.2fm" % (abs(left_fitx[-1]+right_fitx[-1]-img_RGB.shape[1])/2*xm_per_pix)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result, curvature, (30, 60), font, 1, (255,255,255), 2)
    cv2.putText(result, dist_centre, (30, 90), font, 1, (255,255,255), 2)

    return result
    # plt.imshow(result)
    # plt.show()




# white_output = 'result.mp4'
# clip1 = VideoFileClip('../project_video.mp4')
# white_clip = clip1.fl_image(pipeline_video) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)