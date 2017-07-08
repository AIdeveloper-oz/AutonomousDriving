import pickle, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, pdb
import pickle
cal_img_dir = '../camera_cal'
test_img_dir = '../test_images'



def transform(img, src, dst, mtx, dist, M=None, Minv=None, display=True, save=True):
    # # Convert undistorted image to grayscale
    # if 3==img.shape[-1]:
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Given src and dst points, calculate the perspective transform matrix
    if M is None:
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    h, w = img_undist.shape[:2]
    warped_img = cv2.warpPerspective(img_undist, M, (w,h), flags=cv2.INTER_LINEAR) 
    if save:
        cv2.imwrite('warped_img.jpg',warped_img)
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        trans_pickle = {}
        trans_pickle["M"] = M
        trans_pickle["Minv"] = Minv
        pickle.dump( trans_pickle, open( 'trans_M_Minv.p', 'wb' ) )

    if display:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original', fontsize=20)
        ax2.imshow(img_undist, cmap='gray')
        ax2.set_title('Undistorted', fontsize=20)
        ax3.imshow(warped_img, cmap='gray')
        ax3.set_title('Undistorted and Warped', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    # Return the resulting image and matrix
    return warped_img, M, Minv
