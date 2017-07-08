import numpy as np
import cv2
import glob
import os, pdb
import matplotlib.pyplot as plt
# %matplotlib qt
import pickle

def get_calibration(cal_img_dir, test_img_dir, nx, ny, display=False, save=True):
    #############################################################################
    #################### Detected corners on chessboard #########################
    #############################################################################
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(ny,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(os.path.join(cal_img_dir, '*.jpg'))

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img_BGR = cv2.imread(fname)
        gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if display:
                # Draw and display the corners
                cv2.drawChessboardCorners(img_BGR, (nx,ny), corners, ret)
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img_BGR)
                cv2.imshow('img', img_BGR)
                cv2.waitKey(200)
    cv2.destroyAllWindows()

    #############################################################################
    ## Calculate CameraMatrix and DistortionCoefficient with detected corners ###
    #############################################################################
    # Do camera calibration given object points and image points
    img_size = (img_BGR.shape[1], img_BGR.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    if save:
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( 'wide_dist_pickle.p', 'wb' ) )

    # Test undistortion on an image
    if display:
        #
        img_path = sorted(glob.glob(os.path.join(cal_img_dir, '*.jpg')))[0]
        img_BGR = cv2.imread(img_path)
        dst = cv2.undistort(img_BGR, mtx, dist, None, mtx)
        cv2.imwrite('calibration1_undist.jpg',dst)
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.show()
        #
        img_path = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))[0]
        img_BGR = cv2.imread(img_path)
        dst = cv2.undistort(img_BGR, mtx, dist, None, mtx)
        cv2.imwrite('test1_undist.jpg',dst)
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.show()

    # Return camera matrix and distortion coefficients
    return mtx, dist
