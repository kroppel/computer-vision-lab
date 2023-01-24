import cv2
import numpy as np
import os

PATH_TO_IMAGES = 'calib_example/'
images = []

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((12*13, 3), np.float32)
objp[:,:2] = np.mgrid[0:13,0:12].transpose().reshape(-1,2)

# lists to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for file_name in os.listdir(PATH_TO_IMAGES):
    images.append(cv2.imread(PATH_TO_IMAGES+file_name))
    gray = cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (12,13), None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        for corner in corners:
            cv2.circle(images[-1], (int(corner[0,0]), int(corner[0,1])), 2, color=(0,255,0), thickness=-1)
        cv2.imshow("image", images[-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No corners found")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], np.zeros((3,3)), np.zeros((4)))

if ret:
    print(mtx)
    print(dist)
    print(rvecs)
    print(tvecs)

else:
    print("Estimation failed")

