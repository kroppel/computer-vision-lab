import cv2
import numpy as np
import sys
import importlib
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("", "../calibration/calibration_direct_method.py")
calibration_direct_method = importlib.util.module_from_spec(spec)
sys.modules["calibration_direct_method"] = calibration_direct_method
spec.loader.exec_module(calibration_direct_method)

"""Write calibration parameters P, K, R, and t to file indicated by path

Params:
    path (str):     path to file
    P (np.ndarray): projection matrix 3x4 with 8 byte float values
    K (np.ndarray): matrix of internal parameters 3x3 with 8 byte float values
    R (np.ndarray): rotation matrix 3x3 with 8 byte float values
    t (np.ndarray): translation vector 3x1 with 8 byte float values
"""
def write_calibration_parameters_to_file(path, P, K, R, t):
    f = open(path, "wb")
    f.write(P.tobytes(order='C'))
    f.write(K.tobytes(order='C'))
    f.write(R.tobytes(order='C'))
    f.write(t.tobytes(order='C'))
    f.close()

"""Read calibration parameters P, K, R, and t from file indicated by path

Params:
    path (str):     path to file

Returns:
    P (np.ndarray): projection matrix 3x4 with 8 byte float values
    K (np.ndarray): matrix of internal parameters 3x3 with 8 byte float values
    R (np.ndarray): rotation matrix 3x3 with 8 byte float values
    t (np.ndarray): translation vector 3x1 with 8 byte float values
"""
def read_projection_matrix_from_file(path):
    f = open(path, "rb")
    # Note: reshaping order has to match writing order
    P = np.frombuffer(f.read(12*8), dtype=np.float64)[:].reshape((3,4), order='C')
    K = np.frombuffer(f.read(9*8), dtype=np.float64)[:].reshape((3,3), order='C')
    R = np.frombuffer(f.read(9*8), dtype=np.float64)[:].reshape((3,3), order='C')
    t = np.frombuffer(f.read(3*8), dtype=np.float64)[:,np.newaxis]
    f.close()

    return P, K, R, t

"""Mark rectangle region by clicking and holding the left mouse button
and dragging the mouse, similar to desktop behavior

Params:
    event:  Type of mouse event that occurred
    x:      X coordinate of the mouse at the moment of the occured event
    y:      Y coordinate of the mouse at the moment of the occured event
    flags:  - (unused)
    params: List containing the reference to the rectangle bounds and the input image
"""
def mark_rectangle(event, x, y, flags, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        params[0][0] = (int(x),int(y))
    elif event==cv2.EVENT_LBUTTONUP:
        params[0][1] = (int(x),int(y))
        cv2.rectangle(params[1],params[0][0],params[0][1],(0,255,0),2)

"""Let user mark a rectangle on an image and return the rectangle boundaries 
(upper-left and lower-right corners)
"""
def get_marked_rectangle(img):
    rect_bounds = [(0,0), img.shape]
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("", 520, 760)
    cv2.setMouseCallback('image',mark_rectangle, [rect_bounds, img])
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1)
        if(k == ord('q')):
            break
    cv2.destroyAllWindows()

    return rect_bounds

def perform_orb_keypoint_matching(imgL, imgR):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(imgL,None)
    kp2, des2 = orb.detectAndCompute(imgR,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    print("Found {} matches".format(len(matches)))

    return matches, kp1, kp2

"""Filter given keypoint matches by x and y coordinate bounds
"""
def filter_keypoint_matches(matches, kp1, bounds, distance):
    matches_filtered = []
    for match in matches:
        if kp1[match.queryIdx].pt[1] > bounds[0][1] and kp1[match.queryIdx].pt[1] < bounds[1][1] and \
            kp1[match.queryIdx].pt[0] > bounds[0][0] and kp1[match.queryIdx].pt[0] < bounds[1][0] and \
                match.distance < distance:
            matches_filtered.append(match)
    print("{} matches left after filtering".format(len(matches_filtered)))

    return matches_filtered

"""Perform the direct calibration method for the two example image scenes found in the data directory
"""
def calibrate_book_scenes_example():
    # Read in images
    img1 = cv2.imread('../images/scene1.jpg')
    img2 = cv2.imread('../images/scene2.jpg')

    # real-world 3D-calibration points
    M = np.asarray([np.asarray([0,0,0,1], dtype=float),
                    np.asarray([0,0,4,1], dtype=float),
                    np.asarray([0,23,4,1], dtype=float),
                    np.asarray([14.5,23,4,1], dtype=float),
                    np.asarray([14.5,0,4,1], dtype=float),
                    np.asarray([14.5,0,0,1], dtype=float)])

    # collect calibration points for both images from user
    points1 = []
    cv2.imshow("Calibration Scene 1", img1)
    cv2.setMouseCallback('Calibration Scene 1', calibration_direct_method.collect_calibration_points, points1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2D projections
    m1 = np.asarray([np.asarray([points1[0][0], points1[0][1], 1]),
                    np.asarray([points1[1][0], points1[1][1], 1]),
                    np.asarray([points1[2][0], points1[2][1], 1]),
                    np.asarray([points1[3][0], points1[3][1], 1]),
                    np.asarray([points1[4][0], points1[4][1], 1]),
                    np.asarray([points1[5][0], points1[5][1], 1])])

    points2 = []
    cv2.imshow("Calibration Scene 2", img2)
    cv2.setMouseCallback('Calibration Scene 2', calibration_direct_method.collect_calibration_points, points2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2D projections
    m2 = np.asarray([np.asarray([points2[0][0], points2[0][1], 1]),
                    np.asarray([points2[1][0], points2[1][1], 1]),
                    np.asarray([points2[2][0], points2[2][1], 1]),
                    np.asarray([points2[3][0], points2[3][1], 1]),
                    np.asarray([points2[4][0], points2[4][1], 1]),
                    np.asarray([points2[5][0], points2[5][1], 1])])

    P1, K1, R1, t1 = calibration_direct_method.calibration_direct_method(M, m1)
    P2, K2, R2, t2 = calibration_direct_method.calibration_direct_method(M, m2)

    # store projection matrices in files
    write_calibration_parameters_to_file("../images/data/params1", P1, K1, R1, t1)
    write_calibration_parameters_to_file("../images/data/params2", P2, K2, R2, t2)

"""Perform the direct calibration method for the two new example image scenes found in the data directory
"""
def calibrate_new_book_scenes_example():
    # Read in images
    img1 = cv2.imread('../images/new_scene1.jpg')
    img2 = cv2.imread('../images/new_scene2.jpg')

    # real-world 3D-calibration points
    M = np.asarray([np.asarray([0,0,0,1], dtype=float),
                    np.asarray([0,0,4,1], dtype=float),
                    np.asarray([0,23,4,1], dtype=float),
                    np.asarray([14.5,23,4,1], dtype=float),
                    np.asarray([14.5,0,4,1], dtype=float),
                    np.asarray([14.5,0,0,1], dtype=float)])

    # collect calibration points for both images from user
    points1 = []
    cv2.imshow("Calibration Scene 1", img1)
    cv2.setMouseCallback('Calibration Scene 1', calibration_direct_method.collect_calibration_points, points1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2D projections
    m1 = np.asarray([np.asarray([points1[0][0], points1[0][1], 1]),
                    np.asarray([points1[1][0], points1[1][1], 1]),
                    np.asarray([points1[2][0], points1[2][1], 1]),
                    np.asarray([points1[3][0], points1[3][1], 1]),
                    np.asarray([points1[4][0], points1[4][1], 1]),
                    np.asarray([points1[5][0], points1[5][1], 1])])

    points2 = []
    cv2.imshow("Calibration Scene 2", img2)
    cv2.setMouseCallback('Calibration Scene 2', calibration_direct_method.collect_calibration_points, points2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2D projections
    m2 = np.asarray([np.asarray([points2[0][0], points2[0][1], 1]),
                    np.asarray([points2[1][0], points2[1][1], 1]),
                    np.asarray([points2[2][0], points2[2][1], 1]),
                    np.asarray([points2[3][0], points2[3][1], 1]),
                    np.asarray([points2[4][0], points2[4][1], 1]),
                    np.asarray([points2[5][0], points2[5][1], 1])])

    P1, K1, R1, t1 = calibration_direct_method.calibration_direct_method(M, m1)
    P2, K2, R2, t2 = calibration_direct_method.calibration_direct_method(M, m2)

    # store projection matrices in files
    write_calibration_parameters_to_file("../images/data/new_params1", P1, K1, R1, t1)
    write_calibration_parameters_to_file("../images/data/new_params2", P2, K2, R2, t2)
