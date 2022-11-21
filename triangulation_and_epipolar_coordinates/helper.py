import cv2
import numpy as np
import sys
import importlib

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

def calibrate_book_scenes_example():
    # Read in images
    img1 = cv2.imread('data/scene1.jpg')
    img2 = cv2.imread('data/scene2.jpg')

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
    write_calibration_parameters_to_file("data/params1", P1, K1, R1, t1)
    write_calibration_parameters_to_file("data/params2", P2, K2, R2, t2)