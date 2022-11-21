import numpy as np
import importlib.util
import sys
import cv2
spec = importlib.util.spec_from_file_location("", "../calibration/calibration_direct_method.py")
calibration_direct_method = importlib.util.module_from_spec(spec)
sys.modules["calibration_direct_method"] = calibration_direct_method
spec.loader.exec_module(calibration_direct_method)
spec = importlib.util.spec_from_file_location("", "helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)

PERFORM_CALIBRATION = False

"""Collect point on right mouse click and add it to a list
"""
def collect_points(event, x, y, flags, points):
    # collect point on rightclick
    if event==cv2.EVENT_RBUTTONDOWN:
        print((x, y))
        points.append(np.asarray([x, y])[:,np.newaxis])

"""Draw the epipolar line of point m from first scene onto image showing the second scene

Params:
    img (np.ndarray:    image to draw onto
    P1 (np.ndarray):    perspective matrix of first scene
    P2 (np.ndarray):    perspective matrix of second scene
    m (np.ndarray):     reference point of first scene
"""
def draw_epipolar_line(img, P1, P2, m):
    e_prime = -(P2[0:3,0:3].dot(np.linalg.inv(P1[0:3,0:3]).dot(P1[:,-1][:,np.newaxis]))) + P2[:,-1][:,np.newaxis]
    e_prime /= e_prime[-1]
    p_inf = P2[0:3,0:3].dot(np.linalg.inv(P1[0:3,0:3]).dot(m))
    p_inf /= p_inf[-1]

    # Draw epipolar line by calculating the line equation and the points on the image borders
    F = calibration_direct_method.cross_product_matrix(e_prime).dot(P2[0:3,0:3].dot(np.linalg.inv(P1[0:3,0:3])))
    epipolar_line_parameters = F.dot(m)
    epipolar_line_explicit = lambda x: (-epipolar_line_parameters[0]*x - epipolar_line_parameters[2])/epipolar_line_parameters[1]

    # calculate points of the epipolar line at the image boundaries
    p1 = (0, epipolar_line_explicit(0))
    p2 = (img.shape[1]-1, epipolar_line_explicit(img.shape[1]-1))

    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color=(0,0,255), thickness=2)

def main():
    # Read in images
    img1 = cv2.imread('data/scene1.jpg')
    img2 = cv2.imread('data/scene2.jpg')

    if PERFORM_CALIBRATION:
        helper.calibrate_book_scenes_example()
        
    P1, K1, R1, t1 = helper.read_projection_matrix_from_file("data/params1")
    P2, K2, R2, t2 = helper.read_projection_matrix_from_file("data/params2")

    # Attention: Calibration parameters are valid only for original images
    # -> points taken from the resized images have to be transformed back
    # to corresponding coordinates of the original sized images
    # -> lines have to be drawn on the original image, which then are again resized
    img1_resized = cv2.resize(img1, (int(img1.shape[1]/2), int(img1.shape[0]/2)))
    img2_resized = cv2.resize(img2, (int(img2.shape[1]/2), int(img2.shape[0]/2)))

    img_epipolar = np.concatenate((img1_resized,img2_resized), axis=1)

    points_epipolar = []
    cv2.imshow("epipolar line", img_epipolar)
    cv2.setMouseCallback('epipolar line', collect_points, points_epipolar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for point in points_epipolar:
        # transformation of point into original coordinates (multiply each coordinate with scaling factor)
        point_orig = point * 2
        draw_epipolar_line(img2, P1, P2, np.concatenate((point_orig, np.ones((1,1))), axis=0))
        cv2.circle(img1_resized, (point[0], point[1]), 3, color=(0,0,255), thickness=-1)
        img_epipolar = np.concatenate((img1_resized,cv2.resize(img2, (int(img2.shape[1]/2), int(img2.shape[0]/2)))), axis=1)

    
    cv2.imshow("epipolar line", img_epipolar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()