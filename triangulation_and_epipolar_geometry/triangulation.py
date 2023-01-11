import numpy as np
import importlib.util
import sys
import cv2
spec = importlib.util.spec_from_file_location("", "../shared/helper.py")
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

def get_coordinates_3D(m1, m2, P1, P2):
    A = np.zeros((4,4))

    A[0:2,:] = P1[0:2,:]-m1*P1[2,:]
    A[2:4,:] = P2[0:2,:]-m2*P2[2,:]

    # Singular Value Decomposition to retrieve non-trivial solution of lin. hom. equation system
    _, _, VH = np.linalg.svd(A)
    M = VH.transpose()[:,-1][:,np.newaxis]
    M /= M[-1]

    return M   

def main():
    # Read in images
    img1 = cv2.imread('../images/scene1.jpg')
    img2 = cv2.imread('../images/scene2.jpg')

    if PERFORM_CALIBRATION:
        helper.calibrate_book_scenes_example()

    P1, K1, R1, t1 = helper.read_projection_matrix_from_file("../images/data/params1")
    P2, K2, R2, t2 = helper.read_projection_matrix_from_file("../images/data/params2")

    # collect 2 points from each image
    points_img1 = []
    cv2.imshow("image", img1)
    cv2.setMouseCallback('image', collect_points, points_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    points_img2 = []
    cv2.imshow("image", img2)
    cv2.setMouseCallback('image', collect_points, points_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    points_3D = []
    # reconstruct 3D coordinates of every sample point
    for i in np.arange(len(points_img1)):
        M = get_coordinates_3D(points_img1[i], points_img2[i], P1, P2)
        points_3D.append(M)
        print(M)
    if (len(points_3D)==2):
        print("Euclidean distance: " + str(np.sqrt(np.sum(np.power(points_3D[0][0:3]-points_3D[1][0:3], 2)))))

if __name__ == "__main__":
    main()

