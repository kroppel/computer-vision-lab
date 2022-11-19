import numpy as np
import importlib.util
import sys
import cv2
spec = importlib.util.spec_from_file_location("", "../calibration/calibration_direct_method.py")
calibration_direct_method = importlib.util.module_from_spec(spec)
sys.modules["calibration_direct_method"] = calibration_direct_method
spec.loader.exec_module(calibration_direct_method)

LOAD_PROJECTION_MATRICES_FROM_FILE = True

def collect_points(event, x, y, flags, points):
    # collect point on rightclick
    if event==cv2.EVENT_RBUTTONDOWN:
        print((x, y))
        points.append(np.asarray([x, y])[:,np.newaxis])

def write_calibration_parameters_to_file(path, P, K, R, t):
    f = open(path, "wb")
    f.write(P.tobytes(order='C'))
    f.write(K.tobytes(order='C'))
    f.write(R.tobytes(order='C'))
    f.write(t.tobytes(order='C'))
    f.close()

def read_projection_matrix_from_file(path):
    f = open(path, "rb")
    # Note: reshaping order has to match writing order
    P = np.frombuffer(f.read(12*8), dtype=np.float64)[:].reshape((3,4), order='C')
    K = np.frombuffer(f.read(9*8), dtype=np.float64)[:].reshape((3,3), order='C')
    R = np.frombuffer(f.read(9*8), dtype=np.float64)[:].reshape((3,3), order='C')
    t = np.frombuffer(f.read(3*8), dtype=np.float64)[:,np.newaxis]
    f.close()

    return P, K, R, t

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
    img1 = cv2.imread('scene1.jpg')
    img2 = cv2.imread('scene2.jpg')

    if LOAD_PROJECTION_MATRICES_FROM_FILE:
        P1, K1, R1, t1 = read_projection_matrix_from_file("params1")
        P2, K2, R2, t2 = read_projection_matrix_from_file("params2")

    else:
        # real-world 3D-calibration points
        M = np.asarray([np.asarray([0,0,0,1], dtype=float),
                        np.asarray([0,0,4,1], dtype=float),
                        np.asarray([0,23,4,1], dtype=float),
                        np.asarray([14.5,23,4,1], dtype=float),
                        np.asarray([14.5,0,4,1], dtype=float),
                        np.asarray([14.5,0,0,1], dtype=float)])

        # collect calibration points for both images from user
        points1 = []
        cv2.imshow("image", img1)
        cv2.setMouseCallback('image', calibration_direct_method.collect_calibration_points, points1)
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
        cv2.imshow("image", img2)
        cv2.setMouseCallback('image', calibration_direct_method.collect_calibration_points, points2)
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
        write_calibration_parameters_to_file("params1", P1, K1, R1, t1)
        write_calibration_parameters_to_file("params2", P2, K2, R2, t2)

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
        print("Euclidean distance: " + str(np.linalg.norm([points_3D[0][0:3],points_3D[1][0:3]])))

    # Animation along book edges
    points_animation = []
    for i in np.arange(0, 14.5, 0.1):
        p = P1.dot(np.asarray([i,0,0,1])[:np.newaxis])
        points_animation.append(p/p[-1])
    for i in np.arange(0, 4, 0.1):
        p = P1.dot(np.asarray([14.5,0,i,1])[:np.newaxis])
        points_animation.append(p/p[-1])
    for i in np.arange(0, 14.5, 0.1):
        p = P1.dot(np.asarray([14.5-i,0,4,1])[:np.newaxis])
        points_animation.append(p/p[-1])
    for i in np.arange(0, 4, 0.1):
        p = P1.dot(np.asarray([0,0,4-i,1])[:np.newaxis])
        points_animation.append(p/p[-1])

    i = 0
    while True:
        current_point = points_animation[i]
        cpy_img = np.copy(img1)
        cv2.circle(cpy_img, (int(current_point[0]), int(current_point[1])), 3, color=(0,0,255), thickness=-1)
        cv2.imshow("animation", cpy_img)
        i = (i + 1) % len(points_animation)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()

