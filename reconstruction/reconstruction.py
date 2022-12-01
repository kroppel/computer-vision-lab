import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import importlib
import sys
spec = importlib.util.spec_from_file_location("", "../rectification/rectification.py")
rectification = importlib.util.module_from_spec(spec)
sys.modules["rectification"] = rectification
spec.loader.exec_module(rectification)
spec = importlib.util.spec_from_file_location("", "../shared/helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)
spec = importlib.util.spec_from_file_location("", "../triangulation_and_epipolar_geometry/triangulation.py")
triangulation = importlib.util.module_from_spec(spec)
sys.modules["triangulation"] = triangulation
spec.loader.exec_module(triangulation)

PERFORM_CALIBRATION = False

def main():
    # Read in images and calibration parameters
    img1 = cv2.imread('../images/new_scene1.jpg')
    img2 = cv2.imread('../images/new_scene2.jpg')

    if PERFORM_CALIBRATION:
        helper.calibrate_new_book_scenes_example()
        
    P1, K1, R1, t1 = helper.read_projection_matrix_from_file("../images/data/new_params1")
    P2, K2, R2, t2 = helper.read_projection_matrix_from_file("../images/data/new_params2")

    parameters_rect_list_left, parameters_rect_list_right = rectification.get_rectified_projection_matrices([P1, K1, R1, t1], [P2, K2, R2, t2])
    P1_rect, K1_rect, R1_rect, t1_rect = parameters_rect_list_left
    P2_rect, K2_rect, R2_rect, t2_rect = parameters_rect_list_right

    print(np.linalg.det(R1_rect))
    print(np.linalg.det(R2_rect))

    # Define the coordinate transformation matrices
    T1 = P1_rect[:3,:3].dot(np.linalg.inv(P1[:3,:3]))
    T2 = P2_rect[:3,:3].dot(np.linalg.inv(P2[:3,:3]))

    img1_rect = rectification.perform_rect_inverse_transform(img1, T1)
    img2_rect = rectification.perform_rect_inverse_transform(img2, T2)

    imgL = cv2.cvtColor(img1_rect,cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rect,cv2.COLOR_BGR2GRAY)

    # collect 2 points from each image
    points_img1 = []
    cv2.imshow("image", imgL)
    cv2.setMouseCallback('image', triangulation.collect_points, points_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    points_img2 = []
    cv2.imshow("image", imgR)
    cv2.setMouseCallback('image', triangulation.collect_points, points_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    points_3D = []
    # reconstruct 3D coordinates of every sample point
    for i in np.arange(len(points_img1)):
        M = triangulation.get_coordinates_3D(points_img1[i], points_img2[i], P1_rect, P2_rect)
        points_3D.append(M)
        print(M)
    if (len(points_3D)==2):
        print("Euclidean distance: " + str(np.sqrt(np.sum(np.power(points_3D[0][0:3]-points_3D[1][0:3], 2)))))
    
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

    matches_filtered = []
    for match in matches:
        if np.abs(kp1[match.queryIdx].pt[1]-kp2[match.trainIdx].pt[1]) < 50 and \
            kp1[match.queryIdx].pt[1] > 326 and kp1[match.queryIdx].pt[1] < 550 and \
            kp1[match.queryIdx].pt[0] > 372 and kp1[match.queryIdx].pt[0] < 746 and \
            kp2[match.trainIdx].pt[1] > 326 and kp2[match.trainIdx].pt[1] < 550: \
            
            matches_filtered.append(match)

    print(len(matches))
    print(len(matches_filtered))


    img3 = cv2.drawMatches(imgL,kp1,imgR,kp2,matches_filtered,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

    points_3D = []
    for match in matches_filtered:
        points_3D.append(triangulation.get_coordinates_3D(np.asarray([kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1]])[:,np.newaxis], np.asarray([kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1]])[:,np.newaxis], P1_rect, P2_rect))
        print(np.asarray([kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1]]))
        print(points_3D[-1])

    points_3D_array = np.asarray(points_3D)
    points_3D_array = points_3D_array[np.all(np.abs(points_3D_array[:,0:3,:])<100, axis=1)[:,0]]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_3D_array[:,0,:], points_3D_array[:,1,:], points_3D_array[:,2,:])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    samples = np.arange(-100, 100, 2)
    ax.plot(xs=samples, ys=np.zeros_like(samples), zs=0)
    ax.plot(xs=np.zeros_like(samples), ys=samples, zs=0)
    ax.plot(xs=np.zeros_like(samples), ys=np.zeros_like(samples), zs=samples)


    plt.show()






if __name__ == "__main__":
    main()