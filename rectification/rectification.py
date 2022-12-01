import numpy as np
import importlib.util
import sys
import cv2
spec = importlib.util.spec_from_file_location("", "../calibration/calibration_direct_method.py")
calibration_direct_method = importlib.util.module_from_spec(spec)
sys.modules["calibration_direct_method"] = calibration_direct_method
spec.loader.exec_module(calibration_direct_method)
spec = importlib.util.spec_from_file_location("", "../shared/helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)
spec = importlib.util.spec_from_file_location("", "../triangulation_and_epipolar_geometry/epipolar_geometry.py")
epipolar_geometry = importlib.util.module_from_spec(spec)
sys.modules["epipolar_geometry"] = epipolar_geometry
spec.loader.exec_module(epipolar_geometry)

def get_rectified_projection_matrices(parameters_list_left, parameters_list_right):
    P1, K1, R1, t1 = parameters_list_left
    P2, K2, R2, t2 = parameters_list_right

    # Construct new projection matrices:
    R_rect = np.zeros_like(R1)
    # 1. Calculate 3D coordinates of the optical centers
    C1_3D = -np.linalg.inv(P1[:3,:3]).dot(P1[:,-1])
    C2_3D = -np.linalg.inv(P2[:3,:3]).dot(P2[:,-1])
    # 2. Construct the new rotation matrix
    R_rect[0,:] = (C2_3D-C1_3D)/(np.linalg.norm(C2_3D-C1_3D))
    R_rect[1,:] = np.cross(R1[2,:], R_rect[0,:])
    R_rect[2,:] = -np.cross(R_rect[0,:], R_rect[1,:])
    # 3. Calculate the new translation vectors
    t1_rect = -R_rect.dot(C1_3D)
    t2_rect = -R_rect.dot(C2_3D)
    # 4. Calculate the new projection matrices
    P1_rect = K1.dot(np.concatenate([R_rect, t1_rect[:,np.newaxis]], axis=1))
    P2_rect = K1.dot(np.concatenate([R_rect, t2_rect[:,np.newaxis]], axis=1))

    return [P1_rect, K1, R_rect, t1_rect], [P2_rect, K1, R_rect, t2_rect]

def perform_rect_direct_transform(img, T):
    coordinates_x, coordinates_y = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    coordinates_z = np.ones_like(coordinates_x)

    coordinates_transformed = np.einsum("hi,jki->jkh", T, np.concatenate([coordinates_x[:,:,np.newaxis],coordinates_y[:,:,np.newaxis],coordinates_z[:,:,np.newaxis]], axis=2))
    coordinates_transformed[:,:,] /= coordinates_transformed[:,:,-1][:,:,np.newaxis]
    coordinates_transformed[:,:,0] = np.clip(coordinates_transformed[:,:,0], 0, img.shape[1]-1)
    coordinates_transformed[:,:,1] = np.clip(coordinates_transformed[:,:,1], 0, img.shape[0]-1)
    coordinates_transformed = coordinates_transformed.astype(int)

    img_rect = np.zeros_like(img)
    img_rect[coordinates_transformed[:,:,1],coordinates_transformed[:,:,0]] = img

    return img_rect

def perform_rect_inverse_transform(img, T):
    coordinates_x, coordinates_y = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    coordinates_z = np.ones_like(coordinates_x)

    coordinates_transformed = np.einsum("hi,jki->jkh", np.linalg.inv(T), np.concatenate([coordinates_x[:,:,np.newaxis],coordinates_y[:,:,np.newaxis],coordinates_z[:,:,np.newaxis]], axis=2))
    coordinates_transformed[:,:,] /= coordinates_transformed[:,:,-1][:,:,np.newaxis]
    coordinates_transformed = coordinates_transformed.astype(int)

    coordinates_mask = np.logical_and(np.logical_and(coordinates_transformed[:,:,0] >= 0, coordinates_transformed[:,:,0] < img.shape[1]),
                            np.logical_and(coordinates_transformed[:,:,1] >= 0, coordinates_transformed[:,:,1] < img.shape[0]))[:,:,np.newaxis]

    coordinates_transformed[:,:,0] = np.clip(coordinates_transformed[:,:,0], 0, img.shape[1]-1)
    coordinates_transformed[:,:,1] = np.clip(coordinates_transformed[:,:,1], 0, img.shape[0]-1)

    img_rect = np.zeros_like(img)
    img_rect = img[coordinates_transformed[:,:,1],coordinates_transformed[:,:,0]]
    img_rect = img_rect * coordinates_mask


    return img_rect

def main():
    # Read in images and calibration parameters
    img1 = cv2.imread('../images/scene1.jpg')
    img2 = cv2.imread('../images/scene2.jpg')
        
    P1, K1, R1, t1 = helper.read_projection_matrix_from_file("../images/data/params1")
    P2, K2, R2, t2 = helper.read_projection_matrix_from_file("../images/data/params2")

    parameters_rect_list_left, parameters_rect_list_right = get_rectified_projection_matrices([P1, K1, R1, t1], [P2, K2, R2, t2])
    P1_rect, K1_rect, R1_rect, t1_rect = parameters_rect_list_left
    P2_rect, K2_rect, R2_rect, t2_rect = parameters_rect_list_right

    print(np.linalg.det(R1_rect))
    print(np.linalg.det(R2_rect))

    # Define the coordinate transformation matrices
    T1 = P1_rect[:3,:3].dot(np.linalg.inv(P1[:3,:3]))
    T2 = P2_rect[:3,:3].dot(np.linalg.inv(P2[:3,:3]))

    img1_rect = perform_rect_direct_transform(img1, T1)
    img2_rect = perform_rect_direct_transform(img2, T2)

    # resize and show images
    img1_rect_resized = cv2.resize(img1_rect, (int(img1_rect.shape[1]/2), int(img1_rect.shape[0]/2)))
    img2_rect_resized = cv2.resize(img2_rect, (int(img1_rect.shape[1]/2), int(img2_rect.shape[0]/2)))
    img_combined = np.concatenate([img1_rect_resized, img2_rect_resized], axis=1)
    cv2.imshow("images rectified direct transform", img_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img1_rect = perform_rect_inverse_transform(img1, T1)
    img2_rect = perform_rect_inverse_transform(img2, T2)

    # resize and show images
    img1_rect_resized = cv2.resize(img1_rect, (int(img1_rect.shape[1]/2), int(img1_rect.shape[0]/2)))
    img2_rect_resized = cv2.resize(img2_rect, (int(img1_rect.shape[1]/2), int(img2_rect.shape[0]/2)))
    img_combined = np.concatenate([img1_rect_resized, img2_rect_resized], axis=1)
    cv2.imshow("images rectified inverse transform", img_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # draw epipolar lines
    points_epipolar = []
    img_epipolar = np.concatenate([img1_rect_resized, img2_rect_resized], axis=1)
    cv2.imshow("epipolar line", img_epipolar)
    cv2.setMouseCallback('epipolar line', epipolar_geometry.collect_points, points_epipolar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for point in points_epipolar:
        # transformation of point into original coordinates (multiply each coordinate with scaling factor)
        point_orig = point * 2
        epipolar_geometry.draw_epipolar_line(img2_rect, P1_rect, P2_rect, np.concatenate((point_orig, np.ones((1,1))), axis=0))
        cv2.circle(img1_rect_resized, (point[0], point[1]), 3, color=(0,0,255), thickness=-1)
    
    img_epipolar = np.concatenate((img1_rect_resized,cv2.resize(img2_rect, (int(img2_rect.shape[1]/2), int(img2_rect.shape[0]/2)))), axis=1)

    if len(points_epipolar) > 0:
        cv2.imshow("epipolar line", img_epipolar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()

    
