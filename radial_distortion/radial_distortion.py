import cv2
import numpy as np
import sys
import importlib
spec = importlib.util.spec_from_file_location("", "../triangulation_and_epipolar_geometry/helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)

def transformation_distortion(img_old, K, distortion_coefficient= -0.4):
    img_new = np.zeros_like(img_old)
    coordinates_x, coordinates_y = np.meshgrid(np.arange(1, img_old.shape[1]), np.arange(1, img_old.shape[0]))

    C = K[:,-1]
    K = K / C[-1]
    C = C / C[-1]


    alpha_u = K[0,0]
    axis_angle = np.arccos(K[0,1]/alpha_u)
    alpha_v = K[1,1]*np.sin(axis_angle)

    RD_squared = np.power((coordinates_x-C[0]) / (alpha_u), 2) \
            + np.power((coordinates_y-C[1]) / (alpha_v), 2)

    new_coordinates_x = np.clip((coordinates_x-C[0])*(1+distortion_coefficient*RD_squared)+C[0], 0, img_new.shape[1]-1).astype(int)
    new_coordinates_y = np.clip((coordinates_y-C[1])*(1+distortion_coefficient*RD_squared)+C[1], 0, img_new.shape[0]-1).astype(int)

    img_new[new_coordinates_y, new_coordinates_x] = img_old[1:,1:,:]
    


    return img_new

def interpolate(img):
    img_new = np.copy(img)

    for i in np.arange(1, img.shape[0]-1):
        mask = np.logical_and(np.logical_and(img[i,:,0]==0, img[i,:,1]==0), img[i,:,2]==0)
        img_new[i,mask] = img_new[i,mask]+((img[i-1,mask] + img[i+1,mask])/4).astype(np.uint8)
    
    for i in np.arange(1, img.shape[1]-1):
        mask = np.logical_and(np.logical_and(img[:,i,0]==0, img[:,i,1]==0), img[:,i,2]==0)
        img_new[mask,i] = img_new[mask,i]+((img[mask,i-1] + img[mask,i+1])/4).astype(np.uint8)

    return img_new


img_undistorted = cv2.imread('../images/scene1.jpg')

P, K, R, t = helper.read_projection_matrix_from_file("../images/data/params1")

# perform distortion and interpolation
img_distorted = transformation_distortion(img_undistorted, K)
img_distorted_interpolated = interpolate(img_distorted)

# resize and show images
img_undistorted = cv2.resize(img_undistorted, (int(img_undistorted.shape[1]/2), int(img_undistorted.shape[0]/2)))
img_distorted = cv2.resize(img_distorted, (int(img_distorted.shape[1]/2), int(img_distorted.shape[0]/2)))
img_distorted_interpolated = cv2.resize(img_distorted_interpolated, (int(img_distorted_interpolated.shape[1]/2), int(img_distorted_interpolated.shape[0]/2)))
img_combined = np.concatenate([img_undistorted, img_distorted_interpolated], axis=1)

cv2.imshow("images_distorted", img_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

