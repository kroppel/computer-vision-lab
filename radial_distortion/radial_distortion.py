import cv2
import numpy as np
import sys
import importlib
spec = importlib.util.spec_from_file_location("", "../shared/helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)

"""perform one of two actions on a mouse click event:
if right-click: add coordinates of clicked point to list (params)
if left-click: remove last point coordinates from list (params)

Params:
    event:  Type of mouse event that occurred
    x:      X coordinate of the mouse at the moment of the occured event
    y:      Y coordinate of the mouse at the moment of the occured event
    flags:  - (unused)
    params: Reference to list that contains the collected points
"""
def collect_calibration_points(event, x, y, flags, points):
    # remove last item from list of points on leftclick
    if event==cv2.EVENT_LBUTTONDOWN:
        if len(points)!=0:
            print(str(points.pop()) + " removed")
    # collect point on rightclick
    if event==cv2.EVENT_RBUTTONDOWN:
        print((x, y))
        points.append((x, y))

"""Applies radial distortion using a single parameter distortion model via direct transformation

Params:
    img_old (np.ndarray):           The input image.
    K (np.ndarray):                 The matrix of internal parameters
    distortion_coefficient (float): Coefficient that controls the amount of distortion introduced

Returns:
    img_new (np.ndarray):   The distorted image
"""
def transformation_distortion_direct(img_old, K, distortion_coefficient= -0.4):
    img_new = np.zeros_like(img_old)
    coordinates_x, coordinates_y = np.meshgrid(np.arange(1, img_old.shape[1]), np.arange(1, img_old.shape[0]))

    C = K[:,-1]

    alpha_u = K[0,0]
    axis_angle = np.arccos(K[0,1]/alpha_u)
    alpha_v = K[1,1]*np.sin(axis_angle)

    RD_squared = np.power((coordinates_x-C[0]) / (alpha_u), 2) \
            + np.power((coordinates_y-C[1]) / (alpha_v), 2)

    new_coordinates_x = np.clip((coordinates_x-C[0])*(1+distortion_coefficient*RD_squared)+C[0], 0, img_new.shape[1]-1).astype(int)
    new_coordinates_y = np.clip((coordinates_y-C[1])*(1+distortion_coefficient*RD_squared)+C[1], 0, img_new.shape[0]-1).astype(int)

    img_new[new_coordinates_y, new_coordinates_x] = img_old[1:,1:,:]
    
    return img_new

"""Applies radial distortion using a single parameter distortion model via inverse transformation

Params:
    img_old (np.ndarray):           The input image.
    K (np.ndarray):                 The matrix of internal parameters
    distortion_coefficient (float): Coefficient that controls the amount of distortion introduced

Returns:
    img_new (np.ndarray):   The distorted image
"""
def transformation_distortion_inverse(img_old, K, distortion_coefficient= -0.4):
    img_new = np.zeros_like(img_old)
    coordinates_x, coordinates_y = np.meshgrid(np.arange(1, img_old.shape[1]), np.arange(1, img_old.shape[0]))

    C = K[:,-1]

    alpha_u = K[0,0]
    axis_angle = np.arccos(K[0,1]/alpha_u)
    alpha_v = K[1,1]*np.sin(axis_angle)

    RD_squared = np.power((coordinates_x-C[0]) / (alpha_u), 2) \
            + np.power((coordinates_y-C[1]) / (alpha_v), 2)

    old_coordinates_x = (coordinates_x-C[0])/(1+distortion_coefficient*RD_squared)+C[0]
    old_coordinates_y = (coordinates_y-C[1])/(1+distortion_coefficient*RD_squared)+C[1]

    coordinate_mask = np.logical_and(np.logical_and(old_coordinates_x < img_new.shape[1], old_coordinates_x >= 0), \
                np.logical_and(old_coordinates_y < img_new.shape[0], old_coordinates_y >= 0))

    old_coordinates_x = np.clip(old_coordinates_x, 0, img_new.shape[1]-1).astype(int)
    old_coordinates_y = np.clip(old_coordinates_y, 0, img_new.shape[0]-1).astype(int)

    img_new[1:,1:,:] += img_old[old_coordinates_y, old_coordinates_x, :]*coordinate_mask[:,:,np.newaxis]

    return img_new

"""Estimates the distortion coefficient using 3D to 2D correspondences and the perspective matrix of the image.


Note: You have to collect as many image points as you define in points_3D variable.
      Using points close to principal point should be avoided, as impreciseness impacts the estimation significantly.
"""
def estimate_distortion_coefficient(points, points_3D, K, P):
    # 2 equations per point
    coeff_est = np.zeros((2*len(points_3D))) 

    # get principal point
    C = K[:,-1]
    print("Principal Point: {}".format(C))    

    for i in np.arange(len(points_3D)):
        if len(points_3D) != len(points):
            print("Warning: number of points collected ({}) not equal to number of 3D points defined ({})!".format(len(points), len(points_3D)))
            exit(-1)
        # get undistorted point projection (assuming P is correct)
        m = P.dot(points_3D[i][:,np.newaxis])
        m /= m[-1]
        
        # calculate RD_squared term
        alpha_u = K[0,0]
        axis_angle = np.arccos(K[0,1]/alpha_u)
        alpha_v = K[1,1]*np.sin(axis_angle)
        RD_squared = np.power((m[0]-C[0]) / (alpha_u), 2) \
                + np.power((m[1]-C[1]) / (alpha_v), 2)

        # add equations
        coeff_est[2*i] = (points[i][0]-m[0]) / ((m[0]-C[0])*RD_squared)
        coeff_est[2*i+1] = (points[i][1]-m[1]) / ((m[1]-C[1])*RD_squared)
    
    print("Coeff estimates: {}".format(coeff_est))
    print("Coeff Average: {}".format(np.sum(coeff_est)/len(coeff_est)))
    # return least squares estimation for coeff
    return np.sum(np.power(coeff_est,2))/np.sum(coeff_est)
    
##### Read in image and parameters
img_undistorted = cv2.imread('../images/new_scene1.jpg')
P, K, R, t = helper.read_projection_matrix_from_file("../images/data/new_params1")

##### perform distortion transformation
coeff = 0.7 # e.g. 0.7 for pincushion distortion,  e.g. -0.4 for barrel distortion
print("distort image with coeff = " + str(coeff))
#img_distorted = transformation_distortion_direct(img_undistorted, K, coeff)
img_distorted = transformation_distortion_inverse(img_undistorted, K, coeff)

##### resize and show images
img_undistorted_resize = cv2.resize(img_undistorted, (int(img_undistorted.shape[1]/2), int(img_undistorted.shape[0]/2)))
img_distorted_resize = cv2.resize(img_distorted, (int(img_distorted.shape[1]/2), int(img_distorted.shape[0]/2)))
img_combined = np.concatenate([img_undistorted_resize, img_distorted_resize], axis=1)

##### estimate distortion coefficient using points collected by "user" or "ideal" (computed) distorted points
collect_by_user = False
points_3D = [np.asarray([14.5, 23, 4, 1]), np.asarray([14.5, 23, 0, 1]), np.asarray([14.5, 0, 0, 1])]
points = []

if collect_by_user:
    cv2.imshow("distorted image", img_distorted)
    cv2.setMouseCallback('distorted image', collect_calibration_points, points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    for point in points_3D:
        # get undistorted point projection (assuming P is correct)
        p = P.dot(point[:,np.newaxis])
        p /= p[-1]

        C = K[:,-1]
        
        alpha_u = K[0,0]
        axis_angle = np.arccos(K[0,1]/alpha_u)
        alpha_v = K[1,1]*np.sin(axis_angle)

        RD_squared = np.power((p[0,0]-C[0]) / (alpha_u), 2) \
                + np.power((p[1,0]-C[1]) / (alpha_v), 2)

        new_coordinates_x = ((p[0,0]-C[0])*(1+coeff*RD_squared)+C[0]).astype(int)
        new_coordinates_y = ((p[1,0]-C[1])*(1+coeff*RD_squared)+C[1]).astype(int)
        points.append((new_coordinates_x, new_coordinates_y))

print("coeff estimation: " + str(estimate_distortion_coefficient(points, points_3D, K, P)))

cv2.imshow("image_original_and_distorted", img_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
