import cv2
import numpy as np
import sys
import importlib
spec = importlib.util.spec_from_file_location("", "../triangulation_and_epipolar_geometry/helper.py")
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

def estimate_distortion_coefficient(img, K, P):
    # collect points indicated by user
    points = []
    cv2.imshow("image", img)
    cv2.setMouseCallback('image', collect_calibration_points, points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    points_3D = [np.asarray([14.5, 23, 4, 1]), ]
    m = P.dot(points_3D[0][:,np.newaxis])
    m /= m[-1]

    C = K[:,-1]
    K = K / C[-1]
    C = C / C[-1]


    alpha_u = K[0,0]
    axis_angle = np.arccos(K[0,1]/alpha_u)
    alpha_v = K[1,1]*np.sin(axis_angle)


    RD_squared = np.power((m[0]-C[0]) / (alpha_u), 2) \
            + np.power((m[1]-C[1]) / (alpha_v), 2)


    return ((m[0]-C[0])*RD_squared*(points[0][0]-m[0])/(m[0]-C[0])*RD_squared*(m[0]-C[0])*RD_squared)
    



img_undistorted = cv2.imread('../images/scene1.jpg')

P, K, R, t = helper.read_projection_matrix_from_file("../images/data/params1")

# perform distortion and interpolation
img_distorted = transformation_distortion(img_undistorted, K)
img_distorted_interpolated = interpolate(img_distorted)

# resize and show images
img_undistorted_resize = cv2.resize(img_undistorted, (int(img_undistorted.shape[1]/2), int(img_undistorted.shape[0]/2)))
img_distorted_interpolated_resize = cv2.resize(img_distorted_interpolated, (int(img_distorted_interpolated.shape[1]/2), int(img_distorted_interpolated.shape[0]/2)))
img_combined = np.concatenate([img_undistorted_resize, img_distorted_interpolated_resize], axis=1)

print("K estimation: " + str(estimate_distortion_coefficient(img_distorted, K, P)))

cv2.imshow("images_distorted", img_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
