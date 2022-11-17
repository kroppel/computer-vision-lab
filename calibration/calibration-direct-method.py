import numpy as np
import cv2

"""perform one of two actions on a mouse click event:
if right-click: add coordinates of clicked point to list (params)
if left-click: remove last point coordinates from list (params)

event:  Type of mouse event that occurred
x:      X coordinate of the mouse at the moment of the occured event
y:      Y coordinate of the mouse at the moment of the occured event
flags:  - (unused)
params: Reference to list that contains the collected points

"""
def click_event(event, x, y, flags, params):
    # remove last item from list of points on leftclick
    if event==cv2.EVENT_LBUTTONDOWN:
        if len(params)!=0:
            print(str(params.pop()) + " removed")
    # collect point on rightclick
    if event==cv2.EVENT_RBUTTONDOWN and len(params)<6:
        print((x, y))
        points.append((x, y))

"""Return the cross product matrix of a column vector

a (np.ndarray): the input column vector
"""
def cross_product_matrix(a):
    return np.asarray([np.asarray([0, -a[2,0], a[1,0]]),
                    np.asarray([a[2,0], 0, -a[0,0]]),
                    np.asarray([-a[1,0], a[0,0], 0])])
                    
"""Perform direct calibration method and estimate
the projection matrix P, and factorize it to retrieve external and internal camera parameters

M (np.ndarray): Array of 3D calibration points
m (np.ndarray): Array of projected calibration points

Returns: 
P (np.ndarray): The estimated projection matrix
K (np.ndarray): Matrix of internal parameters 
R (np.ndarray): Matrix of external parameters describing the rotation
t (np.ndarray): Vector of external parameters describing the translation
"""
def calibration_direct_method(M, m):
    # fill A with the equations obtained from the pairs (mi, Mi)
    A = np.zeros((12, 12))
    for i in np.arange(len(m)):
        mi = m[i, np.newaxis].transpose()
        Mi = M[i, np.newaxis].transpose()
        A[2*i:2*i+2,:] = np.kron(Mi.transpose(), cross_product_matrix(mi))[0:2,:]

    # Singular Value Decomposition and retrieval of vec(P)
    _, _, VH = np.linalg.svd(A)
    # As VH is the transpose of V, we transpose it again and retrieve the last column.
    # Then we reshape the vector to matrix layout 3x4 in Fortran order (column-major filling)
    P_vec = VH.transpose()[:,-1][:,np.newaxis]
    P = np.reshape(P_vec, (3,4), 'F')

    # P = [Q,q] = [KR, Kt]
    # -> factorize inv(Q) into inv(K*R) = inv(R) * inv(K)
    # -> invert inv(R) to get R
    # -> calculate t = inv(K) * q
    q = P[:,-1,np.newaxis]
    Q = P[:,:-1]
    R_inv, K_inv = np.linalg.qr(Q, 'complete')

    R = np.linalg.inv(R_inv)
    if np.linalg.det(R) < 0:
        R *= -1
    K =np.linalg.inv(K_inv)
    t = K_inv.dot(q)
        
    print("Determinant of R: " + str(np.linalg.det(R)))

    return P, K, R, t 


# collect points indicated by user
points = []
img = cv2.imread('image1.jpg')
img_resized = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
cv2.imshow("image", img_resized)
coordinates = cv2.setMouseCallback('image', click_event, points)
cv2.waitKey(0)
cv2.destroyAllWindows()

# real-world 3D-calibration points
M = np.asarray([np.asarray([0,0,0,1], dtype=float),
                np.asarray([0,0,4,1], dtype=float),
                np.asarray([0,23,4,1], dtype=float),
                np.asarray([14.5,23,4,1], dtype=float),
                np.asarray([14.5,0,4,1], dtype=float),
                np.asarray([14.5,0,0,1], dtype=float)])

# 2D projections
m = np.asarray([np.asarray([points[0][0], points[0][1], 1]),
                np.asarray([points[1][0], points[1][1], 1]),
                np.asarray([points[2][0], points[2][1], 1]),
                np.asarray([points[3][0], points[3][1], 1]),
                np.asarray([points[4][0], points[4][1], 1]),
                np.asarray([points[5][0], points[5][1], 1])])

P, K, R, t = calibration_direct_method(M, m)

print("Projection Matrix P: ")
print(P)

print("Intrinsic Parameters K: ")
print(K)

print("Extrinsic Parameters R, t: ")
print(R)
print(t)


####### DEMO #######

# Project calibration points and see if it works
cpy_img_resized = np.copy(img_resized)
for Mi in M:
    # project points AND! scale them to lie inside the image plane (3rd coordinate=1)
    mi = P.dot(Mi[:,np.newaxis])
    mi = mi/mi[2,0]
    cv2.circle(cpy_img_resized, (int(mi[0,0]), int(mi[1,0])), 2, color=(0,255,0), thickness=-1)
cv2.imshow("image", cpy_img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Do some more projections
cpy_img_resized = np.copy(img_resized)
for i in np.arange(-10, 10):
    # x-axis-projection
    proj_x = np.reshape(P.dot(np.asarray([i*2,0,0,1])), (3,1), 'F')
    proj_x = proj_x/proj_x[2,0]
    cv2.circle(cpy_img_resized, (int(proj_x[0,0]), int(proj_x[1,0])), 1, color=(0,0,255), thickness=-1)
    # y-axis-projection
    proj_y = np.reshape(P.dot(np.asarray([0,i*2,0,1])), (3,1), 'F')
    proj_y = proj_y/proj_y[2,0]
    cv2.circle(cpy_img_resized, (int(proj_y[0,0]), int(proj_y[1,0])), 1, color=(0,0,255), thickness=-1)
    # z-axis-projection
    proj_z = np.reshape(P.dot(np.asarray([0,0,i*2,1])), (3,1), 'F')
    proj_z = proj_z/proj_z[2,0]
    cv2.circle(cpy_img_resized, (int(proj_z[0,0]), int(proj_z[1,0])), 1, color=(0,0,255), thickness=-1)
for i in np.arange(-10, 10, 0.5):
    # projection of sample-points of function f(y)=(0, y, cos(y)) 
    # -> draw cosine of y in plane x=0 where z represents the function value 
    proj_z = np.reshape(P.dot(np.asarray([0,i,np.cos(i),1])), (3,1), 'F')
    proj_z = proj_z/proj_z[2,0]
    cv2.circle(cpy_img_resized, (int(proj_z[0,0]), int(proj_z[1,0])), 1, color=(0,0,0), thickness=-1)

# Define corners of a rectangle on top of the table in 3D space and project it onto image
#cpy_img_resized = np.copy(img_resized)
corner_points = [np.asarray([-10,0,0,1]),
                np.asarray([-20,0,0,1]),
                np.asarray([-10,20,0,1]),
                np.asarray([-20,20,0,1]),
                np.asarray([-10,0,5,1]),
                np.asarray([-20,0,5,1]),
                np.asarray([-10,20,5,1]),
                np.asarray([-20,20,5,1])]

for point1 in corner_points:
    for point2 in corner_points:
        if np.sum(point1==point2)==3:
            point_proj1 = P.dot(point1)
            point_proj1 = point_proj1/point_proj1[2]
            point_proj2 = P.dot(point2)
            point_proj2 = point_proj2/point_proj2[2]
            cv2.line(cpy_img_resized, (int(point_proj1[0]), int(point_proj1[1])), (int(point_proj2[0]), int(point_proj2[1])), color=(255,0,0), thickness=1)

cv2.imshow("image", cpy_img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
