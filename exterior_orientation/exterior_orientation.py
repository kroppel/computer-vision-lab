import numpy as np
import cv2
import importlib
import sys
spec = importlib.util.spec_from_file_location("", "../shared/helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)

PERFORM_CALIBRATION = False

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
def collect_points(event, x, y, flags, points):
    # remove last item from list of points on leftclick
    if event==cv2.EVENT_LBUTTONDOWN:
        if len(points)!=0:
            print(str(points.pop()) + " removed")
    # collect point on rightclick
    if event==cv2.EVENT_RBUTTONDOWN and len(points)<8:
        print((x, y))
        points.append((x, y))

def estimate_exterior_orientation(m, M, K):
    print(K)
    UM,SM,VHM = np.linalg.svd(M.transpose())
    VM = VHM.transpose()
    rank = M.shape[1]
    # threshold for ~0 singular value
    threshold = 0.5
    for i in np.arange(1, rank+1):
        if SM[-i] < threshold:
            rank = rank - 1
        else:
            break
    VRM = VM[:,rank:]

    ### estimate scale factors
    A = np.zeros((M.shape[0]*(VM.shape[1]-rank)*3, M.shape[0]))
    D = np.zeros((3*M.shape[0], M.shape[0]))
    for i in np.arange(m.shape[0]):
        D[i*3:(i+1)*3,i]=m[i,:]
    for i in np.arange(M.shape[0]):
        A[i*(VM.shape[1]-rank)*3:(i+1)*(VM.shape[1]-rank)*3,:] = np.kron(VRM.transpose(),np.linalg.inv(K)).dot(D)
    UA, SA, VHA = np.linalg.svd(A)
    #scale_factors = VHA.transpose()[:,np.argmin(SA)]
    scale_factors = VHA.transpose()[:,-1]

    p = scale_factors[:,np.newaxis]*(np.linalg.inv(K).dot(m.transpose()).transpose())
    M = M[:,:-1]
    ### estimate external orientation

    # compute centroids and center data
    p_mean = np.sum(p, axis=0)/p.shape[0]
    M_mean = np.sum(M, axis=0)/M.shape[0]

    p_centered = p - p_mean[np.newaxis,:]
    M_centered = M - M_mean[np.newaxis,:]
    
    #### sign of s??
    s = np.sum(np.linalg.norm(p_centered, axis=1) / np.linalg.norm(M_centered, axis=1))/p_centered.shape[0]
    print("S: "+str(s))
    
    U, S, VH = np.linalg.svd((s*M_centered).transpose().dot(p_centered))
    S_new = np.identity(3)
    S_new[2,2] = np.linalg.det(VH.transpose().dot(U.transpose()))
    R = VH.transpose().dot(S_new).dot(U.transpose())
    t = (p_mean[:,np.newaxis]/s - R.dot(M_mean[:,np.newaxis]))

    return R, t

def main():
    # Read in images and calibration parameters
    img1 = cv2.imread('../images/new_scene1.jpg')
    img2 = cv2.imread('../images/new_scene2.jpg')

    if PERFORM_CALIBRATION:
        helper.calibrate_new_book_scenes_example()
    
    # load parameters to obtain internal camera parameters
    _, K1, R1_, t1_ = helper.read_projection_matrix_from_file("../images/data/new_params1")
    _, K2, R2_, t2_ = helper.read_projection_matrix_from_file("../images/data/new_params2")

    # real-world 3D-calibration points
    M = np.asarray([np.asarray([0,0,0,1], dtype=float),
                    np.asarray([0,0,4,1], dtype=float),
                    np.asarray([0,23,4,1], dtype=float),
                    np.asarray([14.5,23,4,1], dtype=float),
                    np.asarray([14.5,0,4,1], dtype=float),
                    np.asarray([14.5,0,0,1], dtype=float),
                    np.asarray([14.5,23,0,1], dtype=float),
                    np.asarray([0,23,0,1], dtype=float)])

    # 2D projections for first scene
    m1 = np.asarray([np.asarray([491, 500, 1]),
                    np.asarray([490, 442, 1]),
                    np.asarray([686, 367, 1]),
                    np.asarray([858, 400, 1]),
                    np.asarray([679, 505, 1]),
                    np.asarray([679, 576, 1]),
                    np.asarray([854, 461, 1]),
                    np.asarray([687, 413, 1])])

    R1, t1 = estimate_exterior_orientation(m1, M, K1)
    P1 = K1.dot(np.concatenate((R1,t1), axis=1))

    # Project points and see if it works
    cpy_img_resized = np.copy(img1)
    for i in np.arange(M.shape[0]):
        Mi = M[i,:]
        # project points AND! scale them to lie inside the image plane (3rd coordinate=1)
        mi = P1.dot(Mi[:,np.newaxis])
        mi = mi/mi[2,0]
        cv2.circle(cpy_img_resized, (int(mi[0,0]), int(mi[1,0])), 2, color=(0,255,0), thickness=-1)
        cv2.circle(cpy_img_resized, (int(m1[i,0]), int(m1[i,1])), 2, color=(0,0,255), thickness=-1)

    cv2.imshow("image", cpy_img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2D projections for second scene
    m2 = np.asarray([np.asarray([340, 479, 1]),
                    np.asarray([337, 419, 1]),
                    np.asarray([563, 350, 1]),
                    np.asarray([720, 381, 1]),
                    np.asarray([503, 477, 1]),
                    np.asarray([503, 549, 1]),
                    np.asarray([719, 442, 1]),
                    np.asarray([569, 400, 1])])

    R2, t2 = estimate_exterior_orientation(m2, M, K2)
    P2 = K2.dot(np.concatenate((R2,t2), axis=1))

    # Project points and see if it works
    cpy_img_resized = np.copy(img2)
    for i in np.arange(M.shape[0]):
        Mi = M[i,:]
        # project points AND! scale them to lie inside the image plane (3rd coordinate=1)
        mi = P2.dot(Mi[:,np.newaxis])
        mi = mi/mi[2,0]
        cv2.circle(cpy_img_resized, (int(mi[0,0]), int(mi[1,0])), 2, color=(0,255,0), thickness=-1)
        cv2.circle(cpy_img_resized, (int(m2[i,0]), int(m2[i,1])), 2, color=(0,0,255), thickness=-1)

    cv2.imshow("image", cpy_img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("vector view 1 to view 2: ")
    print((R2.dot(np.linalg.inv(R1).dot(-t1))+t2))

    print("------Rt1------")
    print(R1)
    print(t1)
    print("------Rt1_Calib------")
    print(R1_)
    print(t1_)
    print("------Rt2----")
    print(R2)
    print(t2)
    print("------Rt2_Calib----")
    print(R2_)
    print(t2_)



if __name__ == "__main__":
    main()