import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
import sys
spec = importlib.util.spec_from_file_location("", "../calibration/calibration_direct_method.py")
calibration_direct_method = importlib.util.module_from_spec(spec)
sys.modules["calibration_direct_method"] = calibration_direct_method
spec.loader.exec_module(calibration_direct_method)
spec = importlib.util.spec_from_file_location("", "../shared/helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)

PERFORM_CALIBRATION = False

def estimate_essential_matrix(y, y_prime, K1, K2):
    Y = np.zeros((9,8))
    for i in np.arange(len(y)):
        Y[:,i] = y_prime[i][:,np.newaxis].dot(y[i][:,np.newaxis].transpose()).flatten(order='F')
    
    _, _, VH = np.linalg.svd(Y.transpose())
    E_vec = VH.transpose()[:,-1][:,np.newaxis]
    E = np.reshape(E_vec, (3,3), 'F')

    """U, S, VH = np.linalg.svd(E)

    S_constrained = np.zeros((3,3))
    S_constrained[0,0] = S[0]
    S_constrained[1,1] = S[1]

    return U.dot(S_constrained.dot(VH))"""

    return E

def estimate_essential_matrix_v2(y, y_prime, K1, K2):
    Y = np.zeros((8,9))
    for i in np.arange(len(y)):
        # normalize points and build lin. equ. sys.
        Y[i,:] = np.kron(K1.dot(y[i]), K2.dot(y_prime[i]))

    _, _, VH = np.linalg.svd(Y)
    E_vec = VH.transpose()[:,-1][:,np.newaxis]
    E = np.reshape(E_vec, (3,3), 'F')

    U, S, VH = np.linalg.svd(E)

    print("SVD")
    print(S)

    S_constrained = np.zeros((3,3))
    S_constrained[0,0] = (S[0]+S[1])/2
    S_constrained[1,1] = (S[0]+S[1])/2
    S_constrained[2,2] = 0


    return U.dot(S_constrained.dot(VH))

    return E

def estimate_external_parameters(E):
    U, S, VH = np.linalg.svd(E)
    W = np.asarray([[0,-1,0],[1,0,0],[0,0,1]])
    t = U.dot(W.dot(S.dot(U.transpose())))
    R = U.dot(W.transpose().dot(VH))

    return R, VH.transpose()[:,-1]

def main():
    # Read in images and calibration parameters
    img1 = cv2.imread('../images/new_scene1.jpg')
    img2 = cv2.imread('../images/new_scene2.jpg')

    if PERFORM_CALIBRATION:
        helper.calibrate_new_book_scenes_example()
    
    # load parameters to obtain internal camera parameters (only use K1 as camera is the same from both scenes)
    P1, K1, R1, t1 = helper.read_projection_matrix_from_file("../images/data/new_params1")
    P2, K2, R2, t2 = helper.read_projection_matrix_from_file("../images/data/new_params2")

    imgL = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

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

    print(len(matches))

    img3 = cv2.drawMatches(imgL,kp1,imgR,kp2,matches[0:8],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

    y = []
    y_prime = []
    for match in matches[0:8]:
        y.append(np.array([kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1], 1]))
        y_prime.append(np.array([kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1], 1]))

    E = estimate_essential_matrix(y, y_prime, K1, K1)
    E2 = estimate_essential_matrix_v2(y, y_prime, K1, K1)

    print(E)
    print(E2)

    R, t = estimate_external_parameters(E)

if __name__ == "__main__":
    main()