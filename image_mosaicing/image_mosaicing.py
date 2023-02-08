import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
import sys
spec = importlib.util.spec_from_file_location("", "../shared/helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)

"""Estimate the homography that describes the relation between a planar scene 
shown in two images given a set of corresponding points m1, m2. The specific relation is
m2 ~= H.dot(m1)

Params:
    m1 (list):  list of points in the first (left) image
    m2 (list):  list of points in the second (right) image 
                corresponding to the points in m1

Returns:
    H (np.ndarray): The homography matrix
"""
def estimate_homography(m1, m2):
    A = np.zeros((2*len(m1),9))
    H = None

    for i in np.arange(len(m1)):
        A[2*i:2*i+2,:] = np.kron(m1[i][:,np.newaxis].transpose(), 
            helper.calibration_direct_method.cross_product_matrix(m2[i][:,np.newaxis]))[0:2,:]

        _, _, VH = np.linalg.svd(A)
        H_vec = VH.transpose()[:,-1]
        H = np.reshape(H_vec, (3,3), order='F')

    return H

def perform_mosaicing_dt(imgL, imgR, H):
    img_mosaicing = np.zeros((imgL.shape[0]+imgR.shape[0], imgL.shape[1]+imgR.shape[1]))

    coordinates_x, coordinates_y = np.meshgrid(np.arange(0, imgR.shape[1]), np.arange(0, imgR.shape[0]))

    coordinates_stacked = np.concatenate([coordinates_x[:,:,np.newaxis], coordinates_y[:,:,np.newaxis], np.ones_like(coordinates_x)[:,:,np.newaxis]], axis=2)

    new_coordinates_stacked = np.einsum("hi,jki->jkh", np.linalg.inv(H), coordinates_stacked)
    new_coordinates_stacked /= new_coordinates_stacked[:,:,2][:,:,np.newaxis]
    new_coordinates_stacked[:,:,0] = np.clip(new_coordinates_stacked[:,:,0], 0, img_mosaicing.shape[1]-1)
    new_coordinates_stacked[:,:,1] = np.clip(new_coordinates_stacked[:,:,1], 0, img_mosaicing.shape[0]-1)
    new_coordinates_stacked = new_coordinates_stacked.astype(int)

    img_mosaicing[coordinates_y, coordinates_x] += imgL[coordinates_y, coordinates_x]
    img_mosaicing[new_coordinates_stacked[:,:,1], new_coordinates_stacked[:,:,0]] += imgR[coordinates_y, coordinates_x]

    return img_mosaicing

"""Create a mosaic image using inverse transform: given two input views of a planar scene
and the estimated homography that describes the relation between the first and the second image. 

Params:
    imgL (np.ndarray):          The first image of the planar scene
    imgR (np.ndarray):          The second image of the planar scene
    H (np.ndarray):             The matrix of the homography, for every point in the first image
                                    that lies on in the planar scene should be valid, that p' = H.dot(p),
                                    where p' is the corresponding point in the second image
    mosaicing_shape (tuple):    The desired shape of the mosaicing image

Returns:
    img_mosaicing (np.ndarray): The mosaic image

Note:
    This function will fail if mosaicing shape is smaller that imgL.shape
"""
def perform_mosaicing_it(imgL, imgR, H, mosaicing_shape):
    img_mosaicing = np.zeros(mosaicing_shape)

    coordinates_x, coordinates_y = np.meshgrid(np.arange(0, imgL.shape[1]), np.arange(0, imgL.shape[0]))
    coordinates_mosaic_x, coordinates_mosaic_y = np.meshgrid(np.arange(0, mosaicing_shape[1]), np.arange(0, mosaicing_shape[0]))
    coordinates_mosaic_stacked = np.concatenate([coordinates_mosaic_x[:,:,np.newaxis], coordinates_mosaic_y[:,:,np.newaxis], np.ones_like(coordinates_mosaic_x)[:,:,np.newaxis]], axis=2)

    old_coordinates_mosaic_stacked = np.einsum("hi,jki->jkh", H, coordinates_mosaic_stacked)
    old_coordinates_mosaic_stacked /= old_coordinates_mosaic_stacked[:,:,2][:,:,np.newaxis]

    coordinates_mask = np.logical_and(np.logical_and(old_coordinates_mosaic_stacked[:,:,0] >= 0, old_coordinates_mosaic_stacked[:,:,0] < np.min((imgR.shape[1], mosaicing_shape[1]))),
                            np.logical_and(old_coordinates_mosaic_stacked[:,:,1] >= 0, old_coordinates_mosaic_stacked[:,:,1] < np.min((imgR.shape[0], mosaicing_shape[0]))))#[:,:,np.newaxis]

    old_coordinates_mosaic_stacked[:,:,0] = np.clip(old_coordinates_mosaic_stacked[:,:,0], 0, imgR.shape[1]-1)
    old_coordinates_mosaic_stacked[:,:,1] = np.clip(old_coordinates_mosaic_stacked[:,:,1], 0, imgR.shape[0]-1)
    old_coordinates_mosaic_stacked = old_coordinates_mosaic_stacked.astype(int)

    img_mosaicing[coordinates_mosaic_y, coordinates_mosaic_x] += imgR[old_coordinates_mosaic_stacked[:,:,1], old_coordinates_mosaic_stacked[:,:,0]]
    img_mosaicing *= coordinates_mask
    img_mosaicing[coordinates_y, coordinates_x] += imgL[coordinates_y, coordinates_x]

    return img_mosaicing

def main():
    # Read in images
    img1 = cv2.imread('../images/panorama_image1_big.jpg')
    img2 = cv2.imread('../images/panorama_image2_big.jpg')

    #img1 = cv2.imread('../images/bridge1.jpg')
    #img2 = cv2.imread('../images/bridge2.jpg')

    # Mark area from where keypoints are extracted, then match and filter keypoints
    number_keypoints = 50
    imgL = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    rect_bounds = helper.get_marked_rectangle(np.copy(imgL))
    matches, kp1, kp2 = helper.perform_orb_keypoint_matching(imgL, imgR)
    matches_filtered = helper.filter_keypoint_matches(matches, kp1, rect_bounds, 15) # use threshold 15 for panorama and 30 for bridge
    if len(matches_filtered) < 4:
        print("Only found {} keypoint matches (Not enough!)".format(len(matches_filtered)))
        return

    img3 = cv2.drawMatches(imgL,kp1,imgR,kp2,matches_filtered[0:np.min((number_keypoints,len(matches_filtered)))],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()

    # Transfer 2D points into homogeneous coordinates and estimate homography
    y = []
    y_prime = []
    for match in matches_filtered[0:np.min((number_keypoints,len(matches_filtered)))]:
        y.append(np.array([kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1], 1]))
        y_prime.append(np.array([kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1], 1]))
    H = estimate_homography(y, y_prime)
    #img_mosaicing = perform_mosaicing_dt(imgL, imgR, H)
    img_mosaicing = perform_mosaicing_it(imgL, imgR, H, (imgL.shape[0], imgL.shape[1]+imgR.shape[1]-500))

    print(H)

    plt.imshow(img_mosaicing, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()