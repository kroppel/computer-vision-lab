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
shown in two images given a set of corresponding points m1, m2

Params:
    m1 (list):  list of points in the first (left) image
    m2 (list):  list of points in the second (right) image 
                corresponding to the points in m1

Returns:
    h (np.ndarray): The homography matrix
"""
def estimate_homography(m1, m2):
    A = np.zeros((8,9))

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

def perform_mosaicing_it(imgL, imgR, H, mosaicing_shape):
    img_mosaicing = np.zeros(mosaicing_shape)

    coordinates_x, coordinates_y = np.meshgrid(np.arange(0, imgL.shape[1]), np.arange(0, imgL.shape[0]))
    coordinates_mosaic_x, coordinates_mosaic_y = np.meshgrid(np.arange(0, mosaicing_shape[1]), np.arange(0, mosaicing_shape[0]))


    coordinates_mosaic_stacked = np.concatenate([coordinates_mosaic_x[:,:,np.newaxis], coordinates_mosaic_y[:,:,np.newaxis], np.ones_like(coordinates_mosaic_x)[:,:,np.newaxis]], axis=2)

    old_coordinates_mosaic_stacked = np.einsum("hi,jki->jkh", H, coordinates_mosaic_stacked)
    old_coordinates_mosaic_stacked /= old_coordinates_mosaic_stacked[:,:,2][:,:,np.newaxis]

    coordinates_mask = np.logical_and(np.logical_and(old_coordinates_mosaic_stacked[:,:,0] >= 0, old_coordinates_mosaic_stacked[:,:,0] < img_mosaicing.shape[1]),
                            np.logical_and(old_coordinates_mosaic_stacked[:,:,1] >= 0, old_coordinates_mosaic_stacked[:,:,1] < img_mosaicing.shape[0]))#[:,:,np.newaxis]

    old_coordinates_mosaic_stacked[:,:,0] = np.clip(old_coordinates_mosaic_stacked[:,:,0], 0, imgR.shape[1]-1)
    old_coordinates_mosaic_stacked[:,:,1] = np.clip(old_coordinates_mosaic_stacked[:,:,1], 0, imgR.shape[0]-1)
    old_coordinates_mosaic_stacked = old_coordinates_mosaic_stacked.astype(int)

    img_mosaicing[coordinates_mosaic_y, coordinates_mosaic_x] += imgR[old_coordinates_mosaic_stacked[:,:,1], old_coordinates_mosaic_stacked[:,:,0]]
    img_mosaicing *= coordinates_mask
    img_mosaicing[coordinates_y, coordinates_x] += imgL[coordinates_y, coordinates_x]

    return img_mosaicing

def main():
    # Read in images and calibration parameters
    img1 = cv2.imread('../images/panorama_image1_big.jpg')
    img2 = cv2.imread('../images/panorama_image2_big.jpg')

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

    img3 = cv2.drawMatches(imgL,kp1,imgR,kp2,matches[0:4],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()

    y = []
    y_prime = []
    for match in matches[0:4]:
        y.append(np.array([kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1], 1]))
        y_prime.append(np.array([kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1], 1]))

    H = estimate_homography(y, y_prime)
    #img_mosaicing = perform_mosaicing_dt(imgL, imgR, H)
    img_mosaicing = perform_mosaicing_it(imgL, imgR, H, (imgL.shape[0], imgL.shape[1]+imgR.shape[1]))

    plt.imshow(img_mosaicing, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()