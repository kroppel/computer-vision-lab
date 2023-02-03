import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
import sys
spec = importlib.util.spec_from_file_location("", "../shared/helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)
spec = importlib.util.spec_from_file_location("", "../image_mosaicing/image_mosaicing.py")
mosaicing = importlib.util.module_from_spec(spec)
sys.modules["helper"] = mosaicing
spec.loader.exec_module(mosaicing)

"""Use RANSAC to robustly estimate the homography relationship between two projections
of a planar scene given a set of corresponding points
"""
def estimate_homography_ransac(y, y_prime, sample_size, num_iterations, epsilon):
    consensus_list = []
    consensus_set_list = []

    for i in np.arange(num_iterations):
        sample_indices = np.random.randint(0, len(y), sample_size, dtype=int)
        H_sample = np.linalg.inv(mosaicing.estimate_homography(list(np.asarray(y)[sample_indices]), list(np.asarray(y_prime)[sample_indices])))
        consensus = 0
        consensus_set = []
        for point_index in np.arange(len(y)):
            y_est = H_sample.dot(y_prime[point_index])
            y_est /= y_est[-1]
            if np.linalg.norm(y[point_index]-y_est) < epsilon:
                consensus += 1
                consensus_set.append(point_index)
        consensus_list.append(consensus)
        consensus_set_list.append(np.asarray(consensus_set))

    print(consensus_list)

    return mosaicing.estimate_homography(list(np.asarray(y)[consensus_set_list[np.argmax(consensus_list)]]), list(np.asarray(y_prime)[consensus_set_list[np.argmax(consensus_list)]]))

                

def main():
    # Read in images
    #img1 = cv2.imread('../images/panorama_image1_big.jpg')
    #img2 = cv2.imread('../images/panorama_image2_big.jpg')
    img1 = cv2.imread('../images/bridge1.jpg')
    img2 = cv2.imread('../images/bridge2.jpg')

    # Mark area from where keypoints are extracted, then match and filter keypoints
    number_keypoints = 200
    imgL = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    matches, kp1, kp2 = helper.perform_orb_keypoint_matching(imgL, imgR)
    matches_filtered = matches#helper.filter_keypoint_matches(matches, kp1, rect_bounds, 15)
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

    H_ransac = estimate_homography_ransac(y, y_prime, sample_size=8, num_iterations=50, epsilon = 20)
    img_mosaicing_ransac = mosaicing.perform_mosaicing_it(imgL, imgR, H_ransac, (imgL.shape[0], imgL.shape[1]+imgR.shape[1]-500))

    H_no_ransac = mosaicing.estimate_homography(y, y_prime)
    img_mosaicing_no_ransac = mosaicing.perform_mosaicing_it(imgL, imgR, H_no_ransac, (imgL.shape[0], imgL.shape[1]+imgR.shape[1]-500))

    plt.imshow(img_mosaicing_ransac, cmap="gray")
    plt.title("Image Mosaicing RANSAC")
    plt.show()

    plt.imshow(img_mosaicing_no_ransac, cmap="gray")
    plt.title("Image Mosaicing (with outliers)")
    plt.show()

if __name__ == "__main__":
    main()