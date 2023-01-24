import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

"""Read out data from cnn file
"""
def read_cnn_data(path):
    file = open(path, 'r')
    data = []

    for line in file:
        data.append(np.array(line.strip("\n").split(" ")))
        
    return np.array(data).astype(int)

"""Match points of two lists pairwise non-exclusively based on distance.

Params:
    data1 (list):   first input list of points
    data2 (list):   second input list of points

Returns:
    matches (np.ndarray):   Array of indices matching points from the smaller of the two input lists
                            to points in the larger input list

Note:
    If data1 has more elements than data2, the function calls itself recursively and switches the
    inputs to ensure that data1 always has less or an equal amount of elements.
"""
def pair_closest_points(data1, data2):
    if data1.shape[0] <= data2.shape[0]:
        matches = np.ndarray((data1.shape[0]), dtype=int)
        for i in np.arange(data1.shape[0]):
            match = np.argmin(np.linalg.norm(data2-data1[i,:], axis=1))
            matches[i] = match

        return matches
    else:
        return pair_closest_points(data2, data1)

"""Apply a rigid transformation F(x) = s(Rx+T) to a set of 3D data points

Params:
    data (np.ndarray):  Matrix of data points, coordinates are given along axis 1     
    s (float):          Scale factor of the rigid transform
    R (np.ndarray):     Rotation component of the rigid transform
    T (np.ndarray):     Translation component of the rigid transform

Returns:
    dataT (np.ndarray): The transformed data set
"""
def apply_rigid_transform(data, s, R, t):
    dataT = s*(np.dot(R, data.transpose())+t).transpose()

    return dataT

"""Perform iterative closest point by estimating pairs of corresponding points between two given sets of data points,
and then estimating the ideal transformation (Rotation + translation) to minimize the squared error between the two datasets
"""
def iterative_closest_point(data1, data2, max_iter = 20):
    R_full, t_full, s_full = np.identity(3), np.zeros((3,1)), 1
    
    for i in np.arange(max_iter):
        matches = pair_closest_points(data1, data2)
        error = np.sum(np.linalg.norm(data1-data2[matches,:], axis=1), axis=0)/data1.shape[0]

        # Estimate new s, R, T
        data1_mean = np.sum(data1, axis=0)/data1.shape[0]
        data2_mean = np.sum(data2[matches,:], axis=0)/data2[matches,:].shape[0]
        data1_centered = data1-data1_mean
        data2_centered = data2[matches,:]-data2_mean
        s = np.sum(np.linalg.norm(data1_centered, axis=1)/np.linalg.norm(data2_centered, axis=1))/data1_centered.shape[0]

        U,S,VH = np.linalg.svd(s*data2_centered.transpose().dot(data1_centered))
        det_UT_V = np.linalg.det(VH.transpose())*np.linalg.det(U.transpose())
        D = np.zeros((3,3))
        D[0,0] = 1
        D[1,1] = 1
        D[2,2] = det_UT_V

        R = VH.transpose().dot(D.dot(U.transpose()))
        t = (data1_mean-R.dot(data2_mean))[:,np.newaxis]/s
        
        # Apply estimated transformation and calculate error
        data2 = apply_rigid_transform(data2, s, R, t)

        # calculate full orientation
        R_full = R.dot(s_full*R_full)
        t_full = R.dot(s_full*t_full)+t
        s_full = s
    
    print("Error: {}".format(error))

    return data2, R_full, t_full, s_full

def main():
    data1 = read_cnn_data("../images/data/a4000001.cnn")
    #data2 = read_cnn_data("../images/data/a4000007.cnn")
    # For demo purposes you can also look at two identical datasets, that just differ by a shift and/or rotation
    data2 = data1+50

    # ensure data1 is always smaller/equal than/to data2 
    if data1.shape[0] > data2.shape[0]:
        tmp = data1
        data1 = data2
        data2 = tmp
    
    # run and plot data
    samples1 = np.arange(0, data1.shape[0],2)
    samples2 = np.arange(0, data2.shape[0],2)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=82, elev=34)
    ax.set_xlabel('X Axis'), ax.set_ylabel('Y Axis'), ax.set_zlabel('Z Axis')
    ax.scatter(data1[samples1,0], data1[samples1,1], data1[samples1,2], "r")
    ax.scatter(data2[samples2,0], data2[samples2,1], data2[samples2,2], "b")
    plt.show()
    
    R_full, t_full, s_full = np.identity(3), np.zeros((3,1)), 1
    # perform 4 x 10 iterations of icp and plot each intermediate result
    for i in np.arange(4):
        data2, R, t, s = iterative_closest_point(data1, data2, 10)
        # calculate full orientation
        R_full = R.dot(s_full*R_full)
        t_full = R.dot(s_full*t_full)+t
        s_full = s

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(azim=82, elev=34)
        ax.set_xlabel('X Axis'), ax.set_ylabel('Y Axis'), ax.set_zlabel('Z Axis')
        ax.scatter(data1[samples1,0], data1[samples1,1], data1[samples1,2], "r")
        ax.scatter(data2[samples2,0], data2[samples2,1], data2[samples2,2], "b")
        plt.show()

    print("Full Orientation Estimation:")
    print(R_full)
    print(t_full)
    print(s_full)

    data1 = read_cnn_data("../images/data/a4000001.cnn")
    data2 = read_cnn_data("../images/data/a4000007.cnn")
    # For demo purposes you can also look at two identical datasets, that just differ by a shift and/or rotation
    data2 = data1+50

    # ensure data1 is always smaller/equal than/to data2 
    if data1.shape[0] > data2.shape[0]:
        tmp = data1
        data1 = data2
        data2 = tmp

    data2 = apply_rigid_transform(data2, s_full, R_full, t_full)

    # run and plot data
    samples1 = np.arange(0, data1.shape[0],2)
    samples2 = np.arange(0, data2.shape[0],2)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=82, elev=34)
    ax.set_xlabel('X Axis'), ax.set_ylabel('Y Axis'), ax.set_zlabel('Z Axis')
    ax.scatter(data1[samples1,0], data1[samples1,1], data1[samples1,2], "r")
    ax.scatter(data2[samples2,0], data2[samples2,1], data2[samples2,2], "b")
    plt.show()

    
if __name__ == "__main__":
    main()