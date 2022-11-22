# Triangulation

## Tasks

1. Take two views from your calibrated set-up, estimate the 3D coordinates of two points (ask the user to click on the conjugate pixels), and measure the euclidean distance between these points (if the calibration object is fixed the global reference system is coherent for the two views).

2. Take the two views from the previous excercise and draw epipolar lines on the right image of arbitrary points from the left image.

## Solutions

1. The file triangulation.py contains the function to estimate the coordinates of a point in the 3D scene given the coordinates of the corresponding projections and the projection matrices of the scenes. It also includes a short demo in which the distance of two points indicated by the user is estimated.

2. Running epipolar_geometry.py the user can specify points on the left image in a first step. These points are then drawn onto the image on the right.
![epipolar_line](https://github.com/kroppel/computer-vision-lab/blob/main/images/epipolar_line_README.PNG)
