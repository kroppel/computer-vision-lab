# Camera Calibration

## Tasks

1. Implement the direct method for camera calibration using an arbitrary calibration object.

2. Perfom the tutorial on calibration using the [camera calibration toolbox](http://robots.stanford.edu/cs223b04/JeanYvesCalib/): run the [first calibration example](http://robots.stanford.edu/cs223b04/JeanYvesCalib/htmls/example.html) using the images available on the website.

3. Analyse the [data structure of the camera calibration toolbox](http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html) and use the output of the calibration to project arbitrary 3D points to arbitrary image among the images used for calibration. Generate an animation moving objects (i.e., points) on the 3D space and see how they are correctly projected to arbitrary 2D images using the proper perspective matrix. 

## Solutions

1. The direct method for camera calibration is implemented in calibration_direct_method.py, using a book shown in image1.jpg as the calibration object. With the projection matrix P estimated, arbitrary points from the 3D scene can be projected onto the image.  
![demo](https://github.com/kroppel/computer-vision-lab/blob/main/images/calibration_README.PNG)

2. The image calibration procedure described in the example of the calibration toolbox is implemented in calibration_toolbox_opencv.py.  

3. A short animation of projected points can be found in the directory triangulation_and_epipolar_geometry.  