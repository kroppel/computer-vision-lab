import numpy as np
import importlib.util
import sys
import cv2
spec = importlib.util.spec_from_file_location("", "../calibration/calibration_direct_method.py")
calibration_direct_method = importlib.util.module_from_spec(spec)
sys.modules["calibration_direct_method"] = calibration_direct_method
spec.loader.exec_module(calibration_direct_method)
spec = importlib.util.spec_from_file_location("", "../shared/helper.py")
helper = importlib.util.module_from_spec(spec)
sys.modules["helper"] = helper
spec.loader.exec_module(helper)

PERFORM_CALIBRATION = False

def main():
    # Read in images
    img1 = cv2.imread('../images/scene1.jpg')

    if PERFORM_CALIBRATION:
        helper.calibrate_book_scenes_example()

    P1, K1, R1, t1 = helper.read_projection_matrix_from_file("../images/data/params1")

    # Animation along book edges
    points_animation = []
    for i in np.arange(0, 14.5, 0.1):
        p = P1.dot(np.asarray([i,0,0,1])[:np.newaxis])
        points_animation.append(p/p[-1])
    for i in np.arange(0, 4, 0.1):
        p = P1.dot(np.asarray([14.5,0,i,1])[:np.newaxis])
        points_animation.append(p/p[-1])
    for i in np.arange(0, 14.5, 0.1):
        p = P1.dot(np.asarray([14.5-i,0,4,1])[:np.newaxis])
        points_animation.append(p/p[-1])
    for i in np.arange(0, 4, 0.1):
        p = P1.dot(np.asarray([0,0,4-i,1])[:np.newaxis])
        points_animation.append(p/p[-1])

    i = 0
    while True:
        current_point = points_animation[i]
        cpy_img = np.copy(img1)
        cv2.circle(cpy_img, (int(current_point[0]), int(current_point[1])), 3, color=(0,0,255), thickness=-1)
        cv2.imshow("animation", cpy_img)
        i = (i + 1) % len(points_animation)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()