import math

import numpy as np
from distort_points import distort_points


def undistort_image(
    img: np.ndarray, K: np.ndarray, D: np.ndarray, bilinear_interpolation: bool = False
) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """
    undistorted_img = np.zeros(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            distorted_point = distort_points(np.array([[x], [y]]), K=K, D=D)
            dist_x = round(distorted_point[0][0])
            dist_y = round(distorted_point[1][0])
            undistorted_img[x, y] = img[dist_x, dist_y]
    return undistorted_img
