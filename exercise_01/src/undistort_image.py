import cv2
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
    # TODO: can you use cv2.warpAffine to achieve the same?

    height = img.shape[0]
    width = img.shape[1]
    undistorted_img = np.zeros((height, width))

    for x in range(width):
        for y in range(height):
            distorted_point = distort_points(np.array([[x], [y]]), K=K, D=D)
            u = distorted_point[0][0]
            v = distorted_point[1][0]
            u1 = round(u)
            v1 = round(v)
            if bilinear_interpolation:
                a = u - u1
                b = v - v1
                if (u1 >= 0) & (u1 + 1 < width) & (v1 >= 0) & (v1 + 1 < height):
                    undistorted_img[y, x] = (1 - b) * (
                        (1 - a) * img[v1, u1] + a * img[v1, u1 + 1]
                    ) + b * ((1 - a) * img[v1 + 1, u1] + a * img[v1 + 1, u1 + 1])

            else:
                if (u1 >= 0) & (u1 < width) & (v1 >= 0) & (v1 < height):
                    undistorted_img[y, x] = img[v1, u1]  # note how (y, x) are inverted

    return undistorted_img
