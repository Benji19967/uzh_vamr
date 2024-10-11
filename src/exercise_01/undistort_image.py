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
    height = img.shape[0]
    width = img.shape[1]
    undistorted_img = np.zeros((height, width))

    # for x in range(width):
    #     for y in range(height):
    #         distorted_point = distort_points(np.array([[x], [y]]), K=K, D=D)
    #         dist_x = round(distorted_point[0][0])
    #         dist_y = round(distorted_point[1][0])
    #         if (dist_x >= 0) & (dist_x < width) & (dist_y >= 0) & (dist_y < height):
    #             undistorted_img[y, x] = img[
    #                 dist_y, dist_x
    #             ]  # note how (y, x) are inverted
    # return undistorted_img

    warp_dst = cv2.warpAffine(img, K, img.shape)
    return warp_dst
