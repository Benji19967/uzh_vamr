import logging

import numpy as np
from distort_points import distort_points

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def undistort_image_vectorized(
    img: np.ndarray, K: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """
    Undistorts an image using the camera matrix and distortion coefficients.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        undistorted_img: undistorted image (HxW)
    """
    height, width = img.shape

    img_points = []
    for x in range(width):
        for y in range(height):
            img_points.append((x, y))
    img_points = np.transpose(np.array(img_points))

    distorted_points = distort_points(x=img_points, D=D, K=K)
    logger.debug(f"{distorted_points=}")

    undistorted_img = img[
        np.round(distorted_points[1, :].astype(np.int32)),
        np.round(distorted_points[0, :].astype(np.int32)),
    ]
    undistorted_img = undistorted_img.reshape(img.shape, order="F").astype(
        np.uint32
    )  # there are several ways to reshape! (see order="C" vs order="F")
    logger.debug(f"{undistorted_img=}")

    return undistorted_img
