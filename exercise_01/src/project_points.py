import logging

import numpy as np
from distort_points import distort_points

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def project_points(
    p_C: np.ndarray, K: np.ndarray, D: np.ndarray | None = None
) -> np.ndarray:
    """
    Projects 3d points from the camera frame to the image plane, given the camera matrix.
    If distortion coefficients as provided, apply distortion.

    Args:
        p_C: 3d points in camera frame (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points in pixel coordinates (2xN)
    """
    projected_points = np.matmul(K, p_C)
    projected_points = projected_points[:2, :] / projected_points[2]

    if D is not None:
        projected_points_distorted = distort_points(x=projected_points, K=K, D=D)
        logger.debug(f"{projected_points_distorted=}")
        return projected_points_distorted

    logger.debug(f"{projected_points=}")
    return projected_points
