import logging

import numpy as np
from distort_points import distort_points

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def project_points(
    points_3d: np.ndarray, K: np.ndarray, D: np.ndarray | None = None
) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix.
    If distortion coefficients as provided, apply distortion.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """
    projected_points = np.matmul(K, points_3d)
    projected_points = projected_points / projected_points[2]

    if D:
        pass

    logger.debug(f"projected_points:\n{projected_points}")

    return projected_points
