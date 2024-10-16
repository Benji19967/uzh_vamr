import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pose_vector_to_transformation_matrix(
    pose_vec: np.ndarray, use_opencv: bool = False
) -> np.ndarray:
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Uses Rodrigues' rotation formula to compute R.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """
    w = pose_vec[:3]
    t = pose_vec[3:]
    if use_opencv:
        R, _ = cv2.Rodrigues(w)
    else:
        I = np.identity(3)
        theta = np.linalg.norm(w)
        k_unit = w / theta
        k_x, k_y, k_z = k_unit
        k_cross_product_matrix = np.array(
            [
                [0, -k_z, k_y],
                [k_z, 0, -k_x],
                [-k_y, k_x, 0],
            ]
        )
        K = k_cross_product_matrix

        R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    T = np.c_[R, t]
    T = np.r_[T, [[0, 0, 0, 1]]]

    logger.debug(f"{T=}")

    return T
