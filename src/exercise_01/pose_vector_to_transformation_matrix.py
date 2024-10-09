import math

import numpy as np


def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
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

    R = (
        I
        + math.sin(theta) * k_cross_product_matrix
        + (1 - math.cos(theta)) * (k_cross_product_matrix**2)
    )

    return np.c_[R, t]
