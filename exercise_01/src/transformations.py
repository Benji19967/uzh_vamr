import logging

import numpy as np
from distort_points import distort_points

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def world_to_pixel(
    p_W_hom: np.ndarray,
    T_C_W: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perspective Projection from World to Pixel coordinates.

    Args:
        p_W_hom: 3d points in World frame and as homogeneous coordinates (4xN)
        T_C_W: 4x4 transformation matrix to map points from world to camera frame
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        p_P: 2d points in pixel coordinates (2xN)
    """
    p_C = world_to_camera(p_W_hom=p_W_hom, T_C_W=T_C_W)
    p_P = camera_to_pixel(p_C=p_C, K=K, D=D)
    return p_P


def world_to_camera(
    p_W_hom: np.ndarray,
    T_C_W: np.ndarray,
) -> np.ndarray:
    """
    Transformation from World to Camera frame

    Args:
        p_W_hom: 3d points in World frame and as homogeneous coordinates (4xN)
        T_C_W: 4x4 transformation matrix to map points from world to camera frame

    Returns:
        p_C: 3d points in camera frame (3xN)
    """
    p_C = np.matmul(T_C_W[:3, :], p_W_hom)

    return p_C


def camera_to_pixel(
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
        p_P: 2d points in pixel coordinates (2xN)
    """
    u_v_lambda = np.matmul(K, p_C)
    p_P = u_v_lambda[:2, :] / u_v_lambda[2]

    if D is not None:
        p_P_distorted = distort_points(x=p_P, K=K, D=D)
        logger.debug(f"{p_P_distorted=}")
        return p_P_distorted

    logger.debug(f"{p_P=}")
    return p_P
