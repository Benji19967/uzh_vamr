import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_poses_vec(filename: str) -> np.ndarray:
    """
    Load camera poses.
    Dimensions: [736, 6]

    Each row i of matrix 'poses' contains the transformations that transforms
    points expressed in the world frame to points expressed in the camera frame.

    Each row: (w_x, w_y, w_z, t_x, t_y, t_z), where
    w: axis-angle representation
    t: translation in meters
    """
    poses_vec = np.loadtxt(filename)
    logger.debug(f"{poses_vec=}")

    return poses_vec


def load_camera_matrix(K_filename: str) -> np.ndarray:
    """
    Load camera matrix (a.k.a intrinsic parameter matrix).
    Dimensions: [3, 3]
    """
    K = np.loadtxt(K_filename)
    logger.debug(f"{K=}")

    return K


def load_distortion_coefficients(D_filename: str) -> np.ndarray:
    """
    Load distortion coefficients.
    Dimensions: [1, 2]
    """
    D = np.loadtxt(D_filename)
    logger.debug(f"{D=}")

    return D


def load_img(filename: str):
    """
    Load image as a grayscale image.

    Pixel values range from 0 (black) to 255 (white).
    """
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
