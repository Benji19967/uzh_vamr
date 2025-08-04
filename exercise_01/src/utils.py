import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_poses(filename: str) -> np.ndarray:
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

    Note: shape is (HxW) such that x, y in a traditional Cartesian coordinate
    system are inverted to (y, x).
    """
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


class Timer:
    def __init__(self, label: str = "Elapsed time"):
        self.label = label
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return (
            self  # Optional, useful if you want to access timer info inside the context
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        print(f"{self.label}: {elapsed:.4f} seconds")
