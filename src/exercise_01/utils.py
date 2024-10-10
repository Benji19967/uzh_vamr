import cv2
import numpy as np


def load_poses(filename: str) -> np.ndarray:
    """
    Load camera poses. Dimensions: [736, 6]

    Each row i of matrix 'poses' contains the transformations that transforms
    points expressed in the world frame to points expressed in the camera frame.

    Each row: (w_x, w_y, w_z, t_x, t_y, t_z), where
    w: axis-angle representation
    t: translation in meters
    """

    return np.loadtxt(filename)


def load_camera_intrinsics(
    K_filename: str, D_filename: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.loadtxt(K_filename),
        np.loadtxt(D_filename),
    )


def load_img(filename: str):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
