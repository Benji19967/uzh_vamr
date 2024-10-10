import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import utils
from pose_vector_to_transformation_matrix import pose_vector_to_transformation_matrix
from project_points import project_points

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


X_INDEX = 0
Y_INDEX = 1
IMG_CMAP = "gray"
MARKER_SIZE = 3

NUM_POINTS_BOARD_X = 9
NUM_POINTS_BOARD_Y = 6
EDGE_LENGTH_BOARD_METERS = 0.04


def generate_3D_corner_positions() -> np.ndarray:
    """
    Corner positions in checkerboard as homogeneous coordinates.

    Returns:
        M: matrix of corners of the checkerboard as 3D points (X, Y, Z) expressed
        in the world coordinate system (Nx3).
    """
    nx, ny = (NUM_POINTS_BOARD_X, NUM_POINTS_BOARD_Y)
    x_arr = np.linspace(0, (nx - 1) * EDGE_LENGTH_BOARD_METERS, nx)
    y_arr = np.linspace(0, (ny - 1) * EDGE_LENGTH_BOARD_METERS, ny)
    p_W_corners_hom = np.array([[x, y, 0, 1] for x in x_arr for y in y_arr])
    logger.debug(f"{p_W_corners_hom=}")

    return p_W_corners_hom


def project_and_superimpose_corners_onto_undistorted_img(
    poses_vec: np.ndarray, K: np.ndarray, img_undistorted
) -> None:
    """
    Project the corners of the checkerboard from the world frame to the camera frame
    and superimpose them onth the undistorted image.
    """
    p_W_corners_hom = generate_3D_corner_positions()

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame
    T_C_W = pose_vector_to_transformation_matrix(poses_vec[0])

    # transform 3d points from world to current camera pose
    p_C_corners = np.matmul(T_C_W[:3, :], np.transpose(p_W_corners_hom))
    projected_points = project_points(p_C_corners, K)
    logger.debug(f"{projected_points=}")

    plt.imshow(img_undistorted, cmap=IMG_CMAP)
    plt.plot(
        projected_points[X_INDEX],
        projected_points[Y_INDEX],
        "or",
        markersize=MARKER_SIZE,
    )
    plt.show()
