import logging
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from pose_vector_to_transformation_matrix import pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_3D_corner_positions() -> np.ndarray:
    """
    Corner positions in checkerboard.

    Returns:
        M: matrix of corners of the checkerboard as 3D points (X, Y, Z) expressed
        in the world coordinate system (Nx3)
    """
    nx, ny = (9, 6)
    x_arr = np.linspace(0, 0.32, nx)
    y_arr = np.linspace(0, 0.20, ny)
    matrix = [[x, y, 0, 1] for x in x_arr for y in y_arr]
    return np.array(matrix)


def map_corners_onto_undistorted_img() -> None:
    """
    Map the corners of of the checkerboard onto the undistorted image
    """
    poses = utils.load_poses("./data/poses.txt")

    K, D = utils.load_camera_intrinsics("./data/K.txt", "./data/D.txt")
    logger.debug(f"K:\n{K}")
    logger.debug(f"D:\n{D}")

    corners_world_homogenous = generate_3D_corner_positions()
    logger.debug(f"corners_world_homogenous: {corners_world_homogenous}")

    img_undistorted = utils.load_img("./data/images_undistorted/img_0001.jpg")

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame
    transformation_matrix = pose_vector_to_transformation_matrix(poses[1])
    logger.debug(f"Transformation matrix:\n{transformation_matrix}")

    # transform 3d points from world to current camera pose
    # projected_points = project_points(corners, K, D)
    corners_camera_homogenous = np.matmul(
        K,
        np.matmul(transformation_matrix[:3, :], np.transpose(corners_world_homogenous)),
    )
    logger.debug(f"Corners camera:\n{corners_camera_homogenous}")

    plt.imshow(img_undistorted, cmap="gray")
    plt.plot(
        corners_camera_homogenous[0] / corners_camera_homogenous[2],
        corners_camera_homogenous[1] / corners_camera_homogenous[2],
        "or",
        markersize=3,
    )
    plt.show()
