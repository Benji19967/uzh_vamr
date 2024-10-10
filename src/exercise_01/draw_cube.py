import logging

import matplotlib.pyplot as plt
import numpy as np
import utils
from pose_vector_to_transformation_matrix import pose_vector_to_transformation_matrix

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_3D_cube_vertices(
    x_shift_num_squares: int = 0,
    y_shift_num_squares: int = 0,
    num_squares_per_edge_of_cube: int = 1,
) -> np.ndarray:
    """
    Generate 8 vertices of a cube in homogenous coordinates (X, Y, Z, 1).
    """
    edge_length_board = 0.04  # (meters)
    edge_length_cube_meters = edge_length_board * num_squares_per_edge_of_cube
    x_shift_meters = edge_length_board * x_shift_num_squares
    y_shift_meters = edge_length_board * y_shift_num_squares

    vertices = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                vertices.append(
                    [
                        x_shift_meters + (x * edge_length_cube_meters),
                        y_shift_meters + (y * edge_length_cube_meters),
                        -z * edge_length_cube_meters,
                        1,
                    ]
                )

    return np.array(vertices)


def draw_cube() -> None:
    poses = utils.load_poses_vec("./data/poses.txt")
    K, D = utils.load_camera_intrinsics("./data/K.txt", "./data/D.txt")
    logger.debug(f"K:\n{K}")
    logger.debug(f"D:\n{D}")

    cube_vertices = generate_3D_cube_vertices(
        x_shift_num_squares=3, y_shift_num_squares=2, num_squares_per_edge_of_cube=2
    )
    logger.debug(f"cube_vertices: {cube_vertices}")

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
        np.matmul(transformation_matrix[:3, :], np.transpose(cube_vertices)),
    )
    corners_camera_homogenous = corners_camera_homogenous / corners_camera_homogenous[2]
    logger.debug(f"Corners camera:\n{corners_camera_homogenous}")

    plt.imshow(img_undistorted, cmap="gray")
    plt.plot(
        corners_camera_homogenous[0],
        corners_camera_homogenous[1],
        "or",
        markersize=3,
    )

    corners_camera_homogenous = np.transpose(corners_camera_homogenous)
    lw = 3
    # base layer of the cube
    plt.plot(
        corners_camera_homogenous[[1, 3, 7, 5, 1], 0],
        corners_camera_homogenous[[1, 3, 7, 5, 1], 1],
        "r-",
        linewidth=lw,
    )

    # top layer of the cube
    plt.plot(
        corners_camera_homogenous[[0, 2, 6, 4, 0], 0],
        corners_camera_homogenous[[0, 2, 6, 4, 0], 1],
        "r-",
        linewidth=lw,
    )

    # vertical lines
    plt.plot(
        corners_camera_homogenous[[0, 1], 0],
        corners_camera_homogenous[[0, 1], 1],
        "r-",
        linewidth=lw,
    )
    plt.plot(
        corners_camera_homogenous[[2, 3], 0],
        corners_camera_homogenous[[2, 3], 1],
        "r-",
        linewidth=lw,
    )
    plt.plot(
        corners_camera_homogenous[[4, 5], 0],
        corners_camera_homogenous[[4, 5], 1],
        "r-",
        linewidth=lw,
    )
    plt.plot(
        corners_camera_homogenous[[6, 7], 0],
        corners_camera_homogenous[[6, 7], 1],
        "r-",
        linewidth=lw,
    )

    plt.show()
