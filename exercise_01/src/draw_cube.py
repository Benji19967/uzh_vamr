import logging

import matplotlib.pyplot as plt
import numpy as np

from world_to_pixel import world_to_pixel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_3D_cube_vertices(
    x_shift_num_squares: int = 0,
    y_shift_num_squares: int = 0,
    num_squares_per_edge_of_cube: int = 1,
) -> np.ndarray:
    """
    Generate 8 vertices of a cube in homogenous coordinates (X, Y, Z, 1).

    returns: (Nx4)
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

    cube_vertices = np.array(vertices)
    logger.debug(f"{cube_vertices=}")

    return cube_vertices


def draw_cube(pose_vec: np.ndarray, img_undistorted, K: np.ndarray) -> None:
    p_W_cube_vertices_hom = generate_3D_cube_vertices(
        x_shift_num_squares=3, y_shift_num_squares=2, num_squares_per_edge_of_cube=2
    )
    projected_points = world_to_pixel(
        p_W_hom=p_W_cube_vertices_hom, pose_vec=pose_vec, K=K
    )

    plt.imshow(img_undistorted, cmap="gray")
    plt.plot(
        projected_points[0],
        projected_points[1],
        "or",
        markersize=3,
    )

    projected_points = np.transpose(projected_points)
    lw = 3
    # base layer of the cube
    plt.plot(
        projected_points[[1, 3, 7, 5, 1], 0],
        projected_points[[1, 3, 7, 5, 1], 1],
        "r-",
        linewidth=lw,
    )

    # top layer of the cube
    plt.plot(
        projected_points[[0, 2, 6, 4, 0], 0],
        projected_points[[0, 2, 6, 4, 0], 1],
        "r-",
        linewidth=lw,
    )

    # vertical lines
    plt.plot(
        projected_points[[0, 1], 0],
        projected_points[[0, 1], 1],
        "r-",
        linewidth=lw,
    )
    plt.plot(
        projected_points[[2, 3], 0],
        projected_points[[2, 3], 1],
        "r-",
        linewidth=lw,
    )
    plt.plot(
        projected_points[[4, 5], 0],
        projected_points[[4, 5], 1],
        "r-",
        linewidth=lw,
    )
    plt.plot(
        projected_points[[6, 7], 0],
        projected_points[[6, 7], 1],
        "r-",
        linewidth=lw,
    )

    plt.show()
