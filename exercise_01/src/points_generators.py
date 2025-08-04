import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NUM_POINTS_BOARD_X = 9
NUM_POINTS_BOARD_Y = 6
EDGE_LENGTH_BOARD_METERS = 0.04


def generate_3D_corner_positions() -> np.ndarray:
    """
    Generate corner positions in checkerboard as homogeneous coordinates.

    Returns:
        M: matrix of corners of the checkerboard as 3D points (X, Y, Z) expressed
        in the world coordinate system as homogeneous coordinates (4xN).
    """
    nx, ny = (NUM_POINTS_BOARD_X, NUM_POINTS_BOARD_Y)
    num_points = nx * ny
    x_arr = np.linspace(0, (nx - 1) * EDGE_LENGTH_BOARD_METERS, nx)
    y_arr = np.linspace(0, (ny - 1) * EDGE_LENGTH_BOARD_METERS, ny)
    p = np.array(np.meshgrid(x_arr, y_arr)).reshape(2, -1)
    p_W_corners_hom = np.r_[p, np.zeros((1, num_points)), np.ones((1, num_points))]
    logger.debug(f"{p_W_corners_hom=}")

    return p_W_corners_hom


def generate_3D_cube_vertices(
    x_shift_num_squares: int = 0,
    y_shift_num_squares: int = 0,
    num_squares_per_edge_of_cube: int = 1,
) -> np.ndarray:
    """
    Generate 8 vertices of a cube in homogenous coordinates (X, Y, Z, 1).

    returns: (4xN)
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

    p_W_hom_cube_vertices = np.transpose(np.array(vertices))
    logger.debug(f"{p_W_hom_cube_vertices=}")
    print(p_W_hom_cube_vertices)

    return p_W_hom_cube_vertices
