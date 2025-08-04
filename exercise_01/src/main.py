import logging

import numpy as np
import plot
from points_generators import generate_3D_corner_positions, generate_3D_cube_vertices
from transformation_matrix_from_pose_vector import (
    transformation_matrix_from_pose_vector,
)
from transformations import world_to_pixel
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized

import utils
from utils import Timer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FILENAME_POSES = "./data/poses.txt"
FILENAME_CAMERA_MATRIX = "./data/K.txt"
FILENAME_DISTORTION_COEFFICIENTS = "./data/D.txt"
FILENAME_UNDISTORTED_IMAGE = "./data/images_undistorted/img_0001.jpg"
DIR_DISTORTED_IMAGES = "./data/images"


def get_img_distorted_filename(idx: int) -> str:
    padded_idx = f"{idx}".zfill(4)
    return DIR_DISTORTED_IMAGES + "/" + f"img_{padded_idx}.jpg"


def compute_transformation_matrices(poses: np.ndarray) -> list[np.ndarray]:
    """
    Computes T_C_W for each image
    """
    T_list = []
    for pose in poses:
        T = transformation_matrix_from_pose_vector(pose)
        T_list.append(T)
    return T_list


# TODO: Add CLI using Typer
def main():
    poses = utils.load_poses(filename=FILENAME_POSES)

    # TODO: is T_C_W a misnomer? Is it really the name of the transformation matrix
    # or does it signify, generally, transformation from W to C
    T_C_Ws = compute_transformation_matrices(poses)
    K = utils.load_camera_matrix(K_filename=FILENAME_CAMERA_MATRIX)
    D = utils.load_distortion_coefficients(D_filename=FILENAME_DISTORTION_COEFFICIENTS)
    img_undistorted = utils.load_img(FILENAME_UNDISTORTED_IMAGE)

    p_W_hom_corners = generate_3D_corner_positions()

    # PART 1 -- Projection
    p_P_corners = world_to_pixel(
        p_W_hom=p_W_hom_corners,
        T_C_W=T_C_Ws[0],
        K=K,
    )
    plot.points_on_image(img=img_undistorted, points=p_P_corners, img_idx=1)

    # PART 1 -- Cube
    p_W_hom_cube_vertices = generate_3D_cube_vertices(
        x_shift_num_squares=3, y_shift_num_squares=2, num_squares_per_edge_of_cube=2
    )
    p_P_cube_vertices = world_to_pixel(
        p_W_hom=p_W_hom_cube_vertices,
        T_C_W=T_C_Ws[0],
        K=K,
    )
    plot.draw_cube(img=img_undistorted, vertices=p_P_cube_vertices)

    # PART 2
    for idx in range(1, 20):
        filename = get_img_distorted_filename(idx)
        img_distorted = utils.load_img(filename)

        p_P_corners = world_to_pixel(
            p_W_hom=p_W_hom_corners,
            T_C_W=T_C_Ws[idx - 1],
            K=K,
            D=D,
        )
        plot.points_on_image(img=img_distorted, points=p_P_corners, img_idx=idx)

    # PART Image undistortion
    img_distorted_filename = get_img_distorted_filename(idx=1)
    img_distorted = utils.load_img(img_distorted_filename)

    # undistort image with bilinear interpolation
    with Timer(label="Undistortion with bilinear interpolation"):
        img_undistorted = undistort_image(
            img_distorted, K, D, bilinear_interpolation=True
        )
    # vectorized undistortion without bilinear interpolation
    with Timer(label="Vectorized undistortion"):
        img_undistorted_vectorized = undistort_image_vectorized(img_distorted, K, D)

    plot.two_images(
        img1=img_undistorted,
        img2=img_undistorted_vectorized,
        title1="With bilinear interpolation",
        title2="Without bilinear interpolation",
    )


if __name__ == "__main__":
    main()
