import logging
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from corners_onto_undistorted_img import project_and_superimpose_corners_onto_img
from draw_cube import draw_cube
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FILENAME_POSES_VEC = "./data/poses.txt"
FILENAME_CAMERA_MATRIX = "./data/K.txt"
FILENAME_DISTORTION_COEFFICIENTS = "./data/D.txt"
FILENAME_UNDISTORTED_IMAGE = "./data/images_undistorted/img_0001.jpg"
DIR_DISTORTED_IMAGES = "./data/images"


def get_img_distorted_filename(idx: int) -> str:
    padded_idx = f"{idx}".zfill(4)
    return DIR_DISTORTED_IMAGES + "/" + f"img_{padded_idx}.jpg"


def main():
    poses_vec = utils.load_poses_vec(filename=FILENAME_POSES_VEC)
    K = utils.load_camera_matrix(K_filename=FILENAME_CAMERA_MATRIX)
    D = utils.load_distortion_coefficients(D_filename=FILENAME_DISTORTION_COEFFICIENTS)
    img_undistorted = utils.load_img(FILENAME_UNDISTORTED_IMAGE)

    # PART 1 -- Projection
    project_and_superimpose_corners_onto_img(
        pose_vec=poses_vec[0],
        img=img_undistorted,
        K=K,
    )

    # PART 1 -- Cube
    draw_cube(
        pose_vec=poses_vec[0],
        img_undistorted=img_undistorted,
        K=K,
    )

    # PART 2
    for idx in range(1, 100):
        filename = get_img_distorted_filename(idx)
        img_distorted = utils.load_img(filename)
        project_and_superimpose_corners_onto_img(
            pose_vec=poses_vec[idx - 1],
            img=img_distorted,
            K=K,
            D=D,
        )

    # undistort image with bilinear interpolation
    """Remove this comment if you have completed the code until here
    start_t = time.time()
    img_undistorted = undistort_image(img, K, D, bilinear_interpolation=True)
    print('Undistortion with bilinear interpolation completed in {}'.format(
        time.time() - start_t))

    # vectorized undistortion without bilinear interpolation
    start_t = time.time()
    img_undistorted_vectorized = undistort_image_vectorized(img, K, D)
    print('Vectorized undistortion completed in {}'.format(
        time.time() - start_t))

    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_undistorted, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('With bilinear interpolation')
    axs[1].imshow(img_undistorted_vectorized, cmap='gray')
    axs[1].set_axis_off()
    axs[1].set_title('Without bilinear interpolation')
    plt.show()
    """


if __name__ == "__main__":
    main()
