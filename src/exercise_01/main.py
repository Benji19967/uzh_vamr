import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pose_vector_to_transformation_matrix import pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized


def load_poses(filename: str) -> np.ndarray:
    """
    Load camera poses. Dimensions: [736, 6]

    Each row i of matrix 'poses' contains the transformations that transforms
    points expressed in the world frame to points expressed in the camera frame.

    Each row: (w_x, w_y, w_z, t_x, t_y, t_z), where
    w: axis-angle representation
    t: translation in meters
    """

    return np.loadtxt(filename, delimiter=",")


def load_camera_intrinsics(
    K_filename: str, D_filename: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.loadtxt(K_filename, delimiter=","),
        np.loadtxt(D_filename, delimiter=","),
    )


def generate_3D_corner_positions() -> np.ndarray:
    """
    Corner positions in checkerboard.

    Returns:
        M: matrix of corners of the checkerboard as 3D points (X, Y, Z) expressed
        in the world coordinate system (Nx3)
    """
    nx, ny = (9, 6)
    x_arr = np.linspace(0, 32, nx)
    y_arr = np.linspace(0, 20, ny)
    matrix = [[x, y, 0] for x in x_arr for y in y_arr]
    return np.array(matrix)


def load_img(filename: str):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def main():
    poses = load_poses("./data/poses.txt/")

    corners = generate_3D_corner_positions()

    K, D = load_camera_intrinsics("./data/K.txt", "./data/D.txt")

    img = load_img("./data/images/img_0001.jpg")

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame
    # TODO: Your code here

    # transform 3d points from world to current camera pose
    # TODO: Your code here

    # undistort image with bilinear interpolation
    """ Remove this comment if you have completed the code until here
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

    # calculate the cube points to then draw the image
    # TODO: Your code here

    # Plot the cube
    """ Remove this comment if you have completed the code until here
    plt.clf()
    plt.close()
    plt.imshow(img_undistorted, cmap='gray')

    lw = 3

    # base layer of the cube
    plt.plot(cube_pts[[1, 3, 7, 5, 1], 0],
             cube_pts[[1, 3, 7, 5, 1], 1],
             'r-',
             linewidth=lw)

    # top layer of the cube
    plt.plot(cube_pts[[0, 2, 6, 4, 0], 0],
             cube_pts[[0, 2, 6, 4, 0], 1],
             'r-',
             linewidth=lw)

    # vertical lines
    plt.plot(cube_pts[[0, 1], 0], cube_pts[[0, 1], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[2, 3], 0], cube_pts[[2, 3], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[4, 5], 0], cube_pts[[4, 5], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[6, 7], 0], cube_pts[[6, 7], 1], 'r-', linewidth=lw)

    plt.show()
    """


if __name__ == "__main__":
    main()
