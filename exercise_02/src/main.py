import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from draw_camera import drawCamera
from estimate_pose_dlt import estimatePoseDLT
from plot_trajectory_3D import plotTrajectory3D
from reproject_points import reprojectPoints
from scipy.spatial.transform import Rotation


def main():
    # Load
    #    - an undistorted image
    #    - the camera matrix
    #    - detected corners
    image_idx = 1
    undist_img_path = "data/images_undistorted/img_%04d.jpg" % image_idx
    undist_img = cv2.imread(undist_img_path, cv2.IMREAD_GRAYSCALE)

    # (3xN)
    p_W_corners = (0.01 * np.loadtxt("data/p_W_corners.txt", delimiter=",")).T
    K = np.loadtxt("data/K.txt")

    # Load the 2D projected points that have been detected on the
    # undistorted image into an array
    # (2xN)
    p_P_corners = (
        np.loadtxt("data/detected_corners.txt")[image_idx - 1, :].reshape(-1, 2).T
    )

    # Now that we have the 2D <-> 3D correspondances let's find the camera pose
    # with respect to the world using the DLT algorithm
    M_tilde = estimatePoseDLT(p_P=p_P_corners, p_W=p_W_corners, K=K)

    # Plot the original 2D points and the reprojected points on the image
    # (2xN)
    p_P_corners_reproj = reprojectPoints(p_W=p_W_corners, M_tilde=M_tilde, K=K)

    plt.figure()
    plt.imshow(undist_img, cmap="gray")
    plt.scatter(p_P_corners[0, :], p_P_corners[1, :], marker="o")
    plt.scatter(p_P_corners_reproj[0, :], p_P_corners_reproj[1, :], marker="+")
    plt.show()

    # Make a 3D plot containing the corner positions and a visualization
    # of the camera axis
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(p_W_corners[0, :], p_W_corners[1, :], p_W_corners[2, :])
    plt.show()

    # Position of the camera given in the world frame
    R_C_W = M_tilde[:3, :3]
    t_C_W = M_tilde[:3, 3]
    rotMat = R_C_W.T
    pos = -R_C_W.T @ t_C_W

    drawCamera(ax, pos, rotMat, length_scale=0.1, head_size=10)
    plt.show()


def main_video():
    # (3xN)
    p_W_corners = (0.01 * np.loadtxt("data/p_W_corners.txt", delimiter=",")).T
    K = np.loadtxt("data/K.txt")

    all_pts_2d = np.loadtxt("data/detected_corners.txt")
    num_images = all_pts_2d.shape[0]
    translations = np.zeros((num_images, 3))
    quaternions = np.zeros((num_images, 4))

    for idx in range(num_images):
        # (2xN)
        p_P_corners = np.reshape(all_pts_2d[idx, :], (-1, 2)).T
        M_tilde_dst = estimatePoseDLT(p_P=p_P_corners, p_W=p_W_corners, K=K)

        R_C_W = M_tilde_dst[:3, :3]
        t_C_W = M_tilde_dst[:3, 3]
        quaternions[idx, :] = Rotation.from_matrix(R_C_W.T).as_quat()
        translations[idx, :] = -R_C_W.T @ t_C_W

    fps = 30
    filename = "../motion.avi"
    plotTrajectory3D(fps, filename, translations, quaternions, p_W_corners.T)


if __name__ == "__main__":
    main()
    main_video()
