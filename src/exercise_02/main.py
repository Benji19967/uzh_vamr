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
    undist_img_path = "../data/images_undistorted/img_%04d.jpg" % image_idx
    undist_img = cv2.imread(undist_img_path, cv2.IMREAD_GRAYSCALE)

    K = np.loadtxt("../data/K.txt")
    p_W_corners = 0.01 * np.loadtxt("../data/p_W_corners.txt", delimiter=",")
    num_corners = p_W_corners.shape[0]

    # Load the 2D projected points that have been detected on the
    # undistorted image into an array
    # TODO: Your code here

    # Now that we have the 2D <-> 3D correspondances let's find the camera pose
    # with respect to the world using the DLT algorithm
    # TODO: Your code here

    # Plot the original 2D points and the reprojected points on the image
    # TODO: Your code here

    """ Remove this comment if you have completed the code until here
    plt.figure()
    plt.imshow(undist_img, cmap = "gray")
    plt.scatter(pts_2d[:,0], pts_2d[:,1], marker = 'o')
    plt.scatter(p_reproj[:,0], p_reproj[:,1], marker = '+')
    """

    # Make a 3D plot containing the corner positions and a visualization
    # of the camera axis
    """ Remove this comment if you have completed the code until here
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p_W_corners[:,0], p_W_corners[:,1], p_W_corners[:,2])
    """

    # Position of the camera given in the world frame
    # TODO: Your code here

    """ Remove this comment if you have completed the code until here
    drawCamera(ax, pos, rotMat, length_scale = 0.1, head_size = 10)
    plt.show()
    """


def main_video():
    K = np.loadtxt("../data/K.txt")
    p_W_corners = 0.01 * np.loadtxt("../data/p_W_corners.txt", delimiter=",")
    num_corners = p_W_corners.shape[0]

    all_pts_2d = np.loadtxt("../data/detected_corners.txt")
    num_images = all_pts_2d.shape[0]
    translations = np.zeros((num_images, 3))
    quaternions = np.zeros((num_images, 4))

    # TODO: Your code here

    """ Remove this comment if you have completed the code until here
    fps = 30
    filename = "../motion.avi"
    plotTrajectory3D(fps, filename, translations, quaternions, p_W_corners)
    """


if __name__ == "__main__":
    main()
    """ Remove this comment if you have completed the code until here
    main_video()
    """
