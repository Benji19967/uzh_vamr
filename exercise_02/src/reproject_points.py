import numpy as np


def reprojectPoints(P, M_tilde, K):
    """
    Reproject 3D points given a projection matrix

    P         [n x 3] coordinates of the 3d points in the world frame
    M_tilde   [3 x 4] projection matrix
    K         [3 x 3] camera matrix

    Returns [n x 2] coordinates of the reprojected 2d points
    """
    num_points = P.shape[0]
    P_homogeneous = np.r_[P.T, np.ones((1, num_points))].T

    reprojected_points = K @ M_tilde @ P_homogeneous.T
    return (reprojected_points[:2, :] / reprojected_points[2, :]).T
