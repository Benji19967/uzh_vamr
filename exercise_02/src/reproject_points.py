import numpy as np


def reprojectPoints(p_W, M_tilde, K):
    """
    Reproject 3D points given a projection matrix.

    From World frame to Pixel frame.

    P         [3 x n] coordinates of the 3d points in the world frame
    M_tilde   [3 x 4] projection matrix
    K         [3 x 3] camera matrix

    Returns:
        p_P [2 x n] coordinates in the pixel frame of the reprojected 2d points
    """
    num_points = p_W.shape[1]
    p_W_hom = np.r_[p_W, np.ones((1, num_points))]

    u_v_lambda = K @ M_tilde @ p_W_hom
    p_P = u_v_lambda[:2, :] / u_v_lambda[2, :]
    return p_P
