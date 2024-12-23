import numpy as np


def _build_measurement_matrix_Q(p_norm: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    p  [n x 3] array containing the normalized coordinates of the 2D points
    """
    num_corners = p_norm.shape[0]
    Q = np.zeros((2 * num_corners, 12))

    for i in range(num_corners):
        u = p_norm[i, 0]
        v = p_norm[i, 1]

        Q[2 * i, 0:3] = P[i, :]
        Q[2 * i, 3] = 1
        Q[2 * i, 8:11] = -u * P[i, :]
        Q[2 * i, 11] = -u

        Q[2 * i + 1, 4:7] = P[i, :]
        Q[2 * i + 1, 7] = 1
        Q[2 * i + 1, 8:11] = -v * P[i, :]
        Q[2 * i + 1, 11] = -v

    return Q


def estimatePoseDLT(p, P, K):
    """
    Estimates the pose of a camera using a set of 2D-3D correspondences
    and a given camera matrix.

    p  [n x 2] array containing the undistorted coordinates of the 2D points
    P  [n x 3] array containing the 3D point positions
    K  [3 x 3] camera matrix

    Returns a [3 x 4] projection matrix of the form
              M_tilde = [R_tilde | alpha * t]
    where R is a rotation matrix. M_tilde encodes the transformation
    that maps points from the world frame to the camera frame
    """
    num_points = p.shape[0]

    # Convert 2D to normalized coordinates
    p_homogeneous = np.r_[p.T, np.ones((1, num_points))].T
    p_normalized = (np.linalg.inv(K) @ p_homogeneous.T).T

    # Build measurement matrix Q
    # P_homogeneous = np.r_[P.T, np.ones(num_points)].T
    # q1 = np.kron(P_homogeneous, [[1], [0]])
    # q2 = np.kron(P_homogeneous, [[0], [1]])
    # q3 = np.kron(P_homogeneous, p)

    Q = _build_measurement_matrix_Q(p_norm=p_normalized, P=P)

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    u, s, vh = np.linalg.svd(Q)
    M_tilde = vh.T[:, -1].reshape((3, 4))

    # Extract [R | t] with the correct scale
    if np.linalg.det(M_tilde[:, :3]) < 0:
        M_tilde *= -1

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    R_hat = M_tilde[:, :3]
    u, s, vh = np.linalg.svd(R_hat)
    R_tilde = u @ vh

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    alpha = np.linalg.norm(R_tilde, "fro") / np.linalg.norm(R_hat, "fro")

    # Build M_tilde with the corrected rotation and scale
    M_tilde = np.c_[R_tilde, alpha * M_tilde[:, -1]]

    return M_tilde
