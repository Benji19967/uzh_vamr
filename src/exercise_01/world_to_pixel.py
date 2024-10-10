import numpy as np
from pose_vector_to_transformation_matrix import pose_vector_to_transformation_matrix
from project_points import project_points


def world_to_pixel(
    p_W_hom: np.ndarray,
    pose_vec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perspective Projection from World to Pixel coordinates.

    Args:
        p_W_hom: 3d points in World frame and as homogeneous coordinates (4xN)
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """
    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame
    T_C_W = pose_vector_to_transformation_matrix(pose_vec)

    # transform 3d points from world to current camera pose
    p_C_corners = np.matmul(T_C_W[:3, :], np.transpose(p_W_hom))
    projected_points = project_points(points_3d=p_C_corners, K=K, D=D)

    return projected_points
