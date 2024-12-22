import numpy as np
from utils import cross2Matrix


def linearTriangulation(
    p1: np.ndarray, p2: np.ndarray, M1: np.ndarray, M2: np.ndarray
) -> np.ndarray:
    """Linear Triangulation
    Input:
     - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
     - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
     - M1 np.ndarray(3, 4): projection matrix corresponding to first image
     - M2 np.ndarray(3, 4): projection matrix corresponding to second image

    Output:
     - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    N = p1.shape[1]
    Points = np.zeros((4, N))

    for i, (p1i, p2i) in enumerate(zip(p1.T, p2.T)):
        p1ix = cross2Matrix(p1i)
        p2ix = cross2Matrix(p2i)

        a1 = p1ix @ M1
        a2 = p2ix @ M2
        A = np.r_[a1, a2]

        # vh.shape (4, 4)
        _, _, vh = np.linalg.svd(A)
        Points[:, i] = vh.T[:, -1]

        Points /= Points[3, :]

    return Points
