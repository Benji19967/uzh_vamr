import numpy as np
from code_previous_exercises.estimate_pose_dlt import estimatePoseDLT
from code_previous_exercises.projectPoints import projectPoints

NUM_ITERATIONS = 2000
PIXEL_TOLERANCE = 10
NUM_SAMPLES = 6


def ransacLocalization(
    matched_query_keypoints: np.ndarray,
    corresponding_landmarks: np.ndarray,
    K: np.ndarray,
):
    """
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.

    :param matched_query_keypoints: (2, N)
    :param corresponding_landmarks: (3, N)
    :param K: camera matrix intrinsics

    where N is the number of keypoints

    :returns:
        - R_C_W
        - t_C_W
        - best_inlier_mask: (1, num_matched) False (outlier) / True (inlier)
        - max_num_inliers_history
        - num_iteration_history
    """
    # TODO: compare with provided solution

    num_matched_keypoints = matched_query_keypoints.shape[1]
    # Initialize RANSAC
    best_inlier_mask = np.zeros(num_matched_keypoints)

    # (row, col) to (u, v)
    matched_query_keypoints = np.flip(matched_query_keypoints, axis=0)

    max_num_inliers_history = []
    num_iteration_history = []
    max_num_inliers = 0

    # RANSAC
    for _ in range(NUM_ITERATIONS):
        # Model from k samples (DLT or P3P)
        indices = np.random.choice(
            np.arange(num_matched_keypoints), size=NUM_SAMPLES, replace=False
        )
        landmark_sample = corresponding_landmarks[:, indices]
        keypoint_sample = matched_query_keypoints[:, indices]

        M_C_W_guess = estimatePoseDLT(keypoint_sample.T, landmark_sample.T, K)
        R_C_W_guess = M_C_W_guess[:, :3]
        t_C_W_guess = M_C_W_guess[:, -1]

        # Count inliers
        C_landmarks = (
            np.matmul(R_C_W_guess, corresponding_landmarks.T[:, :, None]).squeeze(-1)
            + t_C_W_guess[None, :]
        )
        projected_points = projectPoints(C_landmarks, K).T
        difference = matched_query_keypoints - projected_points
        errors = (difference**2).sum(0)
        is_inlier = errors < PIXEL_TOLERANCE**2

        min_inlier_count = 6

        if is_inlier.sum() > max_num_inliers and is_inlier.sum() >= min_inlier_count:
            max_num_inliers = is_inlier.sum()
            best_inlier_mask = is_inlier

        num_iteration_history.append(NUM_ITERATIONS)
        max_num_inliers_history.append(max_num_inliers)

    if max_num_inliers == 0:
        R_C_W = None
        t_C_W = None
    else:
        M_C_W = estimatePoseDLT(
            matched_query_keypoints[:, best_inlier_mask].T,
            corresponding_landmarks[:, best_inlier_mask].T,
            K,
        )
        R_C_W = M_C_W[:, :3]
        t_C_W = M_C_W[:, -1]

    return (
        R_C_W,
        t_C_W,
        best_inlier_mask,
        max_num_inliers_history,
        num_iteration_history,
    )
