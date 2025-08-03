import numpy as np


def selectKeypoints(scores, num, r):
    """
    Selects the num best scores as keypoints and performs non-maximum supression of a (2r + 1)*(2r + 1) box around
    the current maximum.

    Returns:
        (2xN) keypoints
    """

    # keep `num` keypoints, each has 2 coordinates (u, v)
    keypoints = np.zeros([2, num])

    # scores with padding
    temp_scores = np.pad(scores, [(r, r), (r, r)], mode="constant", constant_values=0)

    for i in range(num):
        # find max value in 2 array (i, j)
        kp = np.unravel_index(temp_scores.argmax(), temp_scores.shape)

        keypoints[:, i] = np.array(kp) - r  # TODO: why `- r`? Because of padding!

        # nonmaximum-suppression
        temp_scores[(kp[0] - r) : (kp[0] + r + 1), (kp[1] - r) : (kp[1] + r + 1)] = 0

    return keypoints
