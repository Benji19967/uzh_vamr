import numpy as np


def describeKeypoints(img: np.ndarray, keypoints, r):
    """
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
    coordinates. r is the patch "radius".
    """
    N = keypoints.shape[1]

    # `(2 * r + 1) ** 2` is the number of pixels in a patch/descriptor
    descriptors = np.zeros([(2 * r + 1) ** 2, N])
    padded = np.pad(img, [(r, r), (r, r)], mode="constant", constant_values=0)

    for i in range(N):
        kp = keypoints[:, i].astype(int) + r  # `+r` to account for padding

        # store the the pixel intensities of the descriptors in a flattened way
        descriptors[:, i] = padded[
            (kp[0] - r) : (kp[0] + r + 1), (kp[1] - r) : (kp[1] + r + 1)
        ].flatten()

    return descriptors
