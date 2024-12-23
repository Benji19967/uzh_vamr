import cv2
import numpy as np
import scipy.ndimage


def extractKeypoints(
    all_diff_of_gaussians, contrast_threshold
) -> list[list[np.ndarray]]:
    """
    Returns the keypoint locations: 1 array per octave per scale.
    """
    keypoint_locations = []

    for diff_of_gaussians_in_octave in all_diff_of_gaussians:
        keypoints_for_octave = []
        for dog1, dog2, dog3 in zip(
            diff_of_gaussians_in_octave[:-2],
            diff_of_gaussians_in_octave[1:-1],
            diff_of_gaussians_in_octave[2:],
        ):
            dog_max = scipy.ndimage.maximum_filter(
                input=[dog1, dog2, dog3], size=[3, 3, 3]
            )
            dog = np.array([dog1, dog2, dog3])
            is_keypoint = (dog[1, :, :] == dog_max[1, :, :]) & (
                dog[1, :, :] >= contrast_threshold
            )
            keypoints_for_octave.append(np.array(is_keypoint.nonzero()).T)
            # print((dog.shape, dog_max.shape, is_keypoint))
        keypoint_locations.append(keypoints_for_octave)
    return keypoint_locations
