import cv2
import matplotlib.pyplot as plt
import numpy as np

import utils
from compute_blurred_images import computeBlurredImages
from compute_descriptors import computeDescriptors
from compute_difference_of_gaussians import computeDifferenceOfGaussians
from compute_image_pyramid import computeImagePyramid
from extract_keypoints import extractKeypoints

# User parameters
ROTATION_INVARIANT = False  # Enable rotation invariant SIFT
ROTATION_IMG2_DEG = 60  # Rotate the second image to be matched

# SIFT parameters
CONTRAST_THRESHOLD = 0.04  # for feature matching
SIFT_SIGMA = 1.6  # SIGMA USED FOR BLURRING
RESCALE_FACTOR = 0.3  # rescale images to make it faster
NUM_SCALES = 3  # number of scales per octave
NUM_OCTAVES = 5  # number of octaves


def read_img(path: str, scale):
    """
    Convenience function to read in images into grayscale and convert them to double
    """
    return cv2.normalize(
        cv2.resize(
            cv2.imread(path, cv2.IMREAD_GRAYSCALE), (0, 0), fx=scale, fy=scale
        ).astype("float"),
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX,
    )


def main(
    rotation_invariant,
    rotation_img2_deg,
    contrast_threshold,
    sift_sigma,
    rescale_factor,
    num_scales,
    num_octaves,
):

    # Read in images
    img1 = read_img("data/img_1.jpg", rescale_factor)
    img2 = read_img("data/img_2.jpg", rescale_factor)

    # utils.show_np_array_as_img(img1)
    # utils.show_np_array_as_img(img2)

    # If we want to test our rotation invariant features, rotate the second image
    if ROTATION_INVARIANT:
        if np.abs(rotation_img2_deg) > 1e-6:
            pass
            # Lets go and rotate the image
            # - get the original height and width
            # - create rotation matrix
            # - calculate the size of the rotated image
            # - pad the image
            # - rotate the image
        # TODO: Your code here

    # Actually compute the SIFT features.
    # For both images do:
    # - construct the image pyramid
    image_pyramid_1 = computeImagePyramid(img=img1, num_octaves=NUM_OCTAVES)
    image_pyramid_2 = computeImagePyramid(img=img2, num_octaves=NUM_OCTAVES)

    # - compute the blurred images
    blurred_images_1 = computeBlurredImages(
        image_pyramid=image_pyramid_1, num_scales=NUM_SCALES, sift_sigma=SIFT_SIGMA
    )
    blurred_images_2 = computeBlurredImages(
        image_pyramid=image_pyramid_2, num_scales=NUM_SCALES, sift_sigma=SIFT_SIGMA
    )

    # - compute difference of gaussians
    difference_of_gaussians_1 = computeDifferenceOfGaussians(
        blurred_images=blurred_images_1
    )
    difference_of_gaussians_2 = computeDifferenceOfGaussians(
        blurred_images=blurred_images_2
    )

    # - extract the keypoints
    keypoint_locations = (
        extractKeypoints(
            all_diff_of_gaussians=difference_of_gaussians_1,
            contrast_threshold=CONTRAST_THRESHOLD,
        ),
        extractKeypoints(
            all_diff_of_gaussians=difference_of_gaussians_2,
            contrast_threshold=CONTRAST_THRESHOLD,
        ),
    )

    # - compute the descriptors
    keypoint_descriptors = (
        computeDescriptors(
            blurred_images=blurred_images_1,
            keypoint_locations=keypoint_locations[0],
            rotation_invariant=ROTATION_INVARIANT,
        ),
        computeDescriptors(
            blurred_images=blurred_images_2,
            keypoint_locations=keypoint_locations[1],
            rotation_invariant=ROTATION_INVARIANT,
        ),
    )

    imgs = [img1, img2]
    keypoint_locations = []
    keypoint_descriptors = []

    for i in range(len(imgs)):
        pass
    # TODO: Your code here

    # OpenCV brute force matching
    """ Remove this comment if you have completed the code until here
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(
        keypoint_descriptors[0].astype(np.float32),
        keypoint_descriptors[1].astype(np.float32),
        2,
    )
    """

    # Apply ratio test
    # TODO: Your code here

    # Plot the results
    """ Remove this comment if you have completed the code until here
    plt.figure()
    dh = int(img2.shape[0] - img1.shape[0])
    top_padding = int(dh / 2)
    img1_padded = cv2.copyMakeBorder(
        img1, top_padding, dh - int(dh / 2), 0, 0, cv2.BORDER_CONSTANT, 0
    )
    plt.imshow(np.c_[img1_padded, img2], cmap="gray")

    # for match in good:
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        x1 = keypoint_locations[0][img1_idx, 1]
        y1 = keypoint_locations[0][img1_idx, 0] + top_padding
        x2 = keypoint_locations[1][img2_idx, 1] + img1.shape[1]
        y2 = keypoint_locations[1][img2_idx, 0]
        plt.plot(np.array([x1, x2]), np.array([y1, y2]), "o-")
    plt.show()
    """


if __name__ == "__main__":
    main(
        ROTATION_INVARIANT,
        ROTATION_IMG2_DEG,
        CONTRAST_THRESHOLD,
        SIFT_SIGMA,
        RESCALE_FACTOR,
        NUM_SCALES,
        NUM_OCTAVES,
    )
