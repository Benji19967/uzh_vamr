import cv2
import numpy as np


def getGaussianKernel(size, sigma: int):
    pass
    # TODO: Your code here


def getImageGradient(image: np.ndarray):
    pass
    # TODO: Your code here


def derotatePatch(img: np.ndarray, loc, patch_size, orientation):
    # it can't be worse than a 45 degree rotation, so lets pad
    # under this assumption. Then it will be enough for sure.
    pass
    # TODO: Your code here

    # compute derotated patch
    for px in range(patch_size):
        for py in range(patch_size):
            pass

    # TODO: Your code here

    # rotate patch by angle ori
    # TODO: Your code here

    # move coordinates to patch
    # TODO: Your code here

    # sample image (using nearest neighbor sampling as opposed to more
    # accuracte bilinear sampling)
    # TODO: Your code here
    # Return the patch
    # TODO: Your code here


def computeDescriptors(
    blurred_images: list[list[np.ndarray]],
    keypoint_locations: list[np.ndarray],
    rotation_invariant: bool,
):
    # return descriptors and final keypoint locations
    pass
    # TODO: Your code here
