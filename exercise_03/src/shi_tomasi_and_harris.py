import numpy as np
from scipy import signal

import utils


def compute_coefficients(img, patch_size, show_output):
    Sobel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]
    Sobel_y = [
        [-1, -2, 1],
        [0, 0, 0],
        [1, 2, 1],
    ]

    Ix = signal.convolve2d(Sobel_x, img, mode="valid")
    Iy = signal.convolve2d(Sobel_y, img, mode="valid")
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    if show_output:
        utils.show_np_array_as_img(Ix, title="Ix")
        utils.show_np_array_as_img(Iy, title="Iy")
        utils.show_np_array_as_img(Ixx, title="Ixx")
        utils.show_np_array_as_img(Iyy, title="Iyy")
        utils.show_np_array_as_img(Ixy, title="Ixy")

    patch = np.ones([patch_size, patch_size])
    pr = patch_size // 2
    sIxx = signal.convolve2d(Ixx, patch, mode="valid")
    sIyy = signal.convolve2d(Iyy, patch, mode="valid")
    sIxy = signal.convolve2d(Ixy, patch, mode="valid")

    return sIxx, sIyy, sIxy, pr


def pad_scores(scores, pr):
    return np.pad(
        scores, [(pr + 1, pr + 1), (pr + 1, pr + 1)], mode="constant", constant_values=0
    )


def shi_tomasi(img, patch_size, show_output):
    """Returns the shi-tomasi scores for an image and patch size patch_size
    The returned scores are of the same shape as the input image"""

    sIxx, sIyy, sIxy, pr = compute_coefficients(
        img=img, patch_size=patch_size, show_output=show_output
    )

    trace = sIxx + sIyy
    determinant = sIxx * sIyy - sIxy**2

    # the eigen values of a matrix M=[a,b;c,d] are lambda1/2 = (Tr(A)/2 +- ((Tr(A)/2)^2-det(A))^.5
    # The smaller one is the one with the negative sign
    scores = trace / 2 - ((trace / 2) ** 2 - determinant) ** 0.5
    scores[scores < 0] = 0

    scores = pad_scores(scores=scores, pr=pr)

    return scores


def harris(img, patch_size, kappa, show_output):
    """Returns the harris scores for an image given a patch size and a kappa value
    The returned scores are of the same shape as the input image"""

    sIxx, sIyy, sIxy, pr = compute_coefficients(
        img=img, patch_size=patch_size, show_output=show_output
    )

    trace = sIxx + sIyy
    determinant = sIxx * sIyy - sIxy**2

    scores = determinant - kappa * (trace**2)
    scores[scores < 0] = 0

    scores = pad_scores(scores=scores, pr=pr)

    return scores
