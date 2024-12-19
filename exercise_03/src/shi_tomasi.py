import numpy as np
from scipy import signal
import utils


def shi_tomasi(img, patch_size):
    """Returns the shi-tomasi scores for an image and patch size patch_size
    The returned scores are of the same shape as the input image"""

    # show_img(img)

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

    I_x = signal.convolve2d(Sobel_x, img, mode="valid")
    I_y = signal.convolve2d(Sobel_y, img, mode="valid")
    I_xx = I_x * I_x
    I_yy = I_y * I_y
    I_xy = I_x * I_y

    utils.show_np_array_as_img(I_x, title="I_x")
    utils.show_np_array_as_img(I_y, title="I_y")
    utils.show_np_array_as_img(I_xx, title="I_xx")
    utils.show_np_array_as_img(I_yy, title="I_yy")
    utils.show_np_array_as_img(I_xy, title="I_xy")

    Sum_I_xx = np.sum(I_xx)
    Sum_I_yy = np.sum(I_yy)
    Sum_I_xy = np.sum(I_xy)

    return None
