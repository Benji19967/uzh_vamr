import time

import numpy as np


def warpImage(image, W):
    """
    Input
        image   np.ndarray
        W       2 x 3 np.ndarray

    Output
        warped image of the same dimensions
    """
    warped_image = np.zeros(image.shape)
    print(image.shape)

    # for y in range(len(image)):
    #     for x in range(len(image[0])):
    #         warped_pixel = W @ np.array([x, y, 1])
    #         wx, wy = int(warped_pixel[0]), int(warped_pixel[1])
    #         if wy < len(image) and wx < len(image[0]):
    #             warped_image[y][x] = image[wy][wx]
    #         else:
    #             warped_image[y][x] = 0

    # Roughly 50x faster than the above implementation
    min_coords = np.array([0, 0])
    max_coords = image.shape[::-1]
    xm, ym = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xm = np.reshape(xm, (1, -1))
    ym = np.reshape(ym, (1, -1))
    pre_warp = np.r_[xm, ym, np.ones_like(xm)]
    warped = (W @ pre_warp).T

    # TODO: add question about this to numpy notebook
    mask = np.logical_or.reduce(
        np.c_[
            warped[:, 0] >= max_coords[0],
            warped[:, 0] <= min_coords[0],
            warped[:, 1] >= max_coords[1],
            warped[:, 1] <= min_coords[1],
        ],
        axis=1,
    )
    warped[mask, :] = 0
    warped_int = warped.astype("int")
    warped_image = image[warped_int[:, 1], warped_int[:, 0]]
    warped_image[mask] = 0
    warped_image = np.reshape(warped_image, image.shape)

    return warped_image
