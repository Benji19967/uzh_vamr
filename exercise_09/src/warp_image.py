import numpy as np


def warpImage(image, W):
    """
    Input
        image   np.ndarray
        W       2 x 3 np.ndarray

    Output
        warped image of the same dimensions
    """
    print(W.shape)
    print(image.shape)
    warped_image = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[0])):
            warped_pixel = W @ np.array([x, y, 1])
            wx, wy = int(warped_pixel[0]), int(warped_pixel[1])
            if wy < len(image) and wx < len(image[0]):
                warped_image[y][x] = image[wy][wx]
            else:
                warped_image[y][x] = 0

    print(warped_image)

    return warped_image
