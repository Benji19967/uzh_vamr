import cv2
import numpy as np


def computeImagePyramid(img: np.ndarray, num_octaves: int) -> list[np.ndarray]:
    """
    num octaves = len(image_pyramid)
    """
    image_pyramid = [img]

    for i in range(num_octaves - 1):
        # scales resolution down to 0.5 of previous img
        image_pyramid.append(cv2.resize(image_pyramid[i], (0, 0), fx=0.5, fy=0.5))  # type: ignore

    return image_pyramid
