import cv2
import numpy as np


def computeBlurredImages(image_pyramid, num_scales: int, sift_sigma: float):
    # The number of octaves can be inferred from the length of the image pyramid
    S = num_scales + 3
    blurred_images = []
    for img in image_pyramid:
        images_per_sigma = []
        for s in range(-1, S + 2):
            gauss_blur_sigma = (2 ** (s / S)) * sift_sigma
            # where does this come from?
            filter_size = int(2 * np.ceil(2 * gauss_blur_sigma) + 1.0)
            img_blurred = cv2.GaussianBlur(
                src=img,
                ksize=(filter_size, filter_size),
                sigmaX=gauss_blur_sigma,
                borderType=cv2.BORDER_DEFAULT,
            )
            images_per_sigma.append(img_blurred)
        blurred_images.append(images_per_sigma)
    return blurred_images
