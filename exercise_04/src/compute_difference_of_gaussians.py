import cv2
import numpy as np

import utils


def computeDifferenceOfGaussians(blurred_images):
    # The number of octaves can be inferred from the length of blurred_images
    stack_of_DoGs_per_octave = []
    for images_in_octave in blurred_images:
        differences_of_gaussians_for_octave = []
        for img_at_sigma_1, img_at_sigma_2 in zip(
            images_in_octave[:-1], images_in_octave[1:]
        ):
            DoG = img_at_sigma_2 - img_at_sigma_1
            utils.show_np_array_as_img(DoG)
            differences_of_gaussians_for_octave.append(DoG)
        stack_of_DoGs_per_octave.append(differences_of_gaussians_for_octave)
