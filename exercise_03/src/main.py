from tkinter import W

import cv2
import matplotlib.pyplot as plt

from describe_keypoints import describeKeypoints
from harris import harris
from match_descriptors import matchDescriptors
from plot_matches import plotMatches
from select_keypoints import selectKeypoints
from shi_tomasi import shi_tomasi

# Randomly chosen parameters that seem to work well - can you find better ones?
CORNER_PATCH_SIZE = 9
HARRIS_KAPPA = 0.08
NUM_KEYPOINTS = 200
NONMAXIMUM_SUPRESSION_RADIUS = 8
DESCRIPTOR_RADIUS = 9
MATCH_LAMBDA = 4


def read_img():
    return cv2.imread("../data/000000.png", cv2.IMREAD_GRAYSCALE)


def part1(img):
    # Part 1 - Calculate Corner Response Functions
    # Shi-Tomasi
    shi_tomasi_scores = shi_tomasi(img, CORNER_PATCH_SIZE)

    # Harris
    harris_scores = harris(img, CORNER_PATCH_SIZE, HARRIS_KAPPA)
    fig, axs = plt.subplots(2, 2, squeeze=False)
    axs[0, 0].imshow(img, cmap="gray")
    axs[0, 0].axis("off")
    axs[0, 1].imshow(img, cmap="gray")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(shi_tomasi_scores)
    axs[1, 0].set_title("Shi-Tomasi Scores")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(harris_scores)
    axs[1, 1].set_title("Harris Scores")
    axs[1, 1].axis("off")

    fig.tight_layout()
    plt.show()

    return harris_scores


def part2(img, harris_scores):
    # Part 2 - Select keypoints
    keypoints = selectKeypoints(
        harris_scores, NUM_KEYPOINTS, NONMAXIMUM_SUPRESSION_RADIUS
    )

    plt.clf()
    plt.close()
    plt.imshow(img, cmap="gray")
    plt.plot(keypoints[1, :], keypoints[0, :], "rx", linewidth=2)
    plt.axis("off")
    plt.show()


def part3(img):
    # Part 3 - Describe keypoints and show 16 strongest keypoint descriptors
    descriptors = describeKeypoints(img, keypoints, DESCRIPTOR_RADIUS)

    plt.clf()
    plt.close()
    fig, axs = plt.subplots(4, 4)
    patch_size = 2 * DESCRIPTOR_RADIUS + 1
    for i in range(16):
        axs[i // 4, i % 4].imshow(descriptors[:, i].reshape([patch_size, patch_size]))
        axs[i // 4, i % 4].axis("off")

    plt.show()


def part4():
    # Part 4 - Match descriptors between first two images
    img_2 = cv2.imread("../data/000001.png", cv2.IMREAD_GRAYSCALE)
    harris_scores_2 = harris(img_2, CORNER_PATCH_SIZE, HARRIS_KAPPA)
    keypoints_2 = selectKeypoints(
        harris_scores_2, NUM_KEYPOINTS, NONMAXIMUM_SUPRESSION_RADIUS
    )
    descriptors_2 = describeKeypoints(img_2, keypoints_2, DESCRIPTOR_RADIUS)

    matches = matchDescriptors(descriptors_2, descriptors, MATCH_LAMBDA)

    plt.clf()
    plt.close()
    plt.imshow(img_2, cmap="gray")
    plt.plot(keypoints_2[1, :], keypoints_2[0, :], "rx", linewidth=2)
    plotMatches(matches, keypoints_2, keypoints)
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def part5():
    # Part 5 - Match descriptors between all images
    prev_desc = None
    prev_kp = None
    for i in range(200):
        plt.clf()
        img = cv2.imread("../data/{0:06d}.png".format(i), cv2.IMREAD_GRAYSCALE)
        scores = harris(img, CORNER_PATCH_SIZE, HARRIS_KAPPA)
        kp = selectKeypoints(scores, NUM_KEYPOINTS, NONMAXIMUM_SUPRESSION_RADIUS)
        desc = describeKeypoints(img, kp, DESCRIPTOR_RADIUS)

        plt.imshow(img, cmap="gray")
        plt.plot(kp[1, :], kp[0, :], "rx", linewidth=2)
        plt.axis("off")

        if prev_desc is not None:
            matches = matchDescriptors(desc, prev_desc, MATCH_LAMBDA)
            plotMatches(matches, kp, prev_kp)
        prev_kp = kp
        prev_desc = desc

        plt.pause(0.1)


def main():
    img = read_img()
    harris_scores = part1(img)
    part2(img=img, harris_scores=harris_scores)
    part3(img)
    part4()
    part5()


if __name__ == "__main__":
    print("Starting exercise 3")
    # main()
    print("Done exercise 3")
