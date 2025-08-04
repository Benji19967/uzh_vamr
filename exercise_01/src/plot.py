import matplotlib.pyplot as plt
import numpy as np

X_INDEX = 0
Y_INDEX = 1
IMG_CMAP = "gray"
MARKER_SIZE = 3


def points_on_image(img: np.ndarray, points: np.ndarray, img_idx: int) -> None:
    plt.clf()
    plt.close()
    plt.imshow(img, cmap=IMG_CMAP)
    plt.plot(
        points[X_INDEX],
        points[Y_INDEX],
        "or",
        markersize=MARKER_SIZE,
    )
    plt.show()
    # TODO: create a CLI and add an option to chose viewing/saving images

    # img_idx_padded = f"{img_idx}".zfill(4)
    # plt.savefig(f"./data/images_with_corners/img_{img_idx_padded}.jpg")


def draw_cube(img: np.ndarray, vertices: np.ndarray) -> None:

    plt.imshow(img, cmap="gray")
    plt.plot(
        vertices[0],
        vertices[1],
        "or",
        markersize=3,
    )

    vertices = np.transpose(vertices)
    lw = 3
    # base layer of the cube
    plt.plot(
        vertices[[1, 3, 7, 5, 1], 0],
        vertices[[1, 3, 7, 5, 1], 1],
        "r-",
        linewidth=lw,
    )

    # top layer of the cube
    plt.plot(
        vertices[[0, 2, 6, 4, 0], 0],
        vertices[[0, 2, 6, 4, 0], 1],
        "r-",
        linewidth=lw,
    )

    # vertical lines
    plt.plot(
        vertices[[0, 1], 0],
        vertices[[0, 1], 1],
        "r-",
        linewidth=lw,
    )
    plt.plot(
        vertices[[2, 3], 0],
        vertices[[2, 3], 1],
        "r-",
        linewidth=lw,
    )
    plt.plot(
        vertices[[4, 5], 0],
        vertices[[4, 5], 1],
        "r-",
        linewidth=lw,
    )
    plt.plot(
        vertices[[6, 7], 0],
        vertices[[6, 7], 1],
        "r-",
        linewidth=lw,
    )

    plt.show()


def two_images(img1, img2, title1: str, title2: str):
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].imshow(img1, cmap="gray")
    axs[0].set_axis_off()
    axs[0].set_title(title1)
    axs[1].imshow(img2, cmap="gray")
    axs[1].set_axis_off()
    axs[1].set_title(title2)
    plt.show()
