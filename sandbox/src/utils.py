from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt


def read_img(filepath: Path) -> np.ndarray:
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # type: ignore


def show_img(img: np.ndarray, title: str | None = None):
    plt.imshow(img, interpolation="nearest")
    plt.gray()
    if title:
        plt.title(title)
    plt.show()


def show_img_cv(img: np.ndarray):
    cv2.imshow("image", img)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
