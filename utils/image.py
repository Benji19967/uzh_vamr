from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_img(filepath: Path | str) -> np.ndarray:
    return cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)  # type: ignore


def show_img(img: np.ndarray, title: str | None = None):
    plt.imshow(img, interpolation="nearest")
    plt.gray()
    if title:
        plt.title(title)
    plt.show()


def show_img_cv(img: np.ndarray):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Image:
    def __init__(self) -> None:
        self.img: np.ndarray
        self.filepath: str

    def load(self, filepath: str) -> np.ndarray:
        return read_img(filepath=filepath)

    def show(self, title: str | None = None) -> None:
        show_img(img=self.img, title=title)

    def show_cv(self) -> None:
        show_img_cv(img=self.img)
