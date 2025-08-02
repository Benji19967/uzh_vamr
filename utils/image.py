from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
Notes: cv2 works in (width, height) vs numpy in (height, width)
"""


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
    cv2.setWindowProperty(
        "image", cv2.WND_PROP_TOPMOST, 1
    )  # show image as topmost window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Image:
    def __init__(self, filepath: str) -> None:
        self.img: np.ndarray = read_img(filepath=filepath)
        self.filepath: str

    def show(self, title: str | None = None) -> None:
        show_img(img=self.img, title=title)

    def show_cv(self) -> None:
        show_img_cv(img=self.img)

    def shape(self) -> tuple[int, ...]:
        """
        (height, width)
        """
        return self.img.shape

    def resize(self, width: int, height: int) -> np.ndarray:
        return cv2.resize(self.img, (width, height))

    def resize_inplace(self, width: int, height: int) -> None:
        self.img = cv2.resize(self.img, (width, height))
