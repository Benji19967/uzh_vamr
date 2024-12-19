from matplotlib import pyplot as plt
import cv2
from pprint import pprint
import numpy as np


def show_img(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


def show_np_array_as_img(arr, title: str | None = None):
    plt.imshow(arr, interpolation="nearest")
    plt.gray()
    if title:
        plt.title(title)
    plt.show()


def np_fullprint(*args, **kwargs):
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)
