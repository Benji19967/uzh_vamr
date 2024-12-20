import cv2
from matplotlib import pyplot as plt


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
