from glob import glob

import cv2
import cv2 as cv

from utils.image import Image, show_img


def main():
    # image = Image(filepath="./data/iPhone7/IMG_5713.JPG")
    # print(image.shape())
    # image.show_cv()
    #
    # print(image.resize(320, 480).shape)
    #
    # print(image.resize_inplace(320, 480))
    # print(image.shape())
    # image.show_cv()

    # for path in glob("./data/iPhone7/*"):
    #     image = Image(filepath=path)
    #     h, w = image.shape()
    #     image.resize_inplace(width=w // 4, height=h // 4)
    #     new_path = (path.strip(".JPG") + "_small.JPG")[1:]
    #     print(new_path)
    #     image.save(filepath=new_path)

    # SIFT
    img = cv.imread("./data/iPhone7/IMG_5913_small.JPG")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)

    img = cv.drawKeypoints(
        gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    show_img(img)


if __name__ == "__main__":
    main()
