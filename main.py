from glob import glob

from utils.image import Image


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

    pass


if __name__ == "__main__":
    main()
