from utils.image import Image


def main():
    image = Image(filepath="./data/iPhone7/IMG_5713.JPG")
    print(image.shape())
    image.show_cv()

    print(image.resize(320, 480).shape)

    print(image.resize_inplace(320, 480))
    print(image.shape())
    image.show_cv()


if __name__ == "__main__":
    main()
