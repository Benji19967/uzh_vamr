from pathlib import Path

from src import utils

KITTI_DIR = Path("data/kitti")
KITTI_IMAGES_DIR = KITTI_DIR / "05" / "image_0"


def main() -> None:

    img = utils.read_img(filepath=KITTI_IMAGES_DIR / "000000.png")
    utils.show_img(img)


if __name__ == "__main__":
    main()
