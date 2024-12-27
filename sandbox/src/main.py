import cv2
from src.utils import utils
from src.utils.data_reader import KittiDataReader, ParkingDataReader


def main() -> None:
    img = KittiDataReader.read_image(id=0)
    # ParkingDataReader.show_images()
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    utils.show_img(dst)


if __name__ == "__main__":
    main()
