from pathlib import Path

import numpy as np
from src.utils import utils


class DataReader:

    def __init__(self) -> None:
        pass

    def read_image(self, id: int) -> np.ndarray:
        pass

    def read_all_images(self) -> list[np.ndarray]:
        pass


class KittiDataReader:
    BASE_DIR = Path("data/kitti")
    IMAGES_DIR = BASE_DIR / "05" / "image_0"

    def __init__(self) -> None:
        pass

    @classmethod
    def read_image(cls, id: int) -> np.ndarray:
        return utils.read_img(filepath=cls.IMAGES_DIR / f"{id:06}.png")

    @classmethod
    def show_image(cls, id: int = 0) -> None:
        img = cls.read_image(id=id)
        utils.show_img(img=img)

    @classmethod
    def show_images(cls, start_id: int = 0, end_id: int = 10) -> None:
        for id in range(start_id, end_id):
            img = cls.read_image(id=id)
            utils.show_img(img=img)


class ParkingDataReader:
    BASE_DIR = Path("data/parking")
    IMAGES_DIR = BASE_DIR / "images"

    def __init__(self) -> None:
        pass

    @classmethod
    def read_image(cls, id: int) -> np.ndarray:
        return utils.read_img(filepath=cls.IMAGES_DIR / f"img_{id:05}.png")

    @classmethod
    def show_image(cls, id: int = 0) -> None:
        img = cls.read_image(id=id)
        utils.show_img(img=img)

    @classmethod
    def show_images(cls, start_id: int = 0, end_id: int = 10) -> None:
        for id in range(start_id, end_id):
            img = cls.read_image(id=id)
            utils.show_img(img=img)
