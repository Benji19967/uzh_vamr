from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from src.image import Dataset, Image
from src.utils import utils


class DataReader(ABC):
    BASE_DIR = Path("")
    IMAGES_DIR = BASE_DIR

    def __init__(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def _filename_from_id(cls, id: int) -> str:
        raise NotImplementedError

    @classmethod
    def read_image(cls, id: int = 0) -> Image:
        filepath = cls.IMAGES_DIR / cls._filename_from_id(id=id)
        img = utils.read_img(filepath=filepath)
        return Image(img=img, dataset=Dataset.KITTI, id=id, filepath=filepath)

    @classmethod
    def show_image(cls, id: int = 0) -> None:
        image = cls.read_image(id=id)
        utils.show_img_cv(img=image.img)

    @classmethod
    def show_images(cls, start_id: int = 0, end_id: int = 5) -> None:
        for id in range(start_id, end_id):
            image = cls.read_image(id=id)
            utils.show_img_cv(img=image.img)


class KittiDataReader(DataReader):
    BASE_DIR = Path("data/kitti")
    IMAGES_DIR = BASE_DIR / "05" / "image_0"  # left image files

    def __init__(self) -> None:
        pass

    @classmethod
    def _filename_from_id(cls, id: int) -> str:
        return f"{id:06}.png"


class MalagaDataReader(DataReader):
    BASE_DIR = Path("data/malaga-urban-dataset-extract-07")
    IMAGES_DIR = BASE_DIR / "Images"

    @classmethod
    def _filename_from_id(cls, id: int) -> str:
        return cls._filenames()[id]

    @classmethod
    @lru_cache()
    def _filenames(cls) -> list[str]:
        return [
            f"{full_filename.stem}.jpg"
            for full_filename in sorted(cls.IMAGES_DIR.glob("*left.jpg"))
        ]


class ParkingDataReader(DataReader):
    BASE_DIR = Path("data/parking")
    IMAGES_DIR = BASE_DIR / "images"

    def __init__(self) -> None:
        pass

    @classmethod
    def _filename_from_id(cls, id: int) -> str:
        return f"img_{id:05}.png"
