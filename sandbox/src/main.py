from pathlib import Path

import cv2
from src.features import Descriptors, HarrisDetector, HarrisScores, Keypoints
from src.utils import utils
from src.utils.data_reader import (
    Image,
    KittiDataReader,
    MalagaDataReader,
    ParkingDataReader,
)


def main() -> None:
    # img = KittiDataReader.read_image(id=0)
    # KittiDataReader.show_image()
    # ParkingDataReader.show_image()
    # MalagaDataReader.show_images(end_id=100)

    keypoints = []
    descriptors = []
    for path in ["data/iPhone7/IMG_5713.JPG", "data/iPhone7/IMG_5714.JPG"]:
        img = utils.read_img(filepath=path)
        image = Image(img=img, dataset="other", id=0, filepath=Path(""))
        # dst = cv2.cornerHarris(img, 2, 3, 0.04)
        # utils.show_img(dst)
        hs = HarrisScores(image=image)
        kp = Keypoints(image=image, scores=hs.scores)
        keypoints.append(kp.keypoints)
        kp.plot()
        desc = Descriptors(image=image, keypoints=kp.keypoints)
        desc.plot()
        descriptors.append(desc.descriptors)

    matches = Descriptors.match(
        query_descriptors=descriptors[1], db_descriptors=descriptors[0]
    )
    Descriptors.plot_matches(
        matches=matches, query_keypoints=keypoints[1], database_keypoints=keypoints[0]
    )


if __name__ == "__main__":
    main()
