import cv2
from src.features import Descriptors, HarrisDetector, HarrisScores, Keypoints
from src.image import Image
from src.utils import utils
from src.utils.data_reader import KittiDataReader


def main():
    image_1 = KittiDataReader.read_image(id=0)
    image_2 = KittiDataReader.read_image(id=1)

    descriptors = []
    for image in [image_1, image_2]:
        hs = HarrisScores(image=image)
        hs.plot()

        kp = Keypoints(image=image, scores=hs.scores)
        kp.plot()

        desc = Descriptors(image=image, keypoints=kp.keypoints)
        desc.plot()

        descriptors.append(desc.descriptors)

    Descriptors.match(
        query_descriptors=descriptors[1],
        db_descriptors=descriptors[0],
    )

    # scores_cv = cv2.cornerHarris(img, blockSize=9, ksize=3, k=0.08)
    # utils.show_img(scores_cv)

    # keypoints = Keypoints.select(
    #     scores=scores, num_keypoints=100, non_max_suppression_radius=9
    # )
    # print(keypoints)


if __name__ == "__main__":
    main()
