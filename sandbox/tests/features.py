import cv2
from src.features import HarrisDetector, HarrisScores, Keypoints
from src.image import Image
from src.utils import utils
from src.utils.data_reader import KittiDataReader


def main():
    image = KittiDataReader.read_image(id=0)

    hs = HarrisScores(image=image)
    hs.plot()

    kp = Keypoints(image=image, scores=hs.scores())
    kp.plot()

    # scores_cv = cv2.cornerHarris(img, blockSize=9, ksize=3, k=0.08)
    # utils.show_img(scores_cv)

    # keypoints = Keypoints.select(
    #     scores=scores, num_keypoints=100, non_max_suppression_radius=9
    # )
    # print(keypoints)


if __name__ == "__main__":
    main()
