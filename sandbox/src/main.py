from src.utils.data_reader import KittiDataReader, ParkingDataReader


def main() -> None:
    KittiDataReader.show_images()
    ParkingDataReader.show_images()


if __name__ == "__main__":
    main()
