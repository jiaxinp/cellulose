import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from MyLibrary.BG_Calibrator import BG_Calibrator
from MyLibrary.ProcessedImageClass import ProcessedImage


def choose_data(data_dir: Path = Path("./data")):
    data_dict = {n: file.stem for n, file in enumerate(data_dir.iterdir()) if file.is_dir()}
    pprint.pprint(data_dict)
    data_key = int(input("choose the data to analyse: "))
    filepath = Path(f"./data/{data_dict[data_key]}/")
    flist = sorted(list(filepath.glob("*.txt")))
    return flist


def main():
    calibrater = BG_Calibrator(apply_median=False)

    flist = choose_data()
    for n, f in enumerate(flist):
        print(f"\nループカウントは{n}です")
        print(f"{f.stem} is processed now...")
        AFM_data = np.loadtxt(f, skiprows=1)
        image_size = int(
            np.sqrt(len(AFM_data))
        )  # the number of pixels that a side of AFM image has
        height_data = AFM_data.reshape((image_size, image_size))

        image = ProcessedImage(height_data, name=f.stem)
        calibrater(image)

        fig, ax = plt.subplots(1, 1)
        ax.axis("off")
        ax.imshow(image.calibrated_image, vmin=-0.5, vmax=5, cmap="afmhot")
        plt.savefig(
            f"/Users/tomok/Documents/Python/FirstPaperProgram/Figure/zemi/{f.stem}.png", dpi=300
        )
        plt.show()

        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1024), image.calibrated_image[512])
        ax.set_ylim(-1, 2)
        plt.savefig(
            f"/Users/tomok/Documents/Python/FirstPaperProgram/Figure/zemi/{f.stem}2.png", dpi=300
        )
        plt.show()


if __name__ == "__main__":
    main()
