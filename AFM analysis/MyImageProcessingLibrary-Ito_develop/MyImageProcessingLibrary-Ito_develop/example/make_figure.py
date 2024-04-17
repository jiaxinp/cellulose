import pickle
import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from MyLibrary import imptools

plt.rcParams["figure.figsize"] = (3.54, 3.54)
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.labelpad"] = 4
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.major.pad"] = 6
plt.rcParams["ytick.major.pad"] = 6
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["lines.linewidth"] = 2

plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["xtick.major.size"] = 5.7
plt.rcParams["ytick.major.size"] = 5.7

plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["xtick.minor.size"] = 3.3
plt.rcParams["ytick.minor.size"] = 3.3

plt.rcParams["lines.markersize"] = 1
plt.rcParams["figure.autolayout"] = True
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["figure.dpi"] = 600


def save_AFM_image(image_list, save_dir, save_fig):
    if save_fig:
        image_dir = save_dir / "calibrated_image"
        if image_dir.exists():
            image_dir.mkdir()

        for n, image in enumerate(image_list):
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(image.calibrated_image, vmin=-0.5, vmax=5.5, cmap="afmhot")
            ax.axis("off")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax)
            plt.savefig(save_dir / f"{n}_AFM.png", dpi=900)
            plt.close()
    else:
        pass


def kink_posis(image_list, save_dir, save_fig):  # todo まだやっていない
    if save_fig:
        image_dir = save_dir / "kink_posis"
        if image_dir.exists():
            image_dir.mkdir()
        for n, image in enumerate(image_list):
            fig, ax = plt.subplots(1, 1)
            ax.imshow(image.binarized_image, vmin=-0.5, vmax=5.5, cmap="gray")
            ax.axis("off")
            plt.savefig(save_dir / f"{n}_AFM.png", dpi=900)
            plt.close()
    else:
        pass


def total_height_hist(image_list, save_dir, save_fig):
    all_height = []
    for image in image_list:
        height = image.calibrated_image[
            np.where(image.skeleton_image)
        ]  # todo もしかしてlistに変換するのが超おそい原因？
        all_height.extend(list(height))

    if save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.hist(all_height, density=True, bins=np.arange(0, 8, 0.1), color="dimgrey")
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 0.7)
        ax.set_xticks(np.arange(0, 8, 1))
        ax.set_yticks(np.arange(0, 0.8, 0.2))
        plt.savefig(save_dir / "totoal_height.png", dpi=900)
    plt.close()


def calc_average_height(image_list):
    all_height = []
    for image in image_list:
        height = image.calibrated_image[
            np.where(image.skeleton_image)
        ]  # todo もしかしてlistに変換するのが超おそい原因？
        all_height.extend(list(height))
    average_height = np.average(all_height)
    print("average height is ", average_height)
    print(f"standard devuation is {np.std(all_height)}")


def all_length_hist(image_list, save_dir, save_fig):
    all_length = []
    for image in image_list:
        for i in range(1, image.nLabels):  # todo 1
            y, x, h, w, area = image.data[i]
            length = imptools.get_length(image.skeleton_image[x: x + w, y: y + h])
            all_length.append(length)

    if save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.hist(
            length,
            density=True,
            bins=np.logspace(np.log10(1), np.log10(5000), 60),
            color="dimgrey",
        )
        ax.set_xscale("log")
        ax.set_xlim(10, 10**4)
        # ax.set_ylim(0, 0.004)
        # ax.set_xticks(np.arange(0, 8, 1))
        # ax.set_yticks(np.arange(0, 0.8, 0.2))
        plt.savefig(save_dir / "all_length.png", dpi=900)

        plt.savefig(save_dir / (str(i) + ".png"))
        plt.close()


prc_image_dir = Path("../")

if __name__ == "__main__":
    """
    save figure of analytical result mainly about information from all fibers

    """
    data_dict = {
        n: file.stem
        for n, file in enumerate((prc_image_dir / "pickle_data").iterdir())
        if file.is_dir()
    }
    pprint.pprint(data_dict)
    data_key = int(input("choose the data to make figure: "))
    input_dir = prc_image_dir / ("pickle_data/" + data_dict[data_key])
    save_dir = prc_image_dir / ("Figure/" + data_dict[data_key])
    if not save_dir.exists():
        save_dir.mkdir()

    file_list = list(input_dir.glob("*.pickle"))
    print(file_list)

    image_list = []
    for f in file_list:
        with open(f, mode="rb") as pickled_file:
            image = pickle.load(pickled_file)
            image_list.append(image)

    while True:
        ask_save_fig = input("Do you save figures? [y/n]")
        if ask_save_fig == "y":
            save_fig = True
            break
        elif ask_save_fig == "n":
            save_fig = False
            break
        else:
            print('enter "y" or "n": ')

    total_height_hist(image_list, save_dir, save_fig)
    save_AFM_image(image_list, save_dir, save_fig)
    all_length_hist(image_list, save_dir, save_fig)
    calc_average_height(image_list)
