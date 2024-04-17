import pickle
import pprint
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = (3.54, 3.54)
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.labelpad"] = 4
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.major.pad"] = 6
plt.rcParams["ytick.major.pad"] = 6
plt.rcParams["font.size"] = 7
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["lines.linewidth"] = 1.7

plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6

plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["xtick.minor.size"] = 3.3
plt.rcParams["ytick.minor.size"] = 3.3

plt.rcParams["lines.markersize"] = 1
plt.rcParams["figure.autolayout"] = True
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False
plt.rcParams["figure.dpi"] = 900


def save_breakdown_barplot1(breakdown_dict_data_lst: Iterable[dict], savename: Path):
    """
    Save figure that comparing the composition of 'dent', 'normal' and 'bump'
    between samples.
    :param breakdown_dict_data_lst:
    :param savename:
    :return:
    """
    category_colors = ["navy", "red", "darkgreen"]
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 1.5))
    ax.set_xticks(range(0, 120, 20), minor=False)
    ax.set_yticks([0, 1], minor=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.5)
    for i, breakdown_dict_data in enumerate(breakdown_dict_data_lst):
        dent_c = sum(breakdown_dict_data["dent"].values())
        normal_c = breakdown_dict_data["normal"]
        bump_c = breakdown_dict_data["bump"]
        widths = [dent_c, normal_c, bump_c]
        comp_cumsum = [0, dent_c, dent_c + normal_c]
        print("width", widths)
        print("left", comp_cumsum)
        for j, (left, width) in enumerate(zip(comp_cumsum, widths)):
            ax.barh(i, width, height=0.6, left=left, color=category_colors[j])
    plt.savefig(savename, transparent=True)


def save_breakdown_barplot2(breakdown_dict_data_lst: Iterable[dict], savename: Path):
    """
    Save figure that comparing the composition of 'at kink', 'ep', 'on straight' and 'kinked_end'
    between samples.
    :param breakdown_dict_data_lst:
    :param savename:
    :return:
    """
    category_colors = ["coral", "olive", "cadetblue", "seagreen"]
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 1.5))
    ax.set_xticks(range(0, 70, 10), minor=False)
    ax.set_yticks([0, 1], minor=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.5, 1.5)
    for i, breakdown_dict_data in enumerate(breakdown_dict_data_lst):
        kink_c = breakdown_dict_data["dent"]["kink"]
        ep_c = breakdown_dict_data["dent"]["ep"]
        straight_c = breakdown_dict_data["dent"]["straight"]
        kinked_end_c = breakdown_dict_data["dent"]["kinked_end"]

        widths = [kink_c, ep_c, straight_c, kinked_end_c]
        comp_cumsum = [0, kink_c, kink_c + ep_c, kink_c + ep_c + straight_c]
        print("width", widths)
        print("left", comp_cumsum)
        for j, (left, width) in enumerate(zip(comp_cumsum, widths)):
            ax.barh(i, width, height=0.6, left=left, color=category_colors[j])
    plt.savefig(savename, transparent=True)


if __name__ == "__main__":
    _10mmol_data_file = Path(
        "/Users/tomok/Documents/Python/FirstPaperProgram"
        "/pickle_data/Softholo_5mmol_220524/breakdown/breakdown_dict_data.pickle"
    )

    _sonication_data_file = Path(
        "/Users/tomok/Documents/Python/FirstPaperProgram"
        "/pickle_data/sonication_ver3/breakdown/breakdown_dict_data.pickle"
    )

    _221015_data_file = Path(
        "/Users/tomok/Documents/Python/FirstPaperProgram"
        "/pickle_data/221005/breakdown/breakdown_dict_data.pickle"
    )

    _221016_data_file = Path(
        "/Users/tomok/Documents/Python/FirstPaperProgram"
        "/pickle_data/221006/breakdown/breakdown_dict_data.pickle"
    )

    files = [_221016_data_file, _221015_data_file]
    bd_dict_data_lst = []
    for file in files:
        with open(file, "rb") as f:
            bd_data = pickle.load(f)
        bd_dict_data_lst.append(bd_data)
    pprint.pprint(bd_dict_data_lst)

    save_name = Path("/Users/tomok/Documents/Python/FirstPaperProgram/Figure")
    save_breakdown_barplot1(bd_dict_data_lst, save_name / "breakdown1.svg")
    save_breakdown_barplot2(bd_dict_data_lst, save_name / "breakdown2.svg")
