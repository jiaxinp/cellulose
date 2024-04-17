import itertools
import pickle
import pprint
from pathlib import Path
from typing import Dict, Generator, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from MyLibrary.ProcessedImageClass import Fiber


def main():
    fibers = choose_PrcImg_data()
    for fiber in fibers:
        # cubic_spline = CubicSpline(fiber.ytrack, fiber.xtrack)

        relmax_indices = signal.argrelmax(fiber.height, order=2)
        relmax_indices2 = np.intersect1d(
            np.where(fiber.height > 3), relmax_indices
        )  # 最初に検出したピークのうち高さ3以下の場所はカット

        peak_indices = integrate_close_peaks(fiber.height, relmax_indices2)
        fig, axes = plt.subplots(2, 1)
        axes.ravel()
        axes[0].scatter(fiber.ytrack, fiber.xtrack)
        axes[0].imshow(fiber.AFM_image, cmap="afmhot", vmax=5.5, vmin=-0.5)
        axes[1].plot(fiber.horizon, fiber.height)
        axes[1].scatter(fiber.horizon[relmax_indices], fiber.height[relmax_indices], c="red", s=18)
        plt.show()


# wanna_save_result()  # 上記までに処理したデータを辞書の形でここに渡す。


def choose_PrcImg_data(
    datadir: Path = Path("./pickle_data/"), direct_fiber_path: Optional[Path] = None
) -> Generator[Fiber, None, None]:
    """

    :param datadir: directory path which contain ProcessedImage data
    :param direct_fiber_path:
    :return: file list which contain all original txt data
    """
    if direct_fiber_path is None:
        data_dict = {n: file.stem for n, file in enumerate(datadir.iterdir()) if file.is_dir()}
        pprint.pprint(data_dict)
        data_key = int(input("choose the ProcessedImage data to analyse: "))
        fiber_data_path = Path(datadir / (data_dict[data_key]) / "Fiber_data")
        fibers_path = itertools.chain.from_iterable(
            x.glob("*.pickle") for x in fiber_data_path.iterdir()
        )
        for fiber_path in fibers_path:
            with open(fiber_path, mode="rb") as f:
                try:
                    fiber = pickle.load(f)
                    print(fiber_path)
                    yield fiber
                except EOFError:
                    print(f"File size is 0. Skip {fiber_path}")
    else:
        with open(direct_fiber_path, "rb") as f:
            fiber = pickle.load(f)
            yield fiber


def integrate_close_peaks(height_profile: np.ndarray, arg_relmax: np.ndarray, gap: int = 12):
    """

    :param height_profile:
    :param arg_relmax:
    :param gap:
    :return: ndarray of peak indices of height_profile
    """

    diff_between_indices = (np.roll(arg_relmax, -1) - arg_relmax)[:-1]

    close_peaks_starts = []  # 近いピーク集団のスタート位置を表す(diff_between_indicesの中での)インデックスの集合
    close_peaks_ends = []

    integrated_peak_indices = []

    if diff_between_indices[0] < gap:
        close_peaks_starts.append(0)

    for i in range(1, len(diff_between_indices)):
        if diff_between_indices[i - 1] > gap and diff_between_indices[i] > gap:  # 独立したピークを返値に入れる
            integrated_peak_indices.append(arg_relmax[i])

        if diff_between_indices[i - 1] > gap and diff_between_indices[i] <= gap:
            close_peaks_starts.append(i)

        if diff_between_indices[i - 1] <= gap and diff_between_indices[i] > gap:
            close_peaks_ends.append(i)

    if diff_between_indices[-1] <= gap:
        close_peaks_ends.append(len(diff_between_indices))

    start_end_pairs = [(s, e) for s, e in zip(close_peaks_starts, close_peaks_ends)]

    # integrate close peaks to the highest
    for (s, e) in start_end_pairs:
        highest_peak_index = max(range(s, e + 1), key=lambda x: height_profile[arg_relmax[x]])
        integrated_peak_indices.append(arg_relmax[highest_peak_index])

    return sorted(integrated_peak_indices)


def wanna_save_result(
    data_to_save: Dict[str, np.ndarray],  # todo 辞書とかの形式で保存できるようにしておきたい。
    result_dir: Path = Path("./result/"),
) -> None:

    wanna_save = input("\nsave result? [enter 'y' to save]")
    if wanna_save != "y":
        print("finish")
    if wanna_save == "y":
        if not (result_dir / "cross-section").exists:
            (result_dir / "cross-section").mkdir(parents=True)
        for name, data in data_to_save.items():
            with open(f"{name}.pickle", mode="wb") as pkl:
                pickle.dump(data, pkl)


if __name__ == "__main__":
    main()
