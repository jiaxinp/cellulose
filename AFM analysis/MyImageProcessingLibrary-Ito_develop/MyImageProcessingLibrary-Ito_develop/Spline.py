import itertools
import math
import pickle
import pprint
from pathlib import Path
from typing import Dict, Generator, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline, interp1d

from MyLibrary.ProcessedImageClass import Fiber

# todo ゼミ用にめちゃくちゃ汚いコードになっている。後日書き直し。


def main():
    fibers = choose_PrcImg_data(
        direct_fiber_path=Path(
            # "./pickle_data/Alkali_stirring_1d/Fiber_data/201207.004/49.pickle"
            "./pickle_data/test/Fiber_data/201207.004/71.pickle"
        )
    )
    for fiber in fibers:
        cubic_spline = CubicSpline(fiber.ytrack, fiber.xtrack)

        relmax_indices = signal.argrelmax(fiber.height, order=4)
        relmax_indices2 = np.intersect1d(
            np.where(fiber.height > 3), relmax_indices
        )  # 最初に検出したピークのうち高さ3以下の場所はカット

        peak_indices = integrate_close_peaks(fiber.height, relmax_indices2)
        manual_peak_indices = list(peak_indices)
        manual_peak1 = 97
        manual_peak2 = 305
        manual_peak_indices.append(manual_peak1)
        manual_peak_indices.append(manual_peak2)
        manual_peak_indices.sort()

        fig, axes = plt.subplots(2, 1)
        axes.ravel()
        axes[0].scatter(fiber.ytrack, fiber.xtrack)
        axes[0].imshow(fiber.AFM_image, cmap="afmhot", vmax=5.5, vmin=-0.5)
        axes[1].plot(fiber.horizon, fiber.height)
        axes[1].scatter(
            fiber.horizon[manual_peak_indices],
            fiber.height[manual_peak_indices],
            c="red",
            s=18,
        )
        plt.show()

        r_sum = np.zeros_like(np.arange(-3, 3, 0.01))
        for n in range(9):
            x_coor, y_coor = rotate_cross_sec(fiber.height, manual_peak_indices, start_peaks=n)
            x_coor = np.array(x_coor)
            y_coor = np.array(y_coor)
            r, theta = getRD(x_coor, y_coor)
            theta_sorted = np.sort(theta)
            r_key = np.argsort(theta)
            r_sorted = r[r_key]
            interp_func = interp1d(theta_sorted, r_sorted)
            theta_new = np.arange(-3, 3, 0.01)
            r_new = interp_func(theta_new)
            r_sum += r_new
            fig2, ax2 = plt.subplots(2, 1, figsize=(4, 8), subplot_kw=dict(polar=True))
            ax2.ravel()
            ax2[0].plot(theta_new, r_new, lw=2)
            ax2[0].set_rticks([0.5, 1, 1.5, 2])
            ax2[1].plot(theta_sorted, r_sorted, lw=2)
            ax2[1].set_rticks([0.5, 1, 1.5, 2])
            #         ax4.plot(coor_nishi[0], coor_nishi[1], lw=2)
            # ax2.set_xlim(-2, 2)
            # ax2.set_ylim(-2, 2)
            plt.show()
        r_average = r_sum / 9

        simurator_nishi = AFM_Simulator("18chain_nishiyama")
        coor_nishi = simurator_nishi.show_shape(162)

        simurator_daicho = AFM_Simulator("18chain_daicho")
        coor_daicho = simurator_daicho.show_shape(97)

        daicho_r, daicho_theta = getRD(-coor_nishi[0], coor_nishi[1])

        fig3, ax3 = plt.subplots(1, 1, figsize=(4, 4), subplot_kw=dict(polar=True))
        ax3.plot(theta_new, r_average)
        # ax3.plot(daicho_theta, daicho_r)
        ax3.set_rticks([0.5, 1, 1.5, 2])
        plt.show()


# wanna_save_result()  # 上記までに処理したデータを辞書の形でここに渡す。


class AFM_Simulator:
    def __init__(self, model, tip_r=2, tap_dist=2):
        self.model = model
        self.tip_r = tip_r
        self.tap_dist = tap_dist
        """
        断面モデルの頂点の座標を入力する
        1βの格子座標で(x座標のndarray, y座標のndarray)を自分で入力し、lattice2orthogonalで直交系に変換している
        """
        self.formation_dict = {
            "18chain_nishiyama": self.lattice2orthogonal(
                np.array([-0.25, 0.25, 1.5, 1.5, -0.25, -0.75, -2, -2]),
                np.array([-0.25, -0.25, 1, 2, 3.75, 3.75, 2.5, 1.5]),
            ),
            "18chain_daicho": self.lattice2orthogonal(
                np.array([-0.25, 1.25, 2.5, 0.75, -0.75, -2]),
                np.array([-0.25, -0.25, 1, 2.75, 2.75, 1.5]),
            ),
        }

    @staticmethod
    def lattice2orthogonal(lattice_x, lattice_y):
        """
        単位はnm
        入力：セルロ―スⅠβ結晶格子を基底とした座標のndarray
        後に回転するとき用に重心を原点に変換
        出力：直交座標のndarray（小数第3位まで）
        """
        a = 0.8201
        b = 0.7784
        gamma = math.radians(96.5)

        gx = np.mean(lattice_x)
        gy = np.mean(lattice_y)
        x = np.round(a * (lattice_x - gx) + round(math.cos(gamma), 5) * b * (lattice_y - gy), 4)
        y = np.round(round(math.sin(gamma), 5) * b * (lattice_y - gy), 4)
        return x, y

    @staticmethod
    def rotation(x, t, deg=True):
        # ラジアンでも弧度法でも良い(default: 弧度法)
        if deg == True:
            t = np.deg2rad(t)
        a = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        ax = np.dot(a, x)
        return ax

    def run(self):
        max_heights = []
        min_heights = []
        angles = np.arange(0, 181)
        initial_taps = np.linspace(0, self.tap_dist, 100)
        for angle in angles:
            height_list = []
            for initial_tap in initial_taps:
                single_scan = self._single_line_height(angle, initial_tap)
                thin_height = self._thinning(single_scan)
                height_list.append(thin_height)
            max_heights.append(max(height_list))
            min_heights.append(min(height_list))
        return max_heights, min_heights

    def visualize(self):
        return self._single_line_height()

    def _single_line_height(self, angle, initial_tap):
        """
        半径rの球がCNFに衝突するときの球の中心の軌跡を考える
        球がCNFの
        1. 辺に衝突する場合
           CNFの辺を距離rだけ平行移動させる
        2. 頂点に衝突する場合
           CNFの頂点を中心に半径rの円を考える

        水平位置を返していない
        """
        tapping_posi = np.arange(-8 + initial_tap, 8, self.tap_dist)

        coordinate = np.vstack(self.formation_dict[self.model])
        coor_rotated = self.rotation(coordinate, angle)

        cnf_bottom = np.min(coor_rotated[1])
        # 0. 辺に衝突する場合の中心が描く直線の軌跡を求める
        edge_vector = np.roll(coor_rotated, shift=-1, axis=1) - coor_rotated
        moves = (
            self.tip_r * self.rotation(edge_vector, -90) / np.linalg.norm(edge_vector, axis=0)
        )  # 平行移動の方向

        equations = []  # 球の中心が描く直線の方程式
        x_ranges = []  # 方程式の定義域
        for i, move in enumerate(moves.T):
            x1, x2 = (
                coor_rotated[0][i] + move[0],
                np.roll(coor_rotated, shift=-1, axis=1)[0][i] + move[0],
            )
            y1, y2 = (
                coor_rotated[1][i] + move[1],
                np.roll(coor_rotated, shift=-1, axis=1)[1][i] + move[1],
            )

            x_array = np.array([x1, x2])
            y_array = np.array([y1, y2])
            equation = np.poly1d(np.polyfit(x_array, y_array, 1))
            equations.append(equation)
            x_ranges.append((np.min(x_array), np.max(x_array)))

        # スキャンを1回実行
        single_scan = []
        for x in tapping_posi:
            height_candidate = [0]  # CNFにぶつからなかったら、高さ0

            # 1. 辺に衝突する場合
            for equation, x_range in zip(equations, x_ranges):
                if x_range[0] <= x and x <= x_range[1]:
                    height_candidate.append(equation(x) - self.tip_r - cnf_bottom)
            # 2. 頂点に衝突する場合
            for vertex in coor_rotated.T:
                if abs(x - vertex[0]) <= self.tip_r:
                    cy = vertex[1] + math.sqrt(self.tip_r**2 - (x - vertex[0]) ** 2)
                    height_candidate.append(cy - self.tip_r - cnf_bottom)

            single_scan.append(max(height_candidate))
        return single_scan

    def _thinning(self, single_scan):
        heights_array = np.array(single_scan)
        fiber_heights = heights_array[heights_array > 0]

        if len(fiber_heights) % 2 == 1:
            thin_index = len(fiber_heights) // 2
            thin_height = fiber_heights[thin_index]

        else:
            thin_index = int(len(fiber_heights) / 2)
            thin_height = (fiber_heights[thin_index] + fiber_heights[thin_index - 1]) / 2
        return thin_height

    def show_shape(self, angle):
        coordinate = np.vstack(self.formation_dict[self.model])
        coordinate = np.hstack((coordinate, coordinate[:, 0:1]))
        coor_rotated = self.rotation(coordinate, angle)
        return coor_rotated


def lattice2orthogonal(lattice_x, lattice_y):
    """
    単位はnm
    入力：セルロ―スⅠβ結晶格子を基底とした座標のndarray
    後に回転するとき用に重心を原点に変換
    出力：直交座標のndarray（小数第3位まで）
    """
    a = 0.8201
    b = 0.7784
    gamma = math.radians(96.5)

    gx = np.mean(lattice_x)
    gy = np.mean(lattice_y)
    x = np.round(a * (lattice_x - gx) + round(math.cos(gamma), 5) * b * (lattice_y - gy), 4)
    y = np.round(round(math.sin(gamma), 5) * b * (lattice_y - gy), 4)
    return x, y


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


def rotation(x, t, deg=True):
    # ラジアンでも弧度法でも良い(default: 弧度法)
    if deg:
        t = np.deg2rad(t)
    a = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    ax = np.dot(a, x)
    return ax


def rotate_cross_sec(height_profile, peak_indices, start_peaks=0):
    """
    プロットのためのx,y座標のリスト返す
    Input:
        height_profile (list)
        peak_indices (ndarray)

    return:
        x_coor, y_coor (tuple): tuple of two lists. each points on the edge of CNF cross-section
    """
    p0 = peak_indices[start_peaks]
    p1 = peak_indices[start_peaks + 1]
    p2 = peak_indices[start_peaks + 2]
    # print(f"p0: {p0}")
    # print(f"p1: {p1}")

    x_coor = []
    y_coor = []

    # 最初の180度回転
    for rotation_index, height in enumerate(height_profile[p0:p1]):
        rotation_angle = 180 * rotation_index / (p1 - p0)
        x, y = rotation(np.array([0, height / 2]), rotation_angle)
        x_coor.append(x)
        y_coor.append(y)

    for rotation_index, height in enumerate(height_profile[p1:p2]):
        rotation_angle = 180 * rotation_index / (p2 - p1)
        x, y = rotation(np.array([0, height / 2]), rotation_angle)
        x_coor.append(-x)
        y_coor.append(-y)

    return x_coor, y_coor


def getRD(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


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
