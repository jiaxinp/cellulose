import pickle
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf

from MyLibrary.ProcessedImageClass import Fiber


def main():
    fiber_objs, file_names = load_datapath(Fiber_Paths)
    for fiber, file_name in zip(fiber_objs, file_names):
        if len(fiber.height) < WINDOW_SIZE:
            continue

        moving_acorr = calc_moving_acorr(
            fiber.height, window_size=WINDOW_SIZE, window_step=WINDOW_STEP
        )
        sliced_signals = np.array(
            [
                fiber.height[s : s + WINDOW_SIZE]
                for s in range(0, len(fiber.height) - WINDOW_SIZE, WINDOW_STEP)
            ]
        )

        save_name = "-".join(re.split("[./]", file_name)[-4:-1])
        fig, ax = plt.subplots(len(moving_acorr), 2, figsize=(6, 4 * len(moving_acorr)))
        for i, acorr in enumerate(moving_acorr):
            ax[i, 0].plot(np.arange(0, 2 * WINDOW_SIZE, 2), acorr, lw=1.5)
            ax[i, 0].set_ylim(-0.4, 1)
            ax[i, 1].plot(
                np.arange(0, 2 * WINDOW_SIZE, 2), sliced_signals[i], c="firebrick", lw=1.5
            )
            ax[i, 1].set_ylim(0, 4)

        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR/save_name}.svg")
        plt.close()


def load_datapath(paths: Iterable[str]) -> tuple[list[Fiber], list[str]]:
    fiber_objs = []
    file_names = []
    for path in paths:
        file_names.append(path)
        with open(path, "rb") as f:
            fiber = pickle.load(f)
            fiber_objs.append(fiber)
    return fiber_objs, file_names


def calc_moving_acorr(signal, window_size=200, window_step=10):
    """
    np.convolveでいうところのmodeはvaild方式を採用？
    """
    if len(signal) < window_size:
        print("signal is too short.")
        return

    sliced_signals = np.array(
        [signal[s : s + window_size] for s in range(0, len(signal) - window_size, window_step)]
    )
    moving_acorrs = np.apply_along_axis(acf, 1, sliced_signals, nlags=window_size, fft=False)

    return moving_acorrs


if __name__ == "__main__":
    Fiber_Paths = {
        "./pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.007/18.pickle",
        "./pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.007/9.pickle",
        "./pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.007/36.pickle",
        "./pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.008/15.pickle",
        "./pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.008/31.pickle",
        "./pickle_data/softholo_5mmol_220524/Fiber_data/softholo_220524.004/60.pickle",
        "./pickle_data/softholo_5mmol_220524/Fiber_data/softholo_220524.004/63.pickle",
        "./pickle_data/softholo_5mmol_220524/Fiber_data/softholo_220524.004/71.pickle",
        "./pickle_data/softholo_5mmol_220524/Fiber_data/softholo_220524.005/14.pickle",
    }

    WINDOW_SIZE = 100
    WINDOW_STEP = 10

    RESULT_DIR = Path("./result/Periodic_height_analysis")
    if not RESULT_DIR.exists():
        RESULT_DIR.mkdir()

    main()
