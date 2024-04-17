import sys
sys.path.insert(0, '/Users/tomok/PycharmProjects/MyImageProcessingLibrary/MyLibrary')

from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union
from .ProcessedImageClass import Fiber
import numpy as np
import pywt
from scipy.interpolate import interp1d


class WaveletAnalyzer:

    @staticmethod
    def resample_constant_period(
            fiber: Fiber, sampling_period_interp=2000 / 1024
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate height profile of Fiber at constant sampling period in linear manner.
        :param fiber:
        :param sampling_period_interp: be careful about unit. The default unis is /nm
        :return:
        """
        new_horizon: np.ndarray = np.arange(0, fiber.horizon[-1], sampling_period_interp)

        # Perform linear interpolation on the original data
        interpolator = interp1d(fiber.horizon, fiber.height, kind="linear", fill_value="extrapolate")
        new_height: np.ndarray = interpolator(new_horizon)
        return new_height, new_horizon

    def calc_CWT_matrix(
            self, fiber: Fiber, wavelet_type: str, sampling_period, f_min, f_max, f_interval, show_height_profile
    ):
        pass
