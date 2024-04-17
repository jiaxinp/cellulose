import pickle
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.interpolate import interp1d

from MyLibrary.ProcessedImageClass import Fiber


def main():
    fiber_objs, file_names = load_datapath(Fiber_Paths)
    for fiber, file_name in zip(fiber_objs, file_names):
        save_name = "-".join(re.split("[./]", file_name)[-4:-1])
        fig_wav = my_wavelet(fiber, f_min=4, f_max=32)

        plt.tight_layout()
        plt.savefig(f"{RESULT_WAV_DIR / save_name}.png")
        # plt.show()
        plt.close()

        fig_fft = my_fft(fiber)
        plt.savefig(f"{RESULT_FFT_DIR / save_name}.png")
        plt.close()

        # fig_interp = resample_constant_period(fiber)
        # plt.show()
        fig_wav2 = my_wavelet2(fiber, f_min=4, f_max=32)
        plt.tight_layout()
        plt.savefig(f"{RESULT_WAV_DIR / save_name}_ver2.png")
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


def my_wavelet(
        fiber: Fiber,
        wavelet_type: str = "cmor1.5-1.0",
        sampling_period=2 / 1024,
        f_min=4,
        f_max=64,
        f_interval=0.1,
        show_kink_posis: bool = False,
) -> plt.Figure:
    """
    visualize the wavelet analysis of periodic height change of Fiber.
    Note that  frequency should ge calculated in micrometer scale.
    :param fiber: fiber instance to be applied wavelet transformation
    :param wavelet_type: refer to PyWavelets document
    :param sampling_period: be care full about unit. The default unit is /μm
    :param f_min: Greater than 0
    :param f_max: Up to Nyquist frequency.
    :param f_interval: Frequency interval to be analyzed
    :param show_kink_posis:

    :return:
    """
    resampled_height, resampled_horizon = resample_constant_period(
        fiber, sampling_period_interp=2000 / 1024
    )
    # remove DC component
    signal_height = resampled_height - np.mean(resampled_height)

    wav = pywt.ContinuousWavelet(wavelet_type)
    # Convert frequency you wanted to analyse to scale
    fs = 1 / sampling_period  # sampling frequency
    f_nq = fs / 2  # Nyquist frequency
    if f_nq < f_max:
        print(
            f"Maximum frequency to be analysed should be lower than Nyquist frequency.\n"
            f"Maximum frequency was adjusted to {f_nq}"
        )
        f_max = f_nq

    frequencies = (
            np.arange(f_min, f_max + f_interval, f_interval) / fs
    )  # Normalize the frequency by the sampling frequency.(up to the maximum Nyquist frequency)
    scales = pywt.frequency2scale(wav, frequencies)  # Converts frequency to scale
    scales = scales[
             ::-1
             ]  # if not turned over, the lower frequencies will appear on top of wav matrix.

    cwtmatr, freqs = pywt.cwt(signal_height, scales=scales, wavelet=wav)

    fig_width = 6 * fiber.horizon[-1] / 500
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 10))

    axes[0].plot(resampled_horizon, resampled_height)
    image = axes[1].imshow(abs(cwtmatr), aspect="auto")
    image_shape = image.get_array().shape

    num_yticks = 9
    xticks_interval = 100
    num_xticks = len(np.arange(0, resampled_horizon[-1], xticks_interval))
    xticks_posis_image = np.linspace(
        0,
        (num_xticks - 1) * xticks_interval * len(resampled_horizon) / resampled_horizon[-1],
        num_xticks,
    )
    xticks_labels = np.arange(0, resampled_horizon[-1], xticks_interval)

    axes[0].set_xlim(0, resampled_horizon[-1])
    axes[0].set_xticks(np.arange(0, resampled_horizon[-1], xticks_interval))
    axes[0].set_yticks(np.arange(0, 4.5, 0.5))
    axes[0].set_ylabel("Height (nm)")
    axes[1].set_xlabel("Length ($nm$)")
    axes[1].set_ylabel("Frequency ($1/\mu m$)")
    axes[1].set_xticks(
        xticks_posis_image,
        xticks_labels,
    )
    axes[1].set_yticks(
        np.linspace(0, image_shape[0], num_yticks)[::-1], np.linspace(f_min, f_max, num_yticks)
    )

    return fig


def my_wavelet2(
        fiber: Fiber,
        wavelet_type: str = "cmor1.5-1.0",
        sampling_period=2 / 1024,
        f_min=4,
        f_max=64,
        f_interval=0.1,
        show_kink_posis: bool = False,
) -> plt.Figure:
    """
    visualize the wavelet analysis of periodic height change of Fiber.
    Note that  frequency should ge calculated in micrometer scale.
    In this function, the heatmap of wavelet transform and height profile were draw on the same axes.
    :type wavelet_type:
    :param fiber: fiber instance to be applied wavelet transformation
    :param wavelet_type: refer to PyWavelets document
    :param sampling_period: be care full about unit. The default unit is /μm
    :param f_min: Greater than 0
    :param f_max: Up to Nyquist frequency.
    :param f_interval: Frequency interval to be analyzed
    :param show_kink_posis:
    :return:
    """
    sampling_period_interp = 2000 / 1024
    resampled_height, resampled_horizon = resample_constant_period(
        fiber, sampling_period_interp=sampling_period_interp
    )
    # remove DC component
    signal_height = resampled_height - np.mean(resampled_height)

    wav = pywt.ContinuousWavelet(wavelet_type)
    # Convert frequency you wanted to analyse to scale
    fs = 1 / sampling_period  # sampling frequency
    f_nq = fs / 2  # Nyquist frequency
    if f_nq < f_max:
        print(
            f"Maximum frequency to be analysed should be lower than Nyquist frequency.\n"
            f"Maximum frequency was adjusted to {f_nq}"
        )
        f_max = f_nq

    frequencies = (
            np.arange(f_min, f_max + f_interval, f_interval) / fs
    )  # Normalize the frequency by the sampling frequency.(up to the maximum Nyquist frequency)
    scales = pywt.frequency2scale(wav, frequencies)  # Converts frequency to scale
    scales = scales[
             ::-1
             ]  # if not turned over, the lower frequencies will appear on top of wav matrix.

    cwtmatr, freqs = pywt.cwt(signal_height, scales=scales, wavelet=wav)

    fig_width = 6 * fiber.horizon[-1] / 500
    fig, ax1 = plt.subplots(1, 1, figsize=(fig_width, 5))

    image = ax1.imshow(
        abs(cwtmatr), aspect="auto", extent=(0, resampled_horizon[-1], f_min, f_max)
    )
    image_shape = image.get_array().shape

    ax2 = ax1.twinx()
    scaling_param = (f_max - f_min) / np.max(resampled_height)
    scaled_height = resampled_height * scaling_param
    ax2.plot(resampled_horizon, scaled_height, c="white")
    ax2.set_ylim(0, 4 * scaling_param)
    ax2.set_yticks(np.linspace(0, 4 * scaling_param, 9), np.linspace(0, 4, 9))
    ax2.set_ylabel("Height ($nm$)")

    xticks_interval = 100
    num_xticks = len(np.arange(0, resampled_horizon[-1], xticks_interval))
    xticks_posis_image = np.linspace(
        0,
        (num_xticks - 1) * xticks_interval * image_shape[1] / resampled_horizon[-1],
        num_xticks,
    )
    xticks_labels = np.arange(0, resampled_horizon[-1], xticks_interval)
    #
    ax1.set_xlabel("Length ($nm$)")
    ax1.set_ylabel("Frequency ($1/\mu m$)")
    ax1.set_xticks(
        xticks_posis_image
        * sampling_period_interp,  # todo I have no idea why I need to multiply sampling_period_interp
        xticks_labels,
    )
    # num_yticks = 9
    # ax1.set_yticks(
    #     np.linspace(0, image_shape[0], num_yticks)[::-1], np.linspace(f_min, f_max, num_yticks)
    # )

    return fig


def my_fft(fiber: Fiber, sampling_period=2 / 1024) -> plt.Figure:
    resampled_height, resampled_horizon = resample_constant_period(
        fiber, sampling_period_interp=2000 / 1024
    )
    # remove DC component
    sig = resampled_height - np.mean(resampled_height)
    # フーリエ変換
    sig_fft = np.fft.fft(sig)

    # 周波数スケール
    freq = np.fft.fftfreq(len(sig), d=sampling_period)

    # パワースペクトル (振幅スペクトルの2乗)
    power_spec = np.abs(sig_fft) ** 2

    # 結果をプロット
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 8))

    # 元の信号
    ax1.plot(resampled_horizon, resampled_height)
    ax1.set_xlabel("Length ($nm$)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Height profile")

    # パワースペクトル
    ax2.plot(freq, power_spec)
    ax2.set_xlabel("Frequency ($1/\mu m$)")
    ax2.set_ylabel("Power")
    # ax2.set_xlim([0, np.max(freq) / 2])  # ナイキスト周波数まで表示
    ax2.set_xlim([0, 60])
    ax2.set_title("Power Spectrum")

    plt.tight_layout()
    return fig


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


if __name__ == "__main__":
    Fiber_Paths = {
        # "../pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.007/18.pickle",
        # "../pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.007/9.pickle",
        # "../pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.007/36.pickle",
        # "../pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.008/15.pickle",
        # "../pickle_data/softholo_10mmol_220531/Fiber_data/softholo10mmol_220531.008/31.pickle",
        # "../pickle_data/softholo_5mmol_220524/Fiber_data/softholo_220524.004/60.pickle",
        # "../pickle_data/softholo_5mmol_220524/Fiber_data/softholo_220524.004/63.pickle",
        # "../pickle_data/softholo_5mmol_220524/Fiber_data/softholo_220524.004/71.pickle",
        # "../pickle_data/softholo_5mmol_220524/Fiber_data/softholo_220524.005/14.pickle",
        "../pickle_data/230606/Fiber_data/Holo48_sonication.000/20.pickle",
        "../pickle_data/230629/Fiber_data/230629_SBKP48_sonication.004/24.pickle"
    }

    RESULT_WAV_DIR = Path("../result/Wavelet")
    RESULT_FFT_DIR = Path("../result/FFT")
    if not RESULT_WAV_DIR.exists():
        RESULT_WAV_DIR.mkdir(parents=True)
    if not RESULT_FFT_DIR.exists():
        RESULT_FFT_DIR.mkdir(parents=True)

    main()
