# todo まだ改善していない。imptoolのテストが完了したら取り掛かる
from pathlib import Path

import cv2
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from MyLibrary.BG_Calibrator import BG_Calibrator
from MyLibrary.ProcessedImageClass import ProcessedImage

filepath = Path("../data/4pass_data_ver2/")
flist = list(filepath.glob("*.txt"))
flist.sort()

IMAGE_SIZE = 1024  # image size (pixel)
SCALE = 2000  # image size (nm)

datatxt = np.loadtxt(flist[0], skiprows=1)
data2D = np.reshape(datatxt, (IMAGE_SIZE, IMAGE_SIZE))

# bg correction in one line
data2D_cor = np.empty(data2D.shape)

# テストで比較するための変数たち
thresholds_low = list()
thresholds_high = list()
backgrounds = list()
splined_background = list()
for i in range(data2D.shape[1]):
    # remove outliers
    weight_q1 = stats.scoreatpercentile(data2D[i], 25)
    weight_q3 = stats.scoreatpercentile(data2D[i], 75)
    weight_iqr = weight_q3 - weight_q1
    o1 = weight_q1 - weight_iqr * 1.5
    o2 = weight_q3 + weight_iqr * 0

    thresholds_low.append(o1)
    thresholds_high.append(o2)

    data2d_bgc = (
        [0]
        + [j for j in range(1, len(data2D[i]) - 1) if o1 < data2D[i][j] < o2]
        + [len(data2D[i]) - 1]
    )
    backgrounds += data2d_bgc
    data2d_bg = [data2D[i][j] for j in data2d_bgc]

    #

    # determine the background by smoothing
    data2d_bg_sm = savgol_filter(data2d_bg, 31, 2)
    data2d_bg_sm2 = interp1d(data2d_bgc, data2d_bg_sm)(np.linspace(0, IMAGE_SIZE - 1, IMAGE_SIZE))
    splined_background.append(data2d_bg_sm2)

    data2D_cor[i] = data2D[i] - data2d_bg_sm2

    # apply median filter
dataMed = cv2.medianBlur(data2D.astype(np.float32), ksize=3)
data2D_cor_Med = cv2.medianBlur(data2D_cor.astype(np.float32), ksize=3)


def test_bgcalibrater():
    # 最終結果のテスト用
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    bg_calibrater(image)
    np.testing.assert_allclose(image.calibrated_image, data2D_cor_Med, rtol=0, atol=1e-1)


def test_thresholds_low():
    # 各行の閾値が以前のバージョンと一致しているか調べるテスト
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    bg_calibrater(image)
    np.testing.assert_allclose(
        bg_calibrater.threshold_low.reshape((-1,)), thresholds_low, rtol=0, atol=1e-6
    )


def test_thresholds_high():
    # 各行の閾値が以前のバージョンと一致しているか調べるテスト
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    bg_calibrater(image)
    np.testing.assert_allclose(
        bg_calibrater.threshold_high.reshape((-1,)), thresholds_high, rtol=0, atol=1e-4
    )


def test_get_background_filter():
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    bg_calibrater(image)
    bg_filter = np.where(bg_calibrater.background_filter)[1]
    np.testing.assert_allclose(bg_filter, backgrounds)


def test_savgol_smoothed_background():
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    bg_calibrater(image)
    np.testing.assert_allclose(
        bg_calibrater.background_smoothed, splined_background, rtol=0, atol=1e-7
    )


def test_final_result():
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    bg_calibrater(image)
    np.testing.assert_allclose(image.calibrated_image, data2D_cor_Med, rtol=0, atol=1e-7)
