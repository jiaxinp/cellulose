# todo まだ完了していない。test_BG_Calibraterが終了したら取り掛かる
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import canny
from skimage.filters import threshold_local
from skimage.morphology import binary_closing, binary_dilation, binary_erosion
from skimage.transform import probabilistic_hough_line

from MyLibrary.BG_Calibrator import BG_Calibrator
from MyLibrary.ProcessedImageClass import ProcessedImage
from MyLibrary.Segmentater import Segmentater

IMAGE_SIZE = 1024  # image size (pixel)
SCALE = 2000  # image size (nm)
GLOBAL_THRESH = 0.3  # threshold for global binarization
WIN_LOCALBIN = 17  # window size for local binarization
AREA_MIN = 200  # minimum area for detecting as object
AREA_MIN2 = 200  # use after erosing
H_LENGTH = 12  # Houghのパラメータ
HS_RATIO = 0.3  # Hough消去の面積比

filepath = Path("../data/4pass_data_ver2/")
flist = list(filepath.glob("*.txt"))
flist.sort()

datatxt = np.loadtxt(flist[0], skiprows=1)
data2D = np.reshape(datatxt, (IMAGE_SIZE, IMAGE_SIZE))
myimage = ProcessedImage(data2D, name="test")
calibrater = BG_Calibrator()
calibrater(myimage)

# Combination of global and local binarization
binary_global = myimage.calibrated_image > GLOBAL_THRESH
dataBin = np.where(binary_global == True, 1, 0)
local_thresh = threshold_local(myimage.calibrated_image, WIN_LOCALBIN)
binary_local = myimage.calibrated_image > local_thresh
dataBin_gl = binary_local * binary_global

# remove the small components
nLabels1, labelImages1, data1, center1 = cv2.connectedComponentsWithStats(np.uint8(dataBin_gl), 8)
dataBin_gl2 = labelImages1.copy()
for i in range(1, nLabels1):
    if (data1[:, 4] < AREA_MIN)[i]:
        dataBin_gl2[np.where(labelImages1 == i)] = 0
dataBin_gl2 = np.where(dataBin_gl2 > 0, 1, 0)
dataBin_gl2_sm = cv2.medianBlur(dataBin_gl2.astype(np.float32), ksize=3)  ## smoothing the binary
nLabels2, labelImages2, data2, center2 = cv2.connectedComponentsWithStats(
    np.uint8(dataBin_gl2_sm), 8
)

# Detect linear object by probablistic Hough transform
dataBin_hough = np.copy(dataBin_gl2_sm)

for m in range(1, nLabels2):
    Bin_n = np.where(labelImages2 == m, 1, 0)
    Thin_n = canny(Bin_n, sigma=0, low_threshold=0, high_threshold=1)
    Hough = probabilistic_hough_line(Thin_n, line_length=H_LENGTH, line_gap=1)
    if len(Hough) == 0:
        dataBin_hough[np.where(labelImages2 == m)] = 0
    else:
        total_length = []  # total length detected by probablistic Hough
        for line in Hough:
            p0, p1 = line
            total_length.append(np.linalg.norm(np.array(p1) - np.array(p0)))
        total_length = sum(total_length)
        S_ratio = total_length / np.sum(Thin_n)  # ratio of detected lines
        if (
            S_ratio < HS_RATIO and np.sum(Bin_n) < 500
        ):  # keep the big components with the area larger than 500
            dataBin_hough[np.where(labelImages2 == m)] = 0

nLabels3, labelImages3, data3, center3 = cv2.connectedComponentsWithStats(
    np.uint8(dataBin_hough), 8
)

# erosion ->remove small -> dilation ->closing
data_erosion = binary_erosion(dataBin_hough)
nLabels_e, labelImages_e, data_e, center_e = cv2.connectedComponentsWithStats(
    np.uint8(data_erosion), 8
)
data_no_small = labelImages_e.copy()
for i in range(1, nLabels_e):
    if (data_e[:, 4] < AREA_MIN2)[i]:
        data_no_small[np.where(labelImages_e == i)] = 0
data_no_small = binary_dilation(data_no_small)
data_no_small = binary_closing(data_no_small)

nLabels4, labelImages4, data4, center4 = cv2.connectedComponentsWithStats(
    np.uint8(data_no_small), 8
)


def test_segmentater():
    # 最終結果のテスト
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    segmentater = Segmentater()
    bg_calibrater(image)
    segmentater(image)
    np.testing.assert_allclose(image.binarized_image, data_no_small)


def test_binaryzation():
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    segmentater = Segmentater()
    bg_calibrater(image)
    segmentater(image)
    np.testing.assert_allclose(segmentater.binary_image, dataBin_gl)


def test_no_small_binary():
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    segmentater = Segmentater()
    bg_calibrater(image)
    segmentater(image)
    np.testing.assert_allclose(segmentater.no_small_binary_image, dataBin_gl2_sm)


def test_no_linear_binary():
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    segmentater = Segmentater()
    bg_calibrater(image)
    segmentater(image)
    np.testing.assert_allclose(segmentater.no_linear_binary_image, dataBin_hough)


def test_no_connecting_binary():
    image = ProcessedImage(data2D, name="test")
    bg_calibrater = BG_Calibrator()
    segmentater = Segmentater()
    bg_calibrater(image)
    segmentater(image)
    np.testing.assert_allclose(segmentater.no_connecting_binary_image, data_no_small)
