from typing import Optional
import numpy as np
import pandas as pd
from scipy import signal
from lmfit.models import PseudoVoigtModel, PolynomialModel, GaussianModel
import cv2

from .ProcessedImageClass import ProcessedImage

class BG_Calibrator_shimadzu:
    '''
    Background calibrator for AFM images obtained by SPM-9600 (Shimadzu)
    '''
    def __init__(self, threshold_factor=3, fiber_detect_factor=10, noise_detect_factor=2,
                 savgol_window=51, savgol_polyorder=2, apply_median=True):
        """

        :param savgol_window: int
        :param savgol_polyorder: int
        :param apply_median: boolean
        """
        self.threshold_factor = threshold_factor
        self.fiber_detect_factor = fiber_detect_factor
        self.noise_detect_factor = noise_detect_factor

        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

        self.apply_median = apply_median

    def __call__(self, image: ProcessedImage):
        """

        :param image: ProcessedImage class instance

        :return: Image class instance with attribution of calibrated_image
        """

        self.dif_x, self.dif_y = self._difXY(image.original_image)
        self.histx, self.histy, self.outx, self.outy = self._bg_fit(self.dif_x, self.dif_y)
        self.tri_difx, self.tri_dify = self._dif_sep(self.dif_x, self.dif_y, self.outx, self.outy)
        self.tri_difx_fill, self.tri_dify_fill = self._extract_fiber(self.tri_difx, self.tri_dify)
        self.bg_only, self.bg_sm = self._bg_generate(image.original_image, self.tri_difx_fill, self.tri_dify_fill)
        calibrated_image = self._bg_calibrate(image.original_image, self.bg_sm)

        if self.apply_median:
            calibrated_image = cv2.medianBlur(calibrated_image.astype(np.float32), ksize=3)

        image.calibrated_image = calibrated_image

    @staticmethod
    def _difXY(image: np.ndarray):
        dif_x = image[:, 1:] - image[:, 0:-1]
        dif_y = image[1:, :] - image[0:-1, :]
        return dif_x, dif_y

    @staticmethod
    def _bg_fit(dif_x, dif_y, bin_n=150):
        histx = np.histogram(np.ravel(dif_x), bins=bin_n)
        histy = np.histogram(np.ravel(dif_y), bins=bin_n)
        h_arrayx = (histx[1][1:] + histx[1][:-1]) / 2
        h_arrayy = (histy[1][1:] + histy[1][:-1]) / 2
        # define the model
        bg = PolynomialModel(prefix='bg_', degree=1)
        pV1 = GaussianModel(prefix='pv1_')
        model = pV1 + bg
        # 　initial parameters
        pars = (bg.make_params())
        pars['bg_c0'].set(0)
        pars['bg_c1'].set(0)
        pars.update(pV1.make_params())
        pars['pv1_amplitude'].set(len(dif_x) * len(dif_x) / 20)
        pars['pv1_center'].set(np.median(dif_x))
        pars['pv1_sigma'].set(0.1)
        outx = model.fit(histx[0], pars, x=h_arrayx)
        outy = model.fit(histy[0], pars, x=h_arrayy)
        return histx, histy, outx, outy


    def _dif_sep(self, dif_x, dif_y, outx, outy):
        outx_min = outx.best_values['pv1_center'] - self.threshold_factor * outx.best_values['pv1_sigma']
        outx_max = outx.best_values['pv1_center'] + self.threshold_factor * outx.best_values['pv1_sigma']
        outy_min = outy.best_values['pv1_center'] - self.threshold_factor * outy.best_values['pv1_sigma']
        outy_max = outy.best_values['pv1_center'] + self.threshold_factor * outy.best_values['pv1_sigma']
        tri_difx = np.where(dif_x < outx_min, -1, 0) + np.where(dif_x > outx_max, 1, 0)
        tri_dify = np.where(dif_y < outy_min, -1, 0) + np.where(dif_y > outy_max, 1, 0)
        return tri_difx, tri_dify

    def _extract_fiber(self, tri_difx, tri_dify):
        tri_difx_fill = np.zeros(tri_difx.shape)
        for j in range(tri_difx.shape[0] - 1):
            l = list([tri_difx[j, 0]])
            arg = list([0])
            for i in range(tri_difx.shape[1] - 1):
                if (tri_difx[j, (i + 1)] - tri_difx[j, i]) == 0:
                    continue
                else:
                    l.append(tri_difx[j, (i + 1)])
                    arg.append(i + 1)
            for i in range(len(l) - 3):
                if np.array_equal(l[i:(i + 3)], [1, 0, -1]):
                    if (arg[i + 2] - arg[i + 1]) < self.fiber_detect_factor:
                        tri_difx_fill[j, arg[i]:(arg[i + 3] - 1)] = 1
                elif np.array_equal(l[i:(i + 2)], [1, -1]):
                    if (arg[i + 2] - arg[i]) > self.noise_detect_factor:
                        tri_difx_fill[j, arg[i]:(arg[i + 2] - 1)] = 1

        tri_dify_fill = np.zeros(tri_dify.shape)
        for j in range(tri_dify.shape[1] - 1):
            l = list([tri_dify[0, j]])
            arg = list([0])
            for i in range(tri_dify.shape[0] - 1):
                if (tri_dify[(i + 1), j] - tri_dify[i, j]) == 0:
                    continue
                else:
                    l.append(tri_dify[(i + 1), j])
                    arg.append(i + 1)
            for i in range(len(l) - 3):
                if np.array_equal(l[i:(i + 3)], [1, 0, -1]):
                    if (arg[i + 2] - arg[i + 1]) < self.fiber_detect_factor:
                        tri_dify_fill[arg[i]:(arg[i + 3] - 1), j] = 1
                elif np.array_equal(l[i:(i + 2)], [1, -1]):
                    if (arg[i + 2] - arg[i]) > self.noise_detect_factor:
                        tri_dify_fill[arg[i]:(arg[i + 2] - 1), j] = 1
        return tri_difx_fill, tri_dify_fill

    def _bg_generate(self, original, tri_difx_fill, tri_dify_fill):
        bg_only = np.where((np.abs(tri_difx_fill[1:, :]) + np.abs(tri_dify_fill[:, 1:])) == 0, original[1:, 1:],
                           float('nan'))
        bg_int = np.array(pd.DataFrame(bg_only).interpolate('spline', order=2, limit_direction='both'))
        bg_sm = signal.savgol_filter(bg_int, self.savgol_window, self.savgol_polyorder)
        return bg_only, bg_sm

    @staticmethod
    def _bg_calibrate(original, bg_sm):
        height_bgcalib = original[1:, 1:] - bg_sm
        return height_bgcalib