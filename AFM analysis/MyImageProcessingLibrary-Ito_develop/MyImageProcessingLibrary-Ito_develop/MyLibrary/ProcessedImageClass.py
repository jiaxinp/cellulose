import sys

sys.path.append( r'C:\Users\Jess\Dropbox\UTokyo\Research\Cellulose\code\AFM analysis\MyImageProcessingLibrary-Ito_develop\MyImageProcessingLibrary-Ito_develop\MyLibrary')
from typing import Optional, Union

import cv2
import numpy as np

import imptools

from FiberClass import Fiber


class ProcessedImage:
    def __init__(self, original_AFM: np.ndarray, name: str, size_per_pixel: float = 2.0):
        """
        :param name: str
        :param original_AFM: np.ndarray (2D) that represents original AFM image
        :param size_per_pixel: float
        """
        self.name: str = name
        self.original_image: np.ndarray = original_AFM
        self.size_per_pixel: float = size_per_pixel

        self.calibrated_image: Optional[np.ndarray] = None  # This variable should be set by BG_Calibrator
        self.binarized_image: Optional[np.ndarray] = None  # This variable should be set by Segmentater
        self.skeleton_image: Optional[np.ndarray] = None  # This variable should be set by Skeletonizer

        self.nLabels: Optional[int] = None  # This variable should be set by Skeletonizer
        self.data: Optional[tuple] = None  # x, y, w, h, area = data[label]. This variable should be set by Skeletonizer
        self.label_image: Optional[np.ndarray] = None  # This variable should be set by Skeletonizer

        self.bp = None  # this variable should be set by Skeletonizer
        self.ep = None  # this variable should be set by Skeletonizer

        self.all_kink_coordinates: Optional[
            tuple[np.ndarray, np.ndarray]] = None  # this variable should be set by KinkDetecter
        self.all_kink_angles: Optional[np.ndarray] = None  # this variable should be set by KinkDetecter
        self.decomposed_point_coordinates: Optional[np.ndarray] = None  # this variable should be set by KinkDetecter

    def fibers_in_image(self) -> list[Fiber]:
        return self._generate_fiber_instances(self.skeleton_image)

    def specific_height_fibers(self,
                               lower_height: Union[int, float],
                               upper_height: Union[int, float],
                               include_lower_limit=True,
                               include_upper_limit=True) -> list[Fiber]:
        if include_lower_limit and include_upper_limit:
            skeleton_image = np.where(
                (self.calibrated_image >= lower_height) & (self.calibrated_image <= upper_height) & self.skeleton_image, 1,
                0).astype(np.uint8)
        elif include_lower_limit:
            skeleton_image = np.where(
                (self.calibrated_image >= lower_height) & (self.calibrated_image < upper_height) & self.skeleton_image, 1,
                0).astype(np.uint8)
        elif include_upper_limit:
            skeleton_image = np.where(
                (self.calibrated_image > lower_height) & (self.calibrated_image <= upper_height) & self.skeleton_image, 1,
                0).astype(np.uint8)
        else:
            skeleton_image = np.where(
                (self.calibrated_image > lower_height) & (self.calibrated_image < upper_height) & self.skeleton_image, 1,
                0).astype(np.uint8)
        return self._generate_fiber_instances(skeleton_image)

    def _generate_fiber_instances(self, skeleton_image: np.ndarray) -> list[Fiber]:
        no_bp_skel = imptools.remove_bp(skeleton_image)
        no_Lcorner_skel = imptools.remove_Lcorner(no_bp_skel)
        no_square_skel = imptools.remove_square_branches(no_Lcorner_skel)
        nLabels, label_image, data, center = cv2.connectedComponentsWithStats(no_square_skel)
        fiber_instances = []
        for label in range(1, nLabels):
            x, y, w, h, size = data[label]
            target_image = np.where(label_image == label, 1, 0)
            xtrack_prcimg, ytrack_prcimg = imptools.tracking(target_image)
            xtrack = xtrack_prcimg - x
            ytrack = ytrack_prcimg - y
            horizon = imptools.convert_track_to_distance(xtrack, ytrack, self.size_per_pixel)
            height = self.calibrated_image[ytrack_prcimg, xtrack_prcimg]
            fiber_image = self.calibrated_image[y: y + h, x: x + w].copy()
            kink_indices, decomposed_point_indices = \
                self._calc_kink_and_decomposed_point_indices(xtrack_prcimg, ytrack_prcimg)
            ep_indices = self._calc_endpoint_indices(xtrack_prcimg, ytrack_prcimg)
            kink_angles = self._get_kink_angles_in_fiber(xtrack_prcimg, ytrack_prcimg, kink_indices)

            fiber = Fiber(fiber_image, data[label], xtrack, ytrack, horizon, height, kink_indices,ep_indices, kink_angles,
                          decomposed_point_indices)
            fiber_instances.append(fiber)
        return fiber_instances

    def _calc_kink_and_decomposed_point_indices(self,
                                                xtrack_prcimg,
                                                ytrack_prcimg) -> tuple[np.ndarray, np.ndarray]:
        all_kink_coor_x, all_kink_coor_y = self.all_kink_coordinates
        kink_indices = []
        decomposed_point_indices = []
        for i in range(len(xtrack_prcimg)):
            x = xtrack_prcimg[i]
            y = ytrack_prcimg[i]
            if (x, y) in zip(all_kink_coor_x, all_kink_coor_y):
                kink_indices.append(i)
            if (x, y) in zip(self.decomposed_point_coordinates[0], self.decomposed_point_coordinates[1]):
                decomposed_point_indices.append(i)
        return np.array(kink_indices), np.array(decomposed_point_indices)

    def _calc_endpoint_indices(self, xtrack_prcimg, ytrack_prcimg) -> np.ndarray:
        """
        Fiberインスタンスのhorizon, height中でepを表す要素にインデックスを計算する.
        epを含まない場合は空のndarrayを返す.
        """
        all_ep_y, all_ep_x = np.where(self.ep)
        ep_indices = []
        for i in range(len(xtrack_prcimg)):
            x = xtrack_prcimg[i]
            y = ytrack_prcimg[i]
            if (x, y) in zip(all_ep_x, all_ep_y):
                ep_indices.append(i)
        return np.array(ep_indices)

    def _get_kink_angles_in_fiber(self, xtrack_prcimg, ytrack_prcimg, kink_indices) -> np.ndarray:
        all_kink_x, all_kink_y = self.all_kink_coordinates
        kink_angles = []
        for i in kink_indices:
            kink_x = xtrack_prcimg[i]
            kink_y = ytrack_prcimg[i]
            for j in range(len(all_kink_x)):
                if kink_x == all_kink_x[j] and kink_y == all_kink_y[j]:
                    kink_angles.append(self.all_kink_angles[j])
        return np.array(kink_angles)

    def get_height_distribution(self):
        return self.calibrated_image[np.where(self.skeleton_image)]
