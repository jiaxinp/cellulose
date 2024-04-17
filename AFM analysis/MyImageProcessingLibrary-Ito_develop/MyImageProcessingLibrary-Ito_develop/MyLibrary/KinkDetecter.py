import sys

sys.path.insert(0, '/Users/tomok/PycharmProjects/MyImageProcessingLibrary/MyLibrary')

import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Union, Tuple, List
from collections.abc import Iterable

from . import imptools
from .ProcessedImageClass import ProcessedImage, Fiber


class KinkDetecter:
    def __init__(self,
                 threshold_distance=6,
                 threshold_angle_from_decomposed_indices=5 * np.pi / 6,
                 threshold_angle_corner=5 * np.pi / 6,
                 k=5):
        """
        :param threshold_distance: threshold distance for binary decomposition
        :param threshold_angle_from_decomposed_indices: threshold angle for _detect_kink_from_decomposed_indices() function
        :param threshold_angle_corner: threshold angle for _detect_corner_indices() function
        :param k: the number of pixels to be used for _detect_corner_indices() function
        """
        self.threshold_distance = threshold_distance
        self.threshold_angle_from_decomposed_indices = threshold_angle_from_decomposed_indices
        self.threshold_angle_corner = threshold_angle_corner
        self.k = k

    def __call__(self, image: ProcessedImage):
        """
        :param image: ProcessedImage class instance
        :return: Image class instance with attribution of kink_indices
        """
        # todo ProcessedImageクラスにdecomposed_point_coordinatesを設定する。
        try:
            no_bp_skel = imptools.remove_bp(image.skeleton_image)
            no_Lcorner_skel = imptools.remove_Lcorner(no_bp_skel)
            no_square_skel = imptools.remove_square_branches(no_Lcorner_skel)
            nLabels, label_image, data, center = cv2.connectedComponentsWithStats(no_square_skel)

            all_kink_coordinate_x = []
            all_kink_coordinate_y = []
            all_kink_angles = []
            decomposed_point_x = []
            decomposed_point_y = []
            for label in range(1, nLabels):
                target_image = np.where(label_image == label, 1, 0)
                _xtrack, _ytrack = imptools.tracking(target_image)
                decomposed_indices = self._binary_decompose_simple(_xtrack, _ytrack, self.threshold_distance)
                kink_indices, kink_angles = self._detect_kink_from_decomposed_indices(_xtrack, _ytrack, decomposed_indices, self.threshold_angle_from_decomposed_indices)
                all_kink_coordinate_x.extend([_xtrack[i] for i in kink_indices])
                all_kink_coordinate_y.extend([_ytrack[i] for i in kink_indices])
                all_kink_angles.extend(list(kink_angles))
                decomposed_point_x.extend([_xtrack[i] for i in decomposed_indices])
                decomposed_point_y.extend([_ytrack[i] for i in decomposed_indices])

            image.all_kink_coordinates = (np.array(all_kink_coordinate_x), np.array(all_kink_coordinate_y))
            image.all_kink_angles = np.array(all_kink_angles)
            image.decomposed_point_coordinates = (np.array(decomposed_point_x), np.array(decomposed_point_y))


        except Exception as e:
            print(traceback.format_exc())
            raise e

    @staticmethod
    def _calc_angle(x_j, y_j, x_i, y_i, x_k, y_k):
        """
        Calculate the angle between the line segments connecting P_j with pixels P_i and P_j
        :param x_j: x coordinate of P_j
        :param y_j: y coordinate of P_j
        :param x_i: x coordinate of P_i
        :param y_i: y coordinate of P_i
        :param x_k: x coordinate of P_k
        :param y_k: y coordinate of P_k
        :return: angle between the line segments connecting P_j with pixels P_i and P_k
        """
        v1 = np.array([x_i - x_j, y_i - y_j])
        v2 = np.array([x_k - x_j, y_k - y_j])
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _detect_corner_indices(self,
                               skel_coor_x: NDArray,
                               skel_coor_y: NDArray,
                               threshold_angle: Union[int, float],
                               k: int) -> NDArray[np.int_]:
        """
        Detects corners indices from the skeletonized line.
        For each pixel P_i in the skeletonized line, the following steps are performed:
        1. Calculate the angle between the line segments connecting P_i with pixels P_(i-k) and P_(i+k)
        2. If the angle is less than or equal to the threshold, mark pixel P_i as a corner

        :param skel_coor_x: x coordinates of skeletonized line
        :param skel_coor_y: y coordinates of skeletonized line
        :param threshold_angle: threshold angle for corner detection
        :param k: the number of pixels to be used for angle calculation
        :return: coorditaes of corner indices
        """

        if len(skel_coor_x) != len(skel_coor_y):
            raise ValueError('The length of skel_coor_x and skel_coor_y must be the same.')
        if not isinstance(k, int):
            raise TypeError('k must be int.')
        if k < 1:
            raise ValueError('k must be larger than 0.')
        if len(skel_coor_x) <= 2 * k :
            print(f'The length of skeletonized line is too short for the argument k={k}.')
            return np.array([])

        corner_indices = []
        for i in range(k, len(skel_coor_x) - k):
            angle = self._calc_angle(skel_coor_x[i], skel_coor_y[i],
                                     skel_coor_x[i - k], skel_coor_y[i - k],
                                     skel_coor_x[i + k], skel_coor_y[i + k])
            if angle <= threshold_angle:
                corner_indices.append(i)
        return np.array(corner_indices)


    @staticmethod
    def _calc_line_to_points_dist(a: Tuple[Union[int, float]],
                                  b: Tuple[Union[int, float]],
                                  x: Union[int, float, Tuple, List, NDArray],
                                  y: Union[int, float, Tuple, List, NDArray]) -> Union[
        NDArray[np.int_], NDArray[np.float_]]:
        """
        Calculate the distance from a point c to a line defined by two points a and b.
        :param a: the coordinate of an endpoint a.
        :param b: the coordinate of the other endpoint b.
        :param x: the x coordinate of points to be calculated.
        :param y: the y coordinate of points to be calculated.
        :return: ndarray of the distance from the line to the points
        """

        def _calc_distance(a: NDArray, b:NDArray, x:Union[int, float], y:Union[int, float]) -> Union[int, float]:
            """
            Calculate the distance from a point c to a line defined by two points a and b.
            :param a: the coordinate of an endpoint a.
            :param b: the coordinate of the other endpoint b.
            :param x: the x coordinate of a point to be calculated.
            :param y: the y coordinate of a point to be calculated.
            :return: the distance from the line to the point
            """
            AB = b - a
            AC = np.array([x, y]) - a
            area = np.linalg.norm(np.cross(AB, AC))
            length_AB = np.linalg.norm(AB)
            return area / length_AB

        A = np.array(a)
        B = np.array(b)

        if not isinstance(x, Iterable) and not isinstance(y, Iterable):
            return np.array([_calc_distance(A, B, x, y)])

        if isinstance(x, Iterable) and isinstance(y, Iterable) and len(x) != len(y):
            raise ValueError('The length of x and y must be the same.')

        else:
            distances = []
            for _x, _y in zip(x, y):
                distances.append(_calc_distance(A, B, _x, _y))

            return np.array(distances)

    def _get_farthest_point_distances_and_indices(self,
                                                  a: Tuple[Union[int, float]],
                                                  b: Tuple[Union[int, float]],
                                                  x: Union[int, float, Tuple, List, NDArray],
                                                  y: Union[int, float, Tuple, List, NDArray]) -> Tuple:
        """
        get the distances and indices of the farthest points from a line defined by two points a and b.
        :param a: the coordinate of an endpoint a.
        :param b: the coordinate of the other endpoint b.
        :param x: the x coordinates of points to be calculated.
        :param y: the y coordinates of points to be calculated.
        :return: the indices of the farthest points from the line.
        """
        distance = self._calc_line_to_points_dist(a, b, x, y)
        if distance.size == 0:
            print('empty array was returned from _calc_line_to_points_dist function.')
            return None, None

        farthest_distance = np.max(distance)
        return farthest_distance, distance.argmax()


    def _binary_decompose_simple(self, skel_coor_x, skel_coor_y, threshold_distance) -> NDArray:
        """
        binary decomposition method to vectorize the skeletonized line
        :param skel_coor_x: x coordinates of skeletonized line
        :param skel_coor_y: y coordinates of skeletonized line
        :param threshold: threshold distance for binary decomposition
        :return: coorditaes of decomposed indices
        """
        decomposed_indices = [0, len(skel_coor_x) - 1]
        updated = True
        while updated:
            updated = False
            for n, (i, j) in enumerate(zip(decomposed_indices[:-1], decomposed_indices[1:])):
                farthest_distance, farthest_indices = self._get_farthest_point_distances_and_indices(
                    (skel_coor_x[i], skel_coor_y[i]),
                    (skel_coor_x[j], skel_coor_y[j]),
                    skel_coor_x[i + 1:j],
                    skel_coor_y[i + 1:j]
                )

                if farthest_distance == 0: # if the line is straight, do not decompose.
                    continue

                elif farthest_distance >= threshold_distance:
                    added_indices = [farthest_indices + i + 1]
                    decomposed_indices = decomposed_indices[:n+1] + added_indices + decomposed_indices[n+1:]
                    updated = True
                    break

        decomposed_indices.sort()
        return np.array(decomposed_indices)


    def _detect_kink_from_decomposed_indices(self,
                                             skel_coor_x,
                                             skel_coor_y,
                                             decomposed_indices,
                                             threshold_angle) -> Tuple[NDArray, NDArray]:
        """
        Detects kinks indices and its angle from the skeletonized line based the decomposed indices.
        :param skel_coor_x: x coordinates of skeletonized line
        :param skel_coor_y: y coordinates of skeletonized line
        :param decomposed_indices: indices of decomposed line
        :param threshold_angle: threshold angle for kink detection. Kink angle lower than or equal to this value is detected.
        :return: ndarray of the kink indices and ndarray of the kink angles
        """
        kink_indices = []
        kink_angles = []
        if len(decomposed_indices) <= 2:
            return np.array(kink_indices), np.array(kink_angles)

        for i in range(1, len(decomposed_indices) - 1):
            angle = self._calc_angle(
                skel_coor_x[decomposed_indices[i]], skel_coor_y[decomposed_indices[i]],
                skel_coor_x[decomposed_indices[i - 1]], skel_coor_y[decomposed_indices[i - 1]],
                skel_coor_x[decomposed_indices[i + 1]], skel_coor_y[decomposed_indices[i + 1]]
            )
            if angle <= threshold_angle:
                kink_indices.append(decomposed_indices[i])
                kink_angles.append(angle)
        return np.array(kink_indices), np.array(kink_angles)


    def _calc_angle_from_decomposed_indices(self,
                                             skel_coor_x,
                                             skel_coor_y,
                                             decomposed_indices) -> NDArray:
        """
        Calculates the angle of a zigzag line consisting of points on the thin line specified by the index.
        :param skel_coor_x: x coordinates of skeleton line
        :param skel_coor_y: y coordinates of skeleton line
        :param decomposed_indices: indices of decomposed line
        :param threshold_angle: threshold angle for kink detection. Kink angle lower than or equal to this value is detected.
        :return: ndarray of the angles between the line segments.
        """
        # todo: _detect_kink_from_decomposed_indices()とロジックのほとんどが重複しているので、統合する。
        angles = []
        if len(decomposed_indices) <= 2:
            return np.array(angles)

        for i in range(1, len(decomposed_indices) - 1):
            angle = self._calc_angle(
                skel_coor_x[decomposed_indices[i]], skel_coor_y[decomposed_indices[i]],
                skel_coor_x[decomposed_indices[i - 1]], skel_coor_y[decomposed_indices[i - 1]],
                skel_coor_x[decomposed_indices[i + 1]], skel_coor_y[decomposed_indices[i + 1]]
            )
            angles.append(angle)
        return np.array(angles)


