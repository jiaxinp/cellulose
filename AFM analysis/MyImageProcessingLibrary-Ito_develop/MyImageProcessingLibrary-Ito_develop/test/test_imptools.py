import unittest
import numpy as np
import numpy.testing as npt
from MyLibrary.imptools import *


class TestBranchedPoints(unittest.TestCase):
    def test_continuous_branch(self):
        continuous_branch1 = np.array([[0, 0, 1, 0, 0],
                                       [0, 1, 0, 0, 0],
                                       [1, 1, 1, 1, 1],
                                       [0, 0, 1, 0, 0],
                                       [0, 0, 0, 1, 0]], dtype=np.uint8)
        result1 = branchedPoints(continuous_branch1)

        expected1 = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]], dtype=np.uint8)

        continuous_branch2 = np.array([
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0]
        ], dtype=np.uint8)
        result2 = branchedPoints(continuous_branch2)

        expected2 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)

        continuous_branch3 = np.array([
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0]
        ], dtype=np.uint8)
        result3 = branchedPoints(continuous_branch3)

        expected3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)

        npt.assert_array_equal(result1, expected1)
        npt.assert_array_equal(result2, expected2)
        npt.assert_array_equal(result3, expected3)

    def test_circle(self):
        circle = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        result = branchedPoints(circle)

        expected = np.zeros_like(circle, dtype=np.uint8)
        npt.assert_array_equal(result, expected)

    def test_edge_branch(self):
        # todo edge_branch[4, 2]のようになる入力値がありえるのかは微妙なところ
        # todo このテストパターンにtracking関数を適用するとおそらくエラーがでるので、トラッキング失敗ケースを実行プログラムで検出しておきたい
        edge_branch = np.array([
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ], dtype=np.uint8)
        result = branchedPoints(edge_branch)

        expected = np.array([[0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=np.uint8)
        npt.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.uint8)


class TestEndpoints(unittest.TestCase):
    # todo T字の終わり先端のようなepはなかったので、テスト未実装
    def test_ep(self):
        normal_ep = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        result_normal = endPoints(normal_ep)

        expected_normal = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)

        edge_ep = np.array([
            [0, 1, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
        ], dtype=np.uint8)
        result_edge = endPoints(edge_ep)

        expected_edge = np.array([
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
        ], dtype=np.uint8)

        npt.assert_array_equal(result_normal, expected_normal)
        npt.assert_array_equal(result_edge, expected_edge)

    def test_single_pixel(self):
        single_pixel = np.array([[1]])
        result = endPoints(single_pixel)
        expected = np.array([[1]])
        npt.assert_array_equal(result, expected)


class TestTracking(unittest.TestCase):
    # todo branchedPointsのテストをコピペしたので、expectedを修正する。
    def test_vertical_line(self):
        vertical_line = np.zeros((5,5)).astype(np.uint8)
        vertical_line[:, 2] = 1
        result = tracking(vertical_line)
        expected_y = np.array([0, 1, 2, 3, 4])
        expected_x = np.array([2, 2, 2, 2, 2])
        npt.assert_array_equal(result[0], expected_x)
        npt.assert_array_equal(result[1], expected_y)

    def test_single_pixel(self):
        single_pixel = np.array([[1]])
        result = tracking(single_pixel)
        expected = (np.array([0]), np.array([0]))
        npt.assert_array_equal(result, expected)


class TestConvertTrackToDistance(unittest.TestCase):
    def test_vertical_track(self):
        pixel_step_size = 1
        vertical_xtrack = np.zeros(5).astype(np.uint16)
        vertical_ytrack = np.arange(5).astype(np.uint16)
        result = convert_track_to_distance(vertical_xtrack, vertical_ytrack, pixel_step_size=pixel_step_size)
        expected = np.arange(5)
        npt.assert_array_equal(result, expected)

    def test_diagonal_track(self):
        pixel_step_size = 1
        diagonal_xtrack = np.arange(5).astype(np.uint16)
        diagonal_ytrack = np.arange(5).astype(np.uint16)
        result = convert_track_to_distance(diagonal_xtrack, diagonal_ytrack, pixel_step_size=pixel_step_size)
        expected = np.sqrt(2) * np.arange(5)
        npt.assert_array_almost_equal(result, expected, decimal=10)

    def test_combination_track(self):
        pixel_step_size = 1
        xtrack = np.arange(5).astype(np.uint16)
        ytrack = np.array([0, 1, 2, 2, 2]).astype(np.uint16)
        result = convert_track_to_distance(xtrack, ytrack, pixel_step_size=pixel_step_size)
        expected = np.array([0, pixel_step_size*np.sqrt(2), 2*pixel_step_size*np.sqrt(2), 2*pixel_step_size*np.sqrt(2) + pixel_step_size, 2*pixel_step_size*np.sqrt(2) + 2*pixel_step_size])
        npt.assert_array_almost_equal(result, expected, decimal=10)

class TestRemove_square_branches(unittest.TestCase):
    def test_remove_square_branches(self):
        square_corner = np.array([[1, 0, 0, 1],
                                  [0, 1, 1, 0],
                                  [0, 1, 1, 0],
                                  [1, 0, 0, 1]]).astype(np.uint8)

        result = remove_square_branches(square_corner)
        expected = np.array([[1, 0, 0, 1],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [1, 0, 0, 1]]).astype(np.uint8)
        npt.assert_array_equal(result, expected)

class TestRemove_Lcorner(unittest.TestCase):
    def test_remove_Lcorner(self):
        corners = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 0, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]]).astype(np.uint8)

        result = remove_Lcorner(corners)
        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 1, 0, 1, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0]]).astype(np.uint8)
        npt.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
