import unittest

import numpy as np
from MyLibrary.KinkDetecter import KinkDetecter


class TestKinkDetector(unittest.TestCase):
    def setUp(self):
        self.detector = KinkDetecter()

    @staticmethod
    def create_testcase_for_corner_detection(angle):
        # index 10 is the corner
        line1_x = np.arange(1, 0, -0.1)
        line1_y = np.zeros_like(line1_x)
        line2_x = np.linspace(0, np.cos(angle), 11)
        line2_y = np.linspace(0, np.sin(angle), 11)
        testcase_coor_x = np.concatenate([line1_x, line2_x])
        testcase_coor_y = np.concatenate([line1_y, line2_y])
        return testcase_coor_x, testcase_coor_y

    def test_calc_line_to_points_dist_on_line(self):
        A = (0, 0)
        B = (2, 0)
        point_x = 4
        point_y = 0
        expected = np.array([0])
        answer = self.detector._calc_line_to_points_dist(A, B, point_x, point_y)
        self.assertTrue(np.array_equal(answer, expected), f'Expected {expected}, but got {answer}')

    def test_calc_line_to_points_dist_off_line(self):
        A = (0, 0)
        B = (0, 2)
        point_x = 2.0
        point_y = 3.0
        expected = np.array([2.0])
        answer = self.detector._calc_line_to_points_dist(A, B, point_x, point_y)
        self.assertTrue(np.array_equal(answer, expected), f'Expected {expected}, but got {answer}')

    def test_calc_line_to_points_dist_empty_array(self):
        A = (0, 0)
        B = (0, 2)
        point_x = np.array([])
        point_y = np.array([])
        expected = np.array([])
        answer = self.detector._calc_line_to_points_dist(A, B, point_x, point_y)
        self.assertTrue(np.array_equal(answer, expected), f'Expected {expected}, but got {answer}')

    def test_calc_line_to_points_dist_multiple_points(self):
        A = (0, 0)
        B = (0, 2)
        points_x = [2.0, 2.0, 0.0]
        points_y = [3.0, 0.0, 1.0]
        expected = np.array([2.0, 2.0, 0.0])
        self.assertTrue(np.array_equal(self.detector._calc_line_to_points_dist(A, B, x=points_x, y=points_y), expected))

    def test_calc_line_to_points_dist_different_length_xy(self):
        A = (0, 0)
        B = (0, 2)
        points_x = [2.0, 2.0, 0.0]
        points_y = [3.0, 0.0]

        with self.assertRaises(ValueError):
            self.detector._calc_line_to_points_dist(A, B, x=points_x, y=points_y)

    def test_calc_line_to_points_dist_ndarray_input(self):
        A = (0, 0)
        B = (0, 2)
        points_x = np.array([2.0, 2.0, 0.0])
        points_y = np.array([3.0, 0.0, 1.0])
        expected = np.array([2.0, 2.0, 0.0])
        self.assertTrue(np.array_equal(self.detector._calc_line_to_points_dist(A, B, x=points_x, y=points_y), expected))

    def test_get_farthest_point_distance_and_indices_single_answer(self):
        A = (0, 0)
        B = (2, 0)
        points_x = np.arange(0, 10, 1)
        points_y = np.arange(0, 10, 1)
        expected = (9, 9)
        answer = self.detector._get_farthest_point_distances_and_indices(A, B, points_x, points_y)
        self.assertEqual(answer[0], expected[0], f'Expected {expected[0]}, but got {answer[0]}')
        self.assertTrue(np.array_equal(answer[1], expected[1]), f"Expected {expected[1]}, but got {answer[1]}")

    def test_get_farthest_point_distance_and_indices_multiple_answers(self):
        A = (0, 0)
        B = (2, 0)
        points_x = np.array([0, 1, 2])
        points_y = np.array([1, 1, 1])
        expected = (1, 0)
        answer = self.detector._get_farthest_point_distances_and_indices(A, B, points_x, points_y)
        self.assertEqual(answer[0], expected[0], f'Expected {expected[0]}, but got {answer[0]}')
        self.assertTrue(np.array_equal(answer[1], expected[1]), f"Expected {expected[1]}, but got {answer[1]}")

    def test_get_farthest_point_distance_and_indices_single_input(self):
        A = (0, 0)
        B = (2, 0)
        points_x = 0
        points_y = 0
        expected = (0, 0)
        answer = self.detector._get_farthest_point_distances_and_indices(A, B, points_x, points_y)
        self.assertEqual(answer[0], expected[0], f'Expected {expected[0]}, but got {answer[0]}')
        self.assertTrue(np.array_equal(answer[1], expected[1]), f"Expected {expected[1]}, but got {answer[1]}")

    def test_get_farthest_point_distance_and_indices_empty_input(self):
        A = (0, 0)
        B = (2, 0)
        points_x = np.array([])
        points_y = np.array([])
        expected = (None, None)
        answer = self.detector._get_farthest_point_distances_and_indices(A, B, points_x, points_y)
        self.assertEqual(answer[0], expected[0], f'Expected {expected[0]}, but got {answer[0]}')
        self.assertTrue(np.array_equal(answer[1], expected[1]), f"Expected {expected[1]}, but got {answer[1]}")

    def test_detect_corner_indices_too_short_skeleton(self):
        angle = np.pi/6
        skel_coor_x, skel_coor_y = self.create_testcase_for_corner_detection(angle=angle)
        threshold_angle = np.pi/4
        k1 = 50
        k2 = 11
        expected = np.array([])
        answer1 = self.detector._detect_corner_indices(skel_coor_x, skel_coor_y, threshold_angle, k1)
        answer2 = self.detector._detect_corner_indices(skel_coor_x, skel_coor_y, threshold_angle, k2)
        self.assertTrue(np.array_equal(answer1, expected), f'Expected {expected}, but got {answer1}')
        self.assertTrue(np.array_equal(answer2, expected), f'Expected {expected}, but got {answer2}')


    def test_detect_corner_indices_no_corner(self):
        angle = np.pi/4
        skel_coor_x, skel_coor_y = self.create_testcase_for_corner_detection(angle=angle)
        threshold_angle = np.pi/6
        k = 5
        expected = np.array([])
        answer = self.detector._detect_corner_indices(skel_coor_x, skel_coor_y, threshold_angle, k)
        self.assertTrue(np.array_equal(answer, expected), f'Expected {expected}, but got {answer}')


    def test_detect_corner_indices_angle_equal_threshold(self):
        angle = np.pi/3
        skel_coor_x, skel_coor_y = self.create_testcase_for_corner_detection(angle=angle)
        threshold_angle = np.pi/3
        k = 5
        expected = np.array([10])
        answer = self.detector._detect_corner_indices(skel_coor_x, skel_coor_y, threshold_angle, k)
        self.assertTrue(np.array_equal(answer, expected), f'Expected {expected}, but got {answer}')


    def test_detect_corner_indices_angle_lower_than_threshold(self):
        angle = np.pi/4
        skel_coor_x, skel_coor_y = self.create_testcase_for_corner_detection(angle=angle)
        threshold_angle = np.pi/3
        k = 5
        expected = np.array([9, 10, 11]) # 10 is the true corner but 9 and 11 are also detected as corners
        answer = self.detector._detect_corner_indices(skel_coor_x, skel_coor_y, threshold_angle, k)
        self.assertTrue(np.array_equal(answer, expected), f'Expected {expected}, but got {answer}')


    def test_binary_decompose_simple(self):
        test_skel_x = np.arange(0, 17)
        test_skel_y = np.hstack([np.arange(6), np.arange(4, -4, -1), np.arange(-2, 1)])
        threshold_distance = 3

        test_skel_x2 = np.hstack([np.zeros(6), np.arange(1, 6), np.repeat(5, 8), np.arange(6, 11), np.repeat(10, 3)])
        test_skel_y2 = np.hstack([np.arange(6), np.repeat(5, 5), np.arange(4, -4, -1), np.repeat(-3, 5), np.arange(-2, 1, 1)])
        threshold_distance2 = 2

        expected = np.array([0, 5, 13, 16])
        expected2 = np.array([0, 5, 10, 18, 23, 26])
        answer = self.detector._binary_decompose_simple(test_skel_x, test_skel_y, threshold_distance)
        self.assertTrue(np.array_equal(answer, expected), f'Expected {expected}, but got {answer}')
        answer2 = self.detector._binary_decompose_simple(test_skel_x2, test_skel_y2, threshold_distance2)
        self.assertTrue(np.array_equal(answer2, expected2), f'Expected {expected2}, but got {answer2}')

    def test_detect_kink_from_decomposed_indices(self):
        test_skel_x = np.arange(0, 17)
        test_skel_y = np.hstack([np.arange(6), np.arange(4, -4, -1), np.arange(-2, 1)])
        threshold_distance = 3
        decomposed_indices = self.detector._binary_decompose_simple(test_skel_x, test_skel_y, threshold_distance)

        angle_threshold = 5* np.pi / 6
        expected = (np.array([5, 13]), np.array([np.pi/2, np.pi/2]))
        answer = self.detector._detect_kink_from_decomposed_indices(test_skel_x, test_skel_y, decomposed_indices, angle_threshold)
        self.assertTrue(np.array_equal(answer[0], expected[0]), f'Expected {expected[0]}, but got {answer[0]}')
        self.assertTrue(np.array_equal(answer[1], expected[1]), f'Expected {expected[1]}, but got {answer[1]}')


if __name__ == '__main__':
    unittest.main()
