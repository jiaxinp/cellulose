#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.insert(0, '/Users/tomok/PycharmProjects/MyImageProcessingLibrary/MyLibrary')

import time
from functools import wraps

import cv2
import mahotas as mh
import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize, thin
from typing import Union
import math
import matplotlib.pyplot as plt


def branchedPoints(skel: NDArray[np.uint8]) -> NDArray[np.uint8]:
    # todo vh_xbranchのパターンを消去してもfiberは分離されないのでは？
    vh_xbranch = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])

    diagonal_xbranch = np.array([[1, 0, 1],
                                 [0, 1, 0],
                                 [1, 0, 1]])

    vh_ybranch = np.array([[1, 0, 1],
                           [0, 1, 0],
                           [2, 1, 2]])

    # todo diagonal_ybranchは多くのパターンをカバーしすぎでは?
    diagonal_ybranch = np.array([[0, 1, 2],
                                 [1, 1, 2],
                                 [2, 2, 1]])

    vh_tbranch = np.array([[0, 0, 0],
                           [1, 1, 1],
                           [0, 1, 0]])

    diagonal_tbranch = np.array([[1, 0, 1],
                                 [0, 1, 0],
                                 [1, 0, 0]])

    branch_patterns = []
    for rot_time in range(4):
        for branch_pattern in [vh_ybranch, diagonal_ybranch, vh_tbranch, diagonal_tbranch]:
            branch_patterns.append(np.rot90(branch_pattern, k=rot_time))
    branch_patterns.append(vh_xbranch)
    branch_patterns.append(diagonal_xbranch)

    padded_skel = np.pad(skel, pad_width=1, mode='constant', constant_values=0)
    hits = np.zeros_like(padded_skel, dtype=np.uint8)
    for branch_pattern in branch_patterns:
        hits += mh.morph.hitmiss(padded_skel, branch_pattern)
    # Pixels with multiple pattern hits are also corrected to 1.
    hits = np.where(hits > 0, 1, 0).astype(np.uint8)
    return hits[1:-1, 1:-1].copy()


def endPoints(skel: NDArray[np.uint8]) -> NDArray[np.uint8]:
    endpoint1 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [2, 1, 2]])

    endpoint2 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
    end_patterns = []
    for rot_time in range(4):
        for end_pattern in [endpoint1, endpoint2]:
            end_patterns.append(np.rot90(end_pattern, k=rot_time))

    endpoint_single = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])
    end_patterns.append(endpoint_single)

    padded_skel = np.pad(skel, pad_width=1, mode='constant', constant_values=0)
    hits = np.zeros_like(padded_skel, dtype=np.uint8)
    for end_pattern in end_patterns:
        hits += mh.morph.hitmiss(padded_skel, end_pattern).astype(np.uint8)
        # Pixels with multiple pattern hits are also corrected to 1.
    hits = np.where(hits > 0, 1, 0).astype(np.uint8)

    return hits[1:-1, 1:-1].copy()


def remove_bp(img, remove_size=1, min_area=10):  # 10/21 min_areaを追加
    imgcopy = img.copy()
    bp = branchedPoints(imgcopy)
    bp_coor = np.where(bp)
    for bp_x, bp_y in zip(bp_coor[0], bp_coor[1]):  # bpの周囲を除去
        imgcopy[
        bp_x - remove_size: bp_x + remove_size + 1,
        bp_y - remove_size: bp_y + remove_size + 1,
        ] = 0

    if min_area != 0:  # 10/21追加
        tmp_nlabels, tmp_label_image = cv2.connectedComponents(np.uint8(imgcopy))
        for i in range(1, tmp_nlabels):
            size = np.sum(tmp_label_image == i)
            if size < min_area:
                imgcopy[np.where(tmp_label_image == i)] = 0
    return imgcopy
    # return imgcopy, tmp_nlabels


def remove_Lcorner(skeleton_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    remove L corner from skeleton_image.
    """
    # 1. pad 0 around the image
    imgcopy = skeleton_image.copy()
    imgcopy = np.pad(imgcopy, pad_width=1, mode='constant', constant_values=0)

    # 2. Define L corners patterns to be removed.
    corner = np.array([[0, 1, 0],
                       [1, 1, 0],
                       [0, 0, 0]])
    corner2 = np.array([[0, 1, 0],
                        [0, 1, 1],
                        [0, 0, 0]])
    corner3 = np.array([[0, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0]])
    corner4 = np.array([[0, 0, 0],
                        [0, 1, 1],
                        [0, 1, 0]])

    # 3. Detect L corners
    hits = np.zeros_like(imgcopy, dtype=np.uint8)
    for corner_pattern in [corner, corner2, corner3, corner4]:
        hits += mh.morph.hitmiss(imgcopy, corner_pattern)

    # 4. Remove the detected L corners from the original image.
    Lremoved_img = imgcopy - hits
    return Lremoved_img[1:-1, 1:-1].copy() # remove padding

def remove_square_branches(skeleton_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    imgcopy = skeleton_image.copy()
    imgcopy = np.pad(imgcopy, pad_width=1, mode='constant', constant_values=0)
    square_pattern1 = np.array([[1, 0, 0],
                                [0, 1, 1],
                                [0, 1, 1]])
    square_pattern2 = np.array([[0, 0, 1],
                                [1, 1, 0],
                                [1, 1, 0]])
    square_pattern3 = np.array([[1, 1, 0],
                                [1, 1, 0],
                                [0, 0, 1]])
    square_pattern4 = np.array([[0, 1, 1],
                                [0, 1, 1],
                                [1, 0, 0]])

    hits = np.zeros_like(imgcopy, dtype=np.uint8)
    for square in [square_pattern1, square_pattern2, square_pattern3, square_pattern4]:
        hits += mh.morph.hitmiss(imgcopy, square)
    square_removed_img = imgcopy - hits
    return square_removed_img[1:-1, 1:-1].copy() # remove padding




def tracking(skeleton_image :NDArray[np.uint8]) -> tuple[NDArray, NDArray]:
    """
    calculate the coordination of line in skeleton_image.

    :param skeleton_image: ndarray(dtype = np.uint8) represents skeleton image
    :return: (ndarray, ndarray)
    """
    try:
        if np.array_equal(skeleton_image, np.array([[1]])): # 1ピクセルの場合
            return (np.array([0]), np.array([0]))

        imgcopy = skeleton_image.copy()
        imgcopy = np.pad(imgcopy, pad_width=1, mode='constant', constant_values=0)

        ep = endPoints(imgcopy)
        (ep_y_start, ep_y_end),(ep_x_start, ep_x_end) = np.where(ep)

        ytrack = [ep_y_start]
        xtrack = [ep_x_start]

        y = ep_y_start
        x = ep_x_start
        for i in range(np.sum(imgcopy)): # todo これ空の配列だったらどうなるんだ？
            imgcopy[y, x] = 0  # 現在の注目画素の値を0に更新
            # 移動方向の探索
            window = imgcopy[y - 1: y + 2, x - 1: x + 2]
            direction_y, direction_x = np.where(window != 0)
            direction_y = direction_y.item() - 1
            direction_x = direction_x.item() - 1
            y += direction_y
            x += direction_x
            xtrack.append(x)
            ytrack.append(y)
            if x == ep_x_end and y == ep_y_end:
                break
        xtrack = np.asarray(xtrack) - 1 # subtract 1 to compensate for the padding
        ytrack = np.asarray(ytrack) - 1
        return xtrack, ytrack
    except:
        print("error in tracking")
        nLabels, label_image, data, center = cv2.connectedComponentsWithStats(skeleton_image.astype(np.uint8))
        x, y, w, h, area = data[1]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].imshow(skeleton_image, cmap="gray")
        ax[1].imshow(skeleton_image[y:y+h, x:x+w], cmap="gray")
        ax[0].add_patch(plt.Rectangle((x, y), w, h, edgecolor="r", fill=False))
        ax[0].set_title("original_skeleton_image")
        ax[1].set_title("enlarged area of error")
        plt.show()



def convert_track_to_distance(xtrack: np.ndarray,
                              ytrack:np.ndarray,
                              pixel_step_size: Union[int, float]) -> np.ndarray:
    """
    calculate the horizontal distance of each pixel in the track from the first pixel.
    if the negighbor pixel is located in diagonal direction, the distance to the neighbor pixel is calculated as sqrt(2) * pixel_step_size.
    """
    xmove = xtrack - np.roll(xtrack, 1)
    ymove = ytrack - np.roll(ytrack, 1)
    xmove = np.delete(xmove, 0)
    ymove = np.delete(ymove, 0)  # index0の値は意味がないので消去
    a = xmove != 0
    b = ymove != 0  # x,y座標の変化の有無を表す配列(True:変化した False:変化しなかった)
    c = np.vstack((a, b))
    d = np.all(c, axis=0)  # 進行方向を表す配列(True:斜め False:上下左右)
    horizon = [0]  # 0はスタート地点のx座標
    distance = 0
    for j in d:
        if j == True:
            distance += pixel_step_size * math.sqrt(2)
        else:
            distance += pixel_step_size
        horizon.append(distance)
    horizon = np.asarray(horizon)
    return horizon


def get_length(skel_img, pixel_size=2000 / 1024):
    """
    Do not use this function. Too late
    calculate the length of fibers.
    :param skel_img:
    :param pixel_size:
    :return:
    """
    mask1 = np.eye(2).astype(np.uint8)
    mask2 = np.flip(mask1, axis=0)
    row, column = skel_img.shape
    diag = np.zeros((row - 1, column - 1)).astype(np.uint8)
    for i in range(row - 1):
        for j in range(column - 1):
            if np.allclose(mask1, skel_img[i: i + 2, j: j + 2]) or np.allclose(
                    mask2, skel_img[i: i + 2, j: j + 2]
            ):
                diag[i, j] = 1

    length = (np.sum(skel_img) + np.sum(diag) * (np.sqrt(2) - 1)) * pixel_size
    return length


def get_length2(skel):  # 1ピクセルは長さ0として判定されるはず
    """
    calculate sum of length of CNF in skel.
    :param skel: skelton image to calculate length of CNF.
    :return:
    """

    def match_template_sad(image, template):
        """
        Template maching method. please refer to URL below for details.
        https://qiita.com/aa_debdeb/items/a3905a902263402ab8ea
        """
        shape = (
                    image.shape[0] - template.shape[0] + 1,
                    image.shape[1] - template.shape[1] + 1,
                ) + template.shape
        strided_image = np.lib.stride_tricks.as_strided(image, shape, image.strides * 2)
        return np.sum(np.abs(strided_image - template), axis=(2, 3))

    _skel = np.copy(skel).astype("int64")

    mask1 = np.array([[1, 0], [0, 1]], dtype="int64")
    mask2 = np.array([[0, 1], [1, 0]], dtype="int64")
    mask3 = np.array([[1], [1]], dtype="int64")
    mask4 = np.array([[1, 1]], dtype="int64")

    sad_image1 = match_template_sad(_skel, mask1)
    sad_image2 = match_template_sad(_skel, mask2)
    sad_image3 = match_template_sad(_skel, mask3)
    sad_image4 = match_template_sad(_skel, mask4)

    diag_dist = 2 * np.sqrt(2) * (np.sum(sad_image1 == 0) + np.sum(sad_image2 == 0))
    hv_dist = 2 * (np.sum(sad_image3 == 0) + np.sum(sad_image4 == 0))
    return diag_dist + hv_dist


def all_pixel_height(image_list):
    all_height = []
    for image in image_list:
        height = image.calibrated_image[np.where(image.skeleton_image)]
        all_height.extend(list(height))
    return all_height


def length_distribution(image_list):
    all_length = []
    for image in image_list:
        for i in range(1, image.nLabels):
            y, x, h, w, area = image.data[i]
            length = get_length(image.skeleton_image[x: x + w, y: y + h])
            all_length.append(length)
    return all_length


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__}は{elapsed_time}秒かかりました")
        return result

    return wrapper
