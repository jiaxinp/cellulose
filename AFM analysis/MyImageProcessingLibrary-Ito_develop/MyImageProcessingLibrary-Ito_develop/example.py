import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from MyLibrary import imptools
from MyLibrary.BG_Calibrator import BG_Calibrator
from DentAnalyzer import DentAnalyzer
from MyLibrary.ProcessedImageClass import ProcessedImage
from MyLibrary.KinkDetecter import KinkDetecter
from MyLibrary.Segmentater import Segmentater
from MyLibrary.Skeletonizer import Skeletonizer

# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == "__main__":
    calibrater = BG_Calibrator()
    segmentater = Segmentater()
    skeltonizer = Skeletonizer()
    kinkdetecter = KinkDetecter()  # kink detection method in Fiber class is better
    dentanalyzer = DentAnalyzer()
    path = Path(r"C:\Users\Jess\Dropbox\UTokyo\Research\Cellulose\code\AFM analysis\MyImageProcessingLibrary-Ito_develop\MyImageProcessingLibrary-Ito_develop\data")
    data_dict = {n: file.stem for n, file in enumerate(path.iterdir()) if file.is_dir()}
    pprint.pprint(data_dict)
    data_key = int(input("choose the data to analyse: "))
    filepath = path /data_dict[data_key]
    print(filepath)
    flist = sorted(list(filepath.glob("*.txt")))

    print(flist)

    result_dir = Path("./result")
    if not result_dir.exists():
        result_dir.mkdir()

    # 処理開始
    processed_images_list = []

    for n, f in enumerate(flist):
        print(f"\nループカウントは{n}です")
        print(f"{f.stem} is processed now...")
        AFM_data = np.loadtxt(f, skiprows=1)
        image_size = int(
            np.sqrt(len(AFM_data))
        )  # the number of pixels that a side of AFM image has
        height_data = AFM_data.reshape((image_size, image_size))

        image = ProcessedImage(height_data, name=f.stem)
        calibrater(image)
        segmentater(image)
        skeltonizer(image)
        kinkdetecter(image)  # kink_detection method in Fiber is better
        dentanalyzer(image)  # 画像処理の結果はimageの属性として保存されているので、適宜参照
        # 必ずこの順番で画像処理しないと動かないので注意。

        processed_images_list.append(image)

    height_all_pixels = imptools.all_pixel_height(processed_images_list)
    length_distribution = imptools.length_distribution(processed_images_list)

    # dentの各カテゴリ,normal, bumpの構成率を格納した辞書
    dent_breakdown_dict = dentanalyzer.calc_breakdown(processed_images_list)
    dent_dl = dentanalyzer.calc_dl(processed_images_list)

    fig, ax = plt.subplots(1, 1)
    for category, (depth, length) in dent_dl.items():
        ax.scatter(length, depth)
    plt.show()

    print("finish!")
