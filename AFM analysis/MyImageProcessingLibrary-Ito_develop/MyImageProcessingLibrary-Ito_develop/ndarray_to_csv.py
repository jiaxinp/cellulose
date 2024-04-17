import pickle
from typing import Iterable

import pandas as pd

from MyLibrary.ProcessedImageClass import Fiber


def main():
    fiber_objs, file_names = load_datapath(PATH)
    fiber = fiber_objs[0]
    df_fiber = pd.DataFrame(data={"horizon": fiber.horizon, "height": fiber.height})
    df_fiber.to_csv("../../Desktop/fiber_data.csv")


def load_datapath(paths: Iterable[str]) -> tuple[list[Fiber], list[str]]:
    fiber_objs = []
    file_names = []
    for path in paths:
        file_names.append(path)
        with open(path, "rb") as f:
            fiber = pickle.load(f)
            fiber_objs.append(fiber)
    return fiber_objs, file_names


if __name__ == "__main__":
    PATH = {"./pickle_data/softholo_5mmol_220524/Fiber_data/softholo_220524.004/63.pickle"}
    main()
