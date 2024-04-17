from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fiber:
    fiber_image: np.array
    data: tuple
    xtrack: np.array
    ytrack: np.array
    horizon: np.array
    height: np.array
    kink_indices: np.array
    ep_indices : np.array
    kink_angles: np.array
    decomposed_point_indices: np.array

    @property
    def length(self):
        return self.horizon[-1]

