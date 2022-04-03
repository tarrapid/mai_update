from typing import Optional

import numpy as np
from src.model import Model


class AbstractControl:
    def __init__(self, model: Model):
        self.m = model

    def gamma(self, k: int, x: np.ndarray) -> Optional[float]:
        """
        :param k:
        :param x:
        :return:
        """
        raise NotImplementedError
