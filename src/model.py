import numpy as np

from src.utils.condition import Condition


class Model:
    def __init__(
            self,
            mtx_A: np.array, b: float, c: float,
            n: int,
            N: int,
            vec_lambda: np.array,
            fi: float,
            m_ksi: float,
            eps: float
    ):
        self.mtx_A, self.b, self.c = mtx_A, b, c

        self.n = n
        self.N = N

        self.vec_lambda = vec_lambda
        self.fi = fi

        self.eps, self.m_ksi = eps, m_ksi

        self.vec_B = np.array([0] * (n - 1) + [b]).reshape(-1, 1)
        self.vec_C = np.array([0] * (n - 1) + [c]).reshape(-1, 1)
        self.mtx_Lambda_init = np.diag(vec_lambda)

        # TODO: add assert
        # TODO: add DocString
        # TODO: add Dymmy test

    # System & criterion
    def f(self, x: np.array, u: float, ksi: float):
        return self.mtx_A @ x + self.vec_B * u + self.vec_C * ksi

    def FF(self, x: np.array):
        return np.max(np.abs(self.vec_lambda * x))

    def criterion_retention(self, traj_x: list):
        return np.sum(list(map(lambda x: self.FF(x) <= self.fi, traj_x))) / len(traj_x)

    def criterion_math_expectation(self, traj_x: list):
        return np.mean(list(map(lambda x: np.mean(x*x), traj_x)))

    # Additional sets
    def cond_F(self):
        return Condition(self.mtx_Lambda_init, np.full(self.n, self.fi))

    def cond_F_hatch(self):
        return Condition((self.mtx_Lambda_init @ self.mtx_A)[:self.n - 1, :], np.full(self.n - 1, self.fi))
