from typing import Optional

import numpy as np

from src.utils import sgn, cache
from src.condition import Condition, intersection


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

        self.eps_wave = 2 * abs(c * eps / b)

        # TODO: add assert
        # TODO: add DocString
        # TODO: add Dymmy test

    # System & criterion
    def f(self, x: np.array, u: float, ksi: float):
        return self.mtx_A @ x + self.vec_B * u + self.vec_C * ksi

    def FF(self, x: np.array):
        return np.max(np.abs(self.vec_lambda * x))

    def criterion(self, traj_x: list):
        return all(list(map(lambda x: self.FF(x) <= self.fi, traj_x)))

    # Additional sets
    def cond_F(self):
        return Condition(self.mtx_Lambda_init, np.full(self.n, self.fi))

    def cond_F_hatch(self):
        return Condition((self.mtx_Lambda_init @ self.mtx_A)[:self.n - 1, :], np.full(self.n - 1, self.fi))

    # Step N
    def cond_I_N(self):
        if np.abs(self.vec_lambda[-1] * self.c * self.eps) <= self.fi:
            conds = [self.cond_F(), self.cond_F_hatch()]
            return intersection(conds)
        else:
            return None

    def cond_O_N(self):
        pass

    def gamma_N(self, x: np.array):
        k1 = self.vec_lambda[-1] * self.b
        k2 = (self.mtx_Lambda_init @ self.mtx_A @ x)[-1]
        k3 = self.vec_lambda[-1] * self.c * self.m_ksi
        return -(k2 + k3)/k1

    # Step N-1
    @cache
    def cond_I_N_1(self):
        if np.abs(self.vec_lambda[-1] * self.c * self.eps) <= self.fi:
            return intersection([
                self.cond_F(),
                self.cond_F_hatch(),
                self.cond_delta_I(self.N - 1)
            ])
        else:
            return None

    @cache
    def cond_delta_I(self, k: int):
        m_Lambda = self.mtx_Lambda(k)
        return Condition(m_Lambda, np.full(m_Lambda.shape[0], self.fi))

    # Suboptimal control k=0,...,N-2
    @cache
    def nn(self, k):
        if k == self.N - 2:
            return int(0.5 * self.n * (self.n - 1))
        elif k in range(0, (self.N-3)+1):
            return int(0.5 * (self.nn(k + 1) + self.n) * (self.nn(k + 1) + self.n - 1))
        else:
            return None

    #     def pp(self, i, j, k):
    #         if i < j:
    #             return (i-1)*(self.nn(k+1) + n) + j - 1
    #         else:
    #             return None
    @cache
    def ep_Lambda(self, i, j, k):
        bb_val_i = self.bb(i, k)
        bb_val_j = self.bb(j, k)

        # if bb_val_i == 0 or bb_val_j == 0:
        #     z1 = self.fi / (self.eps_wave + 2* self.fi)
        #     z2 = self.aa(i, k).T - self.aa(j, k).T
        # else:
        if bb_val_i != 0 and bb_val_j != 0:
            z1 = self.fi / (self.eps_wave + (sgn(bb_val_i) + sgn(bb_val_j)) * self.fi)
            z2 = self.aa(i, k).T / bb_val_i - self.aa(j, k).T / bb_val_j
            return (z1 * z2).reshape(1, -1)
        else:
            return np.zeros_like(self.aa(i, k).T).reshape(1, -1)

    @cache
    def bb(self, i, k) -> Optional[int]:
        if k in range(0, (self.N-2)+1):
            return (self.mtx_Lambda_wave(k + 1) @ self.vec_B)[i - 1][0]
        elif k == self.N-1:
            return (self.mtx_Lambda(k + 1) @ self.vec_B)[i - 1][0]
        else:
            return None

    @cache
    def aa(self, i, k):
        if k in range(0, (self.N - 2) + 1):
            return (self.mtx_Lambda_wave(k + 1) @ self.mtx_A)[i - 1]
        elif k == self.N - 1:
            return (self.mtx_Lambda(k + 1) @ self.mtx_A)[i - 1]
        else:
            return None

    @cache
    def mtx_Lambda(self, k):
        if k == self.N:
            concat = [
                (self.mtx_Lambda_init @ self.mtx_A)[:-1, :],
                self.mtx_Lambda_init[-1].reshape(1, -1)
            ]
            return np.concatenate(concat, axis=0)

        elif k in range(0, (self.N - 1) + 1):
            if k == self.N-1:
                cc = self.n
            else:
                cc = self.nn(k) + self.n
            res = []
            for i in range(1, cc + 1):
                for j in range(1, cc + 1):
                    if i < j:
                        val = self.ep_Lambda(i, j, k)
                        if val is not None:
                            res.append(val)
            return np.concatenate(res, axis=0)
        else:
            return None

    @cache
    def mtx_Lambda_wave(self, k):
        if k in range(0, (self.N - 1) + 1):
            concat = [
                self.mtx_Lambda(k),
                (self.mtx_Lambda_init @ self.mtx_A)[:-1, :],
                self.mtx_Lambda_init[-1].reshape(1, -1)
            ]
            return np.concatenate(concat, axis=0)
        else:
            return None

    @cache
    def non_zero_I_N_2(self, k: int):
        if k in range(0, (self.N - 2) + 1):
            if k == self.N - 2:
                cc = self.n
            else:
                cc = self.nn(k + 1) + self.n

            return all([abs(self.bb(i, k)) <= 2 * self.fi / self.eps_wave
                        for i in range(1, cc + 1)])
        else:
            return None

    @cache
    def cond_I_other(self, k: int):
        if self.non_zero_I_N_2(k):
            return intersection([
                self.cond_F(),
                self.cond_F_hatch(),
                self.cond_delta_I(k)
            ])
        else:
            return None

    def gamma_other(self, k:int, x: np.array):
        return self.cc(k, x) - (self.c * self.m_ksi) / self.b

    def cc(self, k, x):
        return 1./2 * (self.ffi_up(k, x) + self.ffi_down(k, x))

    def ffi_up(self, k, x):
        cc = self.n if k == self.N-1 else self.nn(k) + self.n
        res = []
        for i in range(1, cc + 1):
            bb_val = self.bb(i, k)
            if bb_val != 0:
                val = (sgn(bb_val)*self.fi - self.aa(i, k).T @ x) / bb_val
                res.append(val)
        return min(res)

    def ffi_down(self, k, x):
        cc = self.n if k == self.N-1 else self.nn(k) + self.n
        res = []
        for i in range(1, cc + 1):
            bb_val = self.bb(i, k)
            if bb_val != 0:
                val = (-sgn(bb_val)*self.fi - self.aa(i, k).T @ x) / bb_val
                res.append(val)
        return max(res)

    # All result
    @cache
    def cond_I(self, k: int):
        if k == self.N + 1:
            return self.cond_F()
        elif k == self.N:
            return self.cond_I_N()
        elif k == self.N - 1:
            return self.cond_I_N_1()
        else:
            return self.cond_I_other(k)

    def gamma(self, k, x) -> Optional[int]:
        if k == self.N:
            return self.gamma_N(x)
        elif k in range(0, (self.N-1) + 1):
            return self.gamma_other(k, x)
        else:
            return None
