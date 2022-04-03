from typing import Optional
import numpy as np

from src.utils.general import cache, sgn
from src.utils.condition import intersection, Condition

from .abstract import AbstractControl


class LPCH(AbstractControl):
    """linear-probabilistic-control-hold"""
    def __init__(self, model):
        super().__init__(model)
        self.eps_wave = 2 * abs(self.m.c * self.m.eps / self.m.b)

    # Step N
    def cond_I_N(self):
        if np.abs(self.m.vec_lambda[-1] * self.m.c * self.m.eps) <= self.m.fi:
            conds = [self.m.cond_F(), self.m.cond_F_hatch()]
            return intersection(conds)
        else:
            return None

    def cond_O_N(self):
        pass

    def gamma_N(self, x: np.array):
        k1 = self.m.vec_lambda[-1] * self.m.b
        k2 = (self.m.mtx_Lambda_init @ self.m.mtx_A @ x)[-1]
        k3 = self.m.vec_lambda[-1] * self.m.c * self.m.m_ksi
        return -(k2 + k3)/k1

    # Step N-1
    @cache
    def cond_I_N_1(self):
        if np.abs(self.m.vec_lambda[-1] * self.m.c * self.m.eps) <= self.m.fi:
            return intersection([
                self.m.cond_F(),
                self.m.cond_F_hatch(),
                self.cond_delta_I(self.m.N - 1)
            ])
        else:
            return None

    @cache
    def cond_delta_I(self, k: int):
        m_Lambda = self.mtx_Lambda(k)
        return Condition(m_Lambda, np.full(m_Lambda.shape[0], self.m.fi))

    # Suboptimal control k=0,...,N-2
    @cache
    def nn(self, k):
        if k == self.m.N - 2:
            return int(0.5 * self.m.n * (self.m.n - 1))
        elif k in range(0, (self.m.N-3)+1):
            return int(0.5 * (self.nn(k + 1) + self.m.n) * (self.nn(k + 1) + self.m.n - 1))
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

        if bb_val_i != 0 and bb_val_j != 0:
            z1 = self.m.fi / (self.eps_wave + (sgn(bb_val_i) + sgn(bb_val_j)) * self.m.fi)
            z2 = self.aa(i, k).T / bb_val_i - self.aa(j, k).T / bb_val_j
            return (z1 * z2).reshape(1, -1)
        else:
            return np.zeros_like(self.aa(i, k).T).reshape(1, -1)

    @cache
    def bb(self, i, k) -> Optional[int]:
        if k in range(0, (self.m.N-2)+1):
            return (self.mtx_Lambda_wave(k + 1) @ self.m.vec_B)[i - 1][0]
        elif k == self.m.N-1:
            return (self.mtx_Lambda(k + 1) @ self.m.vec_B)[i - 1][0]
        else:
            return None

    @cache
    def aa(self, i, k):
        if k in range(0, (self.m.N - 2) + 1):
            return (self.mtx_Lambda_wave(k + 1) @ self.m.mtx_A)[i - 1]
        elif k == self.m.N - 1:
            return (self.mtx_Lambda(k + 1) @ self.m.mtx_A)[i - 1]
        else:
            return None

    @cache
    def mtx_Lambda(self, k):
        if k == self.m.N:
            concat = [
                (self.m.mtx_Lambda_init @ self.m.mtx_A)[:-1, :],
                self.m.mtx_Lambda_init[-1].reshape(1, -1)
            ]
            return np.concatenate(concat, axis=0)

        elif k in range(0, (self.m.N - 1) + 1):
            if k == self.m.N-1:
                cc = self.m.n
            else:
                cc = self.nn(k) + self.m.n
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
        if k in range(0, (self.m.N - 1) + 1):
            concat = [
                self.mtx_Lambda(k),
                (self.m.mtx_Lambda_init @ self.m.mtx_A)[:-1, :],
                self.m.mtx_Lambda_init[-1].reshape(1, -1)
            ]
            return np.concatenate(concat, axis=0)
        else:
            return None

    @cache
    def non_zero_I_N_2(self, k: int):
        if k in range(0, (self.m.N - 2) + 1):
            if k == self.m.N - 2:
                cc = self.m.n
            else:
                cc = self.nn(k + 1) + self.m.n

            return all([abs(self.bb(i, k)) <= 2 * self.m.fi / self.eps_wave
                        for i in range(1, cc + 1)])
        else:
            return None

    @cache
    def cond_I_other(self, k: int):
        if self.non_zero_I_N_2(k):
            return intersection([
                self.m.cond_F(),
                self.m.cond_F_hatch(),
                self.cond_delta_I(k)
            ])
        else:
            return None

    def gamma_other(self, k:int, x: np.array):
        return self.cc(k, x) - (self.m.c * self.m.m_ksi) / self.m.b

    def cc(self, k, x):
        return 1./2 * (self.ffi_up(k, x) + self.ffi_down(k, x))

    def ffi_up(self, k, x):
        cc = self.m.n if k == self.m.N-1 else self.nn(k) + self.m.n
        res = []
        for i in range(1, cc + 1):
            bb_val = self.bb(i, k)
            if bb_val != 0:
                val = (sgn(bb_val)*self.m.fi - self.aa(i, k).T @ x) / bb_val
                res.append(val)
        return min(res)

    def ffi_down(self, k, x):
        cc = self.m.n if k == self.m.N-1 else self.nn(k) + self.m.n
        res = []
        for i in range(1, cc + 1):
            bb_val = self.bb(i, k)
            if bb_val != 0:
                val = (-sgn(bb_val)*self.m.fi - self.aa(i, k).T @ x) / bb_val
                res.append(val)
        return max(res)

    # All result
    @cache
    def cond_I(self, k: int):
        if k == self.m.N + 1:
            return self.m.cond_F()
        elif k == self.m.N:
            return self.cond_I_N()
        elif k == self.m.N - 1:
            return self.cond_I_N_1()
        else:
            return self.cond_I_other(k)

    def gamma(self, k, x) -> Optional[float]:
        if k == self.m.N:
            return self.gamma_N(x)
        elif k in range(0, (self.m.N-1) + 1):
            return self.gamma_other(k, x)
        else:
            return None
