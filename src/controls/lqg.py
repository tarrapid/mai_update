from typing import Optional
from scipy.linalg import inv

from src.utils.general import cache
from .abstract import AbstractControl


class LQG(AbstractControl):
    """
    Linear–quadratic–Gaussian control
    https://en.wikipedia.org/wiki/Algebraic_Riccati_equation
    Block: Context of the discrete-time algebraic Riccati equation
    """
    def __init__(self, model, Q, R):
        super().__init__(model)
        self.Q = Q
        self.R = R

    def gamma(self, k, x) -> Optional[float]:
        if k in range(0, self.m.N + 1):
            return (-self.F(k) @ x)[0]
        else:
            return None

    @cache
    def F(self, k: int):
        if k in range(0, self.m.N + 1):
            return inv(self.m.vec_B.T @ self.P(k+1) @ self.m.vec_B + self.R) \
                 @ self.m.vec_B.T @ self.P(k+1) @ self.m.mtx_A
        else:
            return None

    @cache
    def P(self, k: int):
        if k == self.m.N+1:
            return self.Q
        elif k in range(0, self.m.N + 1):
            s1 = self.Q + self.m.mtx_A.T @ self.P(k+1) @ self.m.mtx_A
            s2 = self.m.mtx_A.T @ self.P(k+1) @ self.m.vec_B \
                 @ inv(self.m.vec_B.T @ self.P(k+1) @ self.m.vec_B + self.R) \
                 @ self.m.vec_B.T @ self.P(k+1) @ self.m.mtx_A
            return s1 - s2
        else:
            return None
