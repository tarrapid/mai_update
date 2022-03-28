import numpy as np


class Condition:
    '''
    Condition type: Ax <= b
    '''

    def __init__(self, A, b, add_symmetric=True):
        self.A = A
        self.b = b

        if add_symmetric:
            self.A = np.concatenate([A, -A], axis=0)
            self.b = np.concatenate([b, b])


def intersection(conds: list):
    A, b = [], []
    for cond in conds:
        A.append(cond.A)
        b.append(cond.b)
    return Condition(
        np.concatenate(A, axis=0),
        np.concatenate(b)
    )