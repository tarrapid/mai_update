import numpy as np
from src.model import Model


if __name__ == '__main__':
    mtx_A = np.array([[1, 1],
                      [0, 1]])
    N = 4
    n = 2

    sys = Model(
        mtx_A, b=1., c=1.,
        n=n, N=N,
        vec_lambda=np.array([0.7, 1]),
        fi=1.,
        m_ksi=0.0,
        eps=0.1,
    )

    #x_0 = np.array([1, 1.1])
    #x_list, u_list, ksi_list = sys.modelling(x_0=x_0, random_seed=10)

    sys.cond_I(k=N+1).A, sys.cond_I(k=N+1).b
    sys.cond_I(k=N).A, sys.cond_I(k=N).b

    # I_N_2 = sys.cond_I(k=N - 2)
    # I_N_2.A, I_N_2.b
    #
    # print('Lambda wave N+1')
    # print(sys.mtx_Lambda_wave(N+1))
    # print('Lambda N+1')
    # print(sys.mtx_Lambda(N + 1))
    #
    # print('Lambda wave N')
    # print(sys.mtx_Lambda_wave(N))
    # print('Lambda N')
    # print(sys.mtx_Lambda(N))
    #
    # print('Lambda wave N-1')
    # print(sys.mtx_Lambda_wave(N-1))
    # print('Lambda N-1')
    # print(sys.mtx_Lambda(N - 1))
    #
    # print('Lambda wave N-2')
    # print(sys.mtx_Lambda_wave(N - 2))
    #
    # print('Lambda N-2')
    # print(sys.mtx_Lambda(N - 2))

    #sys.gamma(N - 1, np.array([0.25, 0.25]))
    #u_k = sys.gamma(1, np.array([0.25, 0.25]))
    #u_k = sys.gamma(2, np.array([0.25, 0.25]))
    #u_k = sys.gamma(3, np.array([0.25, 0.25]))
    #print(u_k)