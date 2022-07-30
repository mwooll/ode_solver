# Gauss_Seidel

from time  import perf_counter_ns

import numpy as np

def check_validity_and_get_length_for_Gauss_Seidel(matrix: np.array, vector: np.array) -> int:
    shape = np.shape(matrix)
    if shape[0] != shape[1]:
        raise ValueError("input matrix needs to be a square matrix")
    # if np.linalg.det(matrix) == 0:
    #     raise ValueError("input matrix needs to be nonsingular")

    length = len(vector)
    if length != shape[0]:
        raise ValueError("input vector needs to have the same length as the input matrix")

    return length


def Gauss_Seidel_Verfahren(A_mat: np.array,
                           b_vec: np.array,
                           num_iter: int,
                           x_vec: np.array,
                           length: int) -> np.array:
    """Solves Ax = b iteratively"""
    # length = check_validity_and_get_length_for_Gauss_Seidel(A_mat, b_vec)
    next_step = np.zeros(length)
    for step in range(num_iter):
        for i in range(length):
            sum_1 = sum([A_mat[i, j]*next_step[j] for j in range(i)])
            sum_2 = sum([A_mat[i, j]*x_vec[j] for j in range(i+1, length)])
            
            next_step[i] = (b_vec[i] - sum_1 - sum_2)/A_mat[i, i]

        # print(f"{iteration}: {iter_step}")
        x_vec = next_step

    return x_vec

def tridiagonal_Gauss_Seidel(A_mat: np.array,
                             b_vec: np.array,
                             num_iter: int,
                             x_vec: np.array,
                             length: int) -> np.array:
    """Solves Ax = b iteratively"""
    # length = check_validity_and_get_length_for_Gauss_Seidel(A_mat, b_vec)
    m = length - 1
    next_step = np.zeros(length)
    for step in range(num_iter):
        next_step[0] = (b_vec[0] - A_mat[0, 1]*x_vec[1])/A_mat[0, 0]
        for i in range(1, m):
            # sum_1 = sum([A_mat[i, j]*next_step[j] for j in range(i)])
            # sum_2 = sum([A_mat[i, j]*x_vec[j] for j in range(i+1, length)])

            sum_1 = A_mat[i, i-1]*next_step[i-1]
            sum_2 = A_mat[i, i+1]*x_vec[i+1]
            
            next_step[i] = (b_vec[i] - sum_1 - sum_2)/A_mat[i, i]

        next_step[m] = (b_vec[m] - A_mat[m, m-1]*next_step[m-1])/A_mat[m, m]
        # print(f"{iteration}: {iter_step}")
        x_vec = next_step

    return x_vec

def seidel(a, b, num_iter, y, length):
    for step in range(num_iter):
        for j in range(0, length):
            # temp variable d to store b[j]
            d = b[j]

            for i in range(0, length):
                if(j != i):
                    d -= a[j][i] * y[i]
            # updating the value of our solution
            y[j] = d / a[j][j]
    # returning our updated solution
    return y 

def tridiagonal_seidel(a, b, num_iter, y, length):
    for step in range(num_iter):
        d = b[0]
        d -= a[0, 1] * y[1]
        y[0] = d / a[0, 0]

        for j in range(1, length-1):
            # temp variable d to store b[j]
            d = b[j]

            # for i in range(0, length):
                # if(j != i):
                    # d -= a[j][i] * y[i]
            d -= a[j, j-1] * y[j-1]
            d -= a[j, j+1] * y[j+1]
            # updating the value of our solution
            y[j] = d / a[j, j]

        d = b[length-1]
        d -= a[length-1, length-2] * y[length-2]
        y[length-1] = d / a[length-1, length-1]
    # returning our updated solution
    return y


def test_Gauss_Seidel() -> None:
    from scipy import sparse
    # import scipy.sparse.linalg as linalg
    n = 20
    iterations = 20
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format="lil")

    b = np.array([k for k in range(n)])

    start_time = perf_counter_ns()
    print("Seidel")
    x = np.zeros(n)
    print(seidel(A.toarray(), b, iterations, x, n))
    x = np.zeros(n)
    seidel_time = perf_counter_ns()
    print(seidel_time - start_time)

    print("Coarse")
    x = np.zeros(n)
    print(tridiagonal_seidel(A, b, iterations, x, n))
    coarse_time = perf_counter_ns()
    print(coarse_time - seidel_time)

    # print("Direct")
    # x = np.zeros(n)
    # print(linalg.spsolve(A,b))
    # direct_time = perf_counter_ns()
    # print(direct_time - coarse_time)


if __name__ == "__main__":
    test_Gauss_Seidel()