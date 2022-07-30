# other_solvers

from types import FunctionType

import numpy as np

from scipy.integrate import quad

from solver_main import (
    calculate_right_hand_side, check_validity_and_get_number_of_intervals)
from simple_visualization import constant


def make_iteration_matrix(partition: np.array,
                          function:  FunctionType=None) -> dict:
    intervals   = np.size(partition) - 1

    length = intervals - 1
    matrix = np.zeros([length, length])
    if function == None:
        values = np.array([(partition[i+1] - partition[i])**(-1)
                           for i in range(intervals)])
    else:
        values = np.zeros(intervals)
        for j in range(length):
            integrant = lambda x: function(x)*(partition[j] - partition[j+1])**(-2)
            try:
                values[j] = quad(integrant, partition[j], partition[j+1])[0]
            except KeyError:
                print(f"other_solvers: {partition.__str__() = }")

    for i in range(length):
        matrix[i, i]   =  values[i] + values[i+1]
        if i%2 == 1:
            matrix[i, i-1] = matrix[i-1, i] = -values[i]
            if i+1 < length:
                matrix[i, i+1] = matrix[i+1, i] = -values[i+1]

    return matrix

""" 
copied code from 
https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy
answer from stackPusher
"""
def transposeMatrix(m):
    return list(map(list,zip(*m)))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors
""""""


def Jacobi(partition: np.array,
           f_func:    FunctionType,
           a_func:    FunctionType or None,
           epsilon:   float=10**(-10),
           interval:  list=[0,1],
           matrices:  dict=None)                -> np.array and np.array:
    """"""
    num_inter = check_validity_and_get_number_of_intervals(partition, interval)
    
    """initialisation"""
    b_vector  = calculate_right_hand_side(partition, num_inter, f_func)
    # print(b_vector)
    if matrices != None:
        A_matrix = matrices["iter"][num_inter]
    else:
        A_matrix = make_iteration_matrix(partition, a_func)
    # print(A_matrix)

    diagonal = np.diag(A_matrix)
    # print(diagonal)

    """setting the starting values"""
    iteration = 0
    alpha_vec = np.ones(num_inter - 1)
    residuum  = 1

    denominator = np.linalg.norm(b_vector, np.inf)
    while epsilon < residuum:
        """"""
        x = np.matmul(A_matrix, alpha_vec) - b_vector
        alpha_vec -= x/diagonal

        """updating the residuum using the infinity norm"""
        numerator   = np.linalg.norm(np.matmul(A_matrix, alpha_vec) - b_vector, np.inf)
        residuum    = abs(numerator/denominator)
    
        iteration  += 1
        # print(f"other_solvers: Jacobi: {iteration = }, {residuum = }")


    """returning the solution, the residuum and the iteration counter"""
    return alpha_vec, residuum, iteration

def direct_solver(partition: np.array,
                  f_func:    FunctionType,
                  a_func:    FunctionType or None,
                  epsilon:   float=10**(-10),
                  interval:  list=[0, 1],
                  matrices:  dict=None)             -> np.array and np.array:    
    """initialisation"""
    num_inter = check_validity_and_get_number_of_intervals(partition, interval)
    b_vector  = calculate_right_hand_side(partition, num_inter, f_func)

    if matrices != None:
        A_matrix = matrices["iter"][num_inter]
    else:
        A_matrix = make_iteration_matrix(partition, a_func)

    """calculating the solution and returning it"""
    solution  = np.matmul(np.linalg.inv(A_matrix), b_vector)
    return solution, 0, 1

def Gauss_Seidel(partition: np.array,
                 f_func:    FunctionType,
                 a_func:    FunctionType or None,
                 epsilon:   float=10**(-10),
                 interval:  list=[0, 1],
                 matrices:  dict=None) -> np.array and np.array:
    # A_mat: np.array, b_vec: np.array, num_iter: int, x_vec: np.array
    """Solves Ax = b iteratively"""
    num_inter = check_validity_and_get_number_of_intervals(partition, interval)
    b_vec = calculate_right_hand_side(partition, num_inter, f_func)

    if matrices != None:
        A_mat = matrices["iter"][num_inter]
    else:
        A_mat = make_iteration_matrix(partition, a_func)

    next_step = np.zeros(num_inter+1)
    residuum = epsilon + 1
    denominator = np.linalg.norm(b_vec, np.inf)

    length = len(b_vec)
    x_vec = np.zeros(length)
    next_step = np.zeros(length)

    iterations = 0
    while residuum > epsilon:
        for j in range(length):
            d = b_vec[j]

            for i in range(0, length):
                if(j != i):
                    d -= A_mat[j][i] *x_vec[i]
            # updating the value of our solution
            next_step[j] = d / A_mat[j][j]
            # returning our updated solution

            # sum_1 = sum([A_mat[i, j]*next_step[j] for j in range(i)])
            # sum_2 = sum([A_mat[i, j]*x_vec[j] for j in range(i+1, length)])

            # next_step[i] = (b_vec[i] - sum_1 - sum_2)/A_mat[i, i]

        # print(f"{iteration}: {iter_step}")
        x_vec = next_step

        numerator = np.linalg.norm(np.matmul(A_mat, x_vec) - b_vec, np.inf)
        residuum = abs(numerator/denominator)
        iterations += 1

    return x_vec, residuum, iterations

def main():
    num_nodes  = 9
    partition  = np.linspace(0, 1, num_nodes)
    print(Jacobi(partition, lambda x: x*(x-1), constant))

if __name__ == "__main__":
    main()