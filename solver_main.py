# solver_main

from types import FunctionType

import numpy as np
from scipy.integrate import quad
from scipy import sparse
import scipy.sparse.linalg as linalg

from Gauss_Seidel import tridiagonal_seidel, seidel
from mesh_conversion import downsize_mesh, downsize_values, augment_values


def check_validity_and_get_number_of_intervals(partition: np.array,
                                               interval:  list)     -> int:
    try:
        if partition[0] != interval[0] or partition[-1] != interval[-1]:
            raise Exception("partition must be a set fully contained in interval")
    except IndexError:
        raise IndexError("too few intervals, include more nodes in the partition")

    intervals = len(partition) - 1
    if intervals < 2:
        raise Exception("too few intervals, include more nodes in the partition")
    for k in range(intervals):
        if partition[k+1] <= partition[k]:
            raise Exception("the given partition is not valid")

    return intervals

def get_small_partitions(partition: np.array,
                         intervals: int)        -> dict:
    thetas = {intervals: partition}
    cur_int = intervals
    cur_par = partition
    cur_val = np.zeros(intervals-1)

    while cur_int > 2 and cur_int%2 == 0:
        cur_par, cur_val, cur_int = downsize_mesh(cur_par, cur_val, cur_int)
        thetas[cur_int] = cur_par
    return thetas

def make_iteration_matrices(partitions: np.array,
                            function:   FunctionType or None=None,
                            base:       int=2)             -> dict:
    intervals   = list(partitions.keys())[0]
    matrix_dict = {}

    cur_intervals = intervals
    while cur_intervals >= base:
        partition = partitions[cur_intervals]
        length = cur_intervals - 1
        matrix = np.zeros([length, length])
        if function == None:
            values = np.array([(partition[i+1] - partition[i])**(-1)
                               for i in range(cur_intervals)])
        else:
            values = np.zeros(cur_intervals)
            for j in range(cur_intervals):
                integrant = lambda x: function(x)*(partition[j] - partition[j+1])**(-2)
                values[j] = quad(integrant, partition[j], partition[j+1])[0]

        for i in range(length):
            matrix[i, i]   =  values[i] + values[i+1]
            if i%2 == 1:
                matrix[i, i-1] = matrix[i-1, i] = -values[i]
                if i+1 < length:
                    matrix[i, i+1] = matrix[i+1, i] = -values[i+1]

        matrix_dict[cur_intervals] = matrix
        if cur_intervals%2 == 1:
            break
        cur_intervals = cur_intervals//2
    return matrix_dict

def make_sparse_matrices(partitions:    np.array,
                          function:      FunctionType or None=None,
                          base:          int=2) -> dict:
    intervals   = list(partitions.keys())[0]
    matrix_dict = {}

    cur_intervals = intervals
    while cur_intervals >= base:
        partition = partitions[cur_intervals]
        length = cur_intervals - 1
        matrix = np.zeros([length, length])
        if function == None:
            values = np.array([(partition[i+1] - partition[i])**(-1)
                                for i in range(cur_intervals)])
        else:
            values = np.zeros(cur_intervals)
            for j in range(cur_intervals):
                integrant = lambda x: function(x)*(partition[j] - partition[j+1])**(-2)
                values[j] = quad(integrant, partition[j], partition[j+1])[0]

        # print(np.size(values), length)
        diagonals = [values[1:-1], values[:-1] + values[1:], values[1:-1]]
        matrix = sparse.diags(diagonals, [-1, 0, 1], format="lil")
        # print(matrix)
        # print(matrix.toarray())

        matrix_dict[cur_intervals] = matrix
        if cur_intervals%2 == 1:
            break
        cur_intervals = cur_intervals//2
    return matrix_dict

def hut_basis_vector(partition: np.array,
                     X:         float or np.array,
                     index:     int)                -> float or np.array:
    if index < 1 or len(partition) <= index:
        raise Exception("index should be between 1 and len(partition)-1")
    i = index

    left   = partition[i-1]
    middle = partition[i]
    right  = partition[i+1]

    if hasattr(X, "__iter__"):
        return np.array([0 + ((x - left)/(middle - left)*(x < middle) +
        (right - x)/(right - middle)*(middle <= x))*(left < x and x < right)
                for x in X])

    return 0 + ((X - left)/(middle - left)*(X < middle) +
        (right - X)/(right - middle)*(middle <= X))*(left < X and X < right)

def calculate_right_hand_sides(partitions: np.array,
                               function=None)        -> dict:
    right_sides = {}
    intervals = [par for par in partitions]
    for cur_int in intervals:
        right_sides[cur_int] = calculate_right_hand_side(partitions[cur_int],
                                                         cur_int,
                                                         function)
    return right_sides
    
def calculate_right_hand_side(partition: np.array,
                              intervals,
                              function=None)        -> np.array:
    if function == None:
        right_vec = np.array([(partition[i+1]-partition[i-1])/2 
                              for i in range(1, intervals)])

    else:
        right_vec = np.zeros(intervals-1)
        for i in range(intervals-1):
            integrant    = lambda x: function(x)*hut_basis_vector(partition, x, i+1)
            right_vec[i] = quad(integrant, partition[i], partition[i+2])[0] 

    return right_vec


def initialization(partition:   np.array,
                   f_func:      FunctionType or None,
                   a_func:      FunctionType or None,
                   epsilon:     float=10**(-10),
                   interval:    list=[0,1],
                   base:        int=2,
                   matrices:    dict=None,
                   iterations:  int=2):
    """
    This is an algorithm to compute differential equations of the form -(au')'= f.
    Where a, f are the parameters a_func, f_func and u is the solution to be calculated.
    To do this in a finite amount of time, we need to discretize everythig using partition.
    alpha_vec will then be a discretized approximation of u.

    note: epsilon should be greater than 10**(-15) to avoid infinite loops due to rounding
    this method was designed to use full matrices
    """

    """sorting the partition and calculating the number of subintervals"""
    partition = sorted(partition)
    num_int   = check_validity_and_get_number_of_intervals(partition, interval)

    small_partitions = get_small_partitions(partition, num_int)

    """computing the matrices needed and their inverses"""
    if matrices != None:
        iteration_matrices = matrices["iter"]
    else:
        iteration_matrices = make_iteration_matrices(small_partitions, a_func, base)

    """making sure base is compatible with the number of intervals"""
    try:
        base = min(num_int, min(iteration_matrices.keys()))
    except ValueError:
        base = num_int

    iteration_matrices["mini inverse"] = np.linalg.inv(iteration_matrices[base])

    """calculating r(s)"""
    r_vector = calculate_right_hand_side(partition, num_int, f_func)
    # print(iteration_matrices)
    # print(r_vectors)
    # return

    """setting the starting values for alpha_vec and residuum to enter the loop"""
    alpha_vec = np.zeros(num_int-1)

    iterations = 0
    residuum = epsilon + 1
    denominator = np.linalg.norm(r_vector, np.inf)

    while residuum > epsilon:
        # print(f"initialization: {iterations = }")
        alpha_vec = V_cycle(alpha_vec,
                            iteration_matrices,
                            r_vector,
                            num_int,
                            base,
                            iterations)

        numerator = np.linalg.norm(np.matmul(iteration_matrices[num_int],
                                             alpha_vec) - r_vector, np.inf)
        residuum = abs(numerator/denominator)
        # print(residuum)
        iterations += 1

    # print(residuum)
    return alpha_vec, residuum, iterations

def V_cycle(alpha_vec:  np.array,
            matrices:   dict,
            b_vector:   np.array,
            length:     int,
            base:       int,
            iterations: int):
    if length == base:
        if length == 2:
            eps = b_vector/matrices[length]
        else:
            eps = np.matmul(matrices["mini inverse"], b_vector)
        return eps
        
    """pre smoothing"""
    alpha_vec = seidel(matrices[length],
                                       b_vector,
                                       iterations,
                                       alpha_vec,
                                       length-1)

    """calculating residual"""
    d_vector = b_vector - np.matmul(matrices[length], alpha_vec)

    """restriction"""
    small_d, length = downsize_values(d_vector, length)
    eps = np.zeros_like(small_d)

    """recursion"""
    eps = V_cycle(eps,
                  matrices,
                  small_d,
                  length,
                  base,
                  iterations)

    """prolongation and correction"""
    correction, length = augment_values(eps, length)
    alpha_vec = alpha_vec + correction

    """post smoothing"""
    alpha_vec = seidel(matrices[length],
                       b_vector,
                       iterations,
                       alpha_vec,
                       length-1)

    return alpha_vec


def sparse_init(partition:   np.array,
                f_func:      FunctionType or None,
                a_func:      FunctionType or None,
                epsilon:     float=10**(-10),
                interval:    list=[0,1],
                base:        int=2,
                iterations:  int=2):
    """
    This is an algorithm to compute differential equations of the form -(au')'= f.
    Where a, f are the parameters a_func, f_func and u is the solution to be calculated.
    To do this in a finite amount of time, we need to discretize everythig using partition.
    alpha_vec will then be a discretized approximation of u.

    note: epsilon should be greater than 10**(-15) to avoid infinite loops due to rounding
    """

    """sorting the partition and calculating the number of subintervals"""
    partition = sorted(partition)
    num_int   = check_validity_and_get_number_of_intervals(partition, interval)

    small_partitions = get_small_partitions(partition, num_int)

    """computing the matrices needed and their inverses"""
    iteration_matrices = make_sparse_matrices(small_partitions, a_func, base)

    """making sure base is compatible with the number of intervals"""
    try:
        base = min(num_int, min(iteration_matrices.keys()))
    except ValueError:
        base = num_int

    iteration_matrices["mini inverse"] = linalg.inv(iteration_matrices[base])

    """calculating r(s)"""
    r_vector = calculate_right_hand_side(partition, num_int, f_func)
    # print(iteration_matrices)
    # print(r_vectors)
    # return

    """setting the starting values for alpha_vec and residuum to enter the loop"""
    alpha_vec = np.zeros(num_int-1)

    iterations = 0
    residuum = epsilon + 1
    denominator = np.linalg.norm(r_vector, np.inf)

    while residuum > epsilon:
        # print(f"initialization: {iterations = }")
        alpha_vec = V_sparse(alpha_vec,
                             iteration_matrices,
                             r_vector,
                             num_int,
                             base,
                             iterations)

        numerator = np.linalg.norm(iteration_matrices[num_int]*alpha_vec
                                   - r_vector, np.inf)
        # residuum = abs(numerator)
        residuum = abs(numerator/denominator)
        # print(residuum)
        iterations += 1

    return alpha_vec, residuum, iterations

def V_sparse(alpha_vec:  np.array,
             matrices:   dict,
             b_vector:   np.array,
             length:     int,
             base:       int,
             iterations: int):
    """base case"""
    if length == base:
        eps = matrices["mini inverse"]*b_vector
        # print(f"base: {eps = }")
        return eps
    
    """pre smoothing"""
    alpha_vec = tridiagonal_seidel(matrices[length],
                                   b_vector,
                                   iterations,
                                   alpha_vec,
                                   length-1)

    """calculating residual"""
    d_vector = b_vector - matrices[length]*alpha_vec

    """restriction"""
    small_d, length = downsize_values(d_vector, length)
    eps = np.zeros_like(small_d)
    # print(f"loop: {eps = })

    """recursion"""
    eps = V_sparse(eps,
                   matrices,
                   small_d,
                   length,
                   base,
                   iterations)

    """prolongation and correction"""
    correction, length = augment_values(eps, length)
    alpha_vec = alpha_vec + correction

    """post smoothing"""
    alpha_vec = tridiagonal_seidel(matrices[length],
                                   b_vector,
                                   iterations,
                                   alpha_vec,
                                   length-1)
    return alpha_vec

def test_function(X: float or np.array) -> float or np.array:
    return X**2 - 5*X + 3

def identity(X: float or np.array) -> float or np.array:
    return X

def constant(X: float) -> 1:
    return 1

def main_ode():
    num_nodes  = 9
    partition  = np.linspace(0, 1, num_nodes)
    # solution, residue, iterations, mes = ode_solver(partition, constant, constant, 10**(-10))
    solution, residue, iterations = sparse_init(partition,
                                                   lambda x: np.sin(x*np.pi),
                                                   None,
                                                   10**(-10),
                                                   base=4,
                                                   iterations=1)
    # print(solution)
    # print(residue)
    # print(iterations)
    return residue

def compare_full_and_sparse():
    num_nodes = 5
    partition = np.linspace(0, 1, num_nodes)
    function_f = lambda x: np.sin(x*np.pi)
    function_a = None
    print(sparse_init(partition,
                      function_f,
                      function_a))
    print(initialization(partition,
                          function_f,
                          function_a))

if __name__ == "__main__":
    # test_iter_matrices()
    out = main_ode()
    # compare_full_and_sparse()