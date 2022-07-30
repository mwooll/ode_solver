import numpy as np

from mesh_conversion import downsize_mesh


def calculate_diagonal(m: int,
                       n: int,
                       grid: np.array(float),
                       dist: np.array(float)) -> float:
    """calculates the values on the diagonal"""
    diag_1_h = (dist[n]**3 + dist[n+1]**3)/15

    diag_2_h = dist[m + 1]*dist[n]/6
    diag_2_z = 2*grid[m] + grid[m + 1] - 2*grid[n] - grid[n - 1]
    diag_2 = diag_2_h*diag_2_z

    return diag_1_h + diag_2

def calculate_upper_diagonal(m: int,
                             n: int,
                             grid: np.array(float),
                             dist: np.array(float)) -> float:
    """calculates the values corresponding to m <= n + 1 / m >= n + 1"""
    udia_1_h = dist[m]*dist[n]/12
    udia_1_z = 2*grid[m] + grid[m - 1] - 2*grid[n] - grid[n - 1]
    udia_1 = udia_1_h*udia_1_z

    udia_2_h = dist[m + 1]*dist[n + 1]/12
    udia_2_z = 2*grid[m] + grid[m + 1] - 2*grid[n] - grid[n + 1]
    udia_2 = udia_2_h*udia_2_z

    udia_3_h = dist[m + 1]*dist[n]/12
    udia_3_z = 2*grid[m] + grid[m + 1] - 2*grid[n] - grid[n - 1]
    udia_3 = udia_3_h*udia_3_z

    udia_4_h = dist[n + 1]**3/10

    return udia_1 + udia_2 + udia_3 + udia_4_h

def calculate_rest(m: int,
                   n: int,
                   grid: np.array(float),
                   dist: np.array(float)) -> float:
    """calculates the values corresponding to m <= n + 2 / m >= n + 2"""
    rest_1_h = dist[m]*dist[n]/12
    rest_1_z = 2*grid[m] + grid[m - 1] - 2*grid[n] - grid[n - 1]
    rest_1 = rest_1_h*rest_1_z

    rest_2_h = dist[m + 1]*dist[n + 1]/12
    rest_2_z = 2*grid[m] + grid[m + 1] - 2*grid[n] - grid[n + 1]
    rest_2 = rest_2_h*rest_2_z

    rest_3_h = dist[m]*dist[n + 1]/12
    rest_3_z = 2*grid[m] + grid[m - 1] - 2*grid[n] - grid[n + 1]
    rest_3 = rest_3_h*rest_3_z

    rest_4_h = dist[m + 1]*dist[n + 1]/12
    rest_4_z = 2*grid[m] + grid[m + 1] - 2*grid[n] - grid[n - 1]
    rest_4 = rest_4_h*rest_4_z

    return rest_1 + rest_2 + rest_3 + rest_4

def build_matrix(mesh: np.array(float)) -> np.array(np.array(float)):
    """docstring"""
    dimension = len(mesh) - 2
    step = np.array([mesh[i+1] - mesh[i] for i in range(dimension+1)])
    upper_tri = np.zeros((dimension, dimension))
    diagonal = np.zeros(dimension)

    """main part"""
    for m in range(0, dimension):
        for n in range(m, dimension):
            if m == n:
                diagonal[m] = calculate_diagonal(m, n, mesh, step)
            elif m == n - 1:
                upper_tri[m, n] = calculate_upper_diagonal(m, n, mesh, step)
            else:
                upper_tri[m, n] = calculate_rest(m, n, mesh, step)

    """stitching the components together"""
    lower_tri = np.transpose(upper_tri)
    diagonal_matrix = np.diag(diagonal)
    matrix = diagonal_matrix + upper_tri + lower_tri
    return matrix

def build_matrices(mesh: np.array(float)) -> np.array(np.array(float)):
    length = len(mesh) - 1
    matrices = {}
    cur_mesh = mesh
    cur_val  = np.zeros(length+1)
    matrices[length] = build_matrix(cur_mesh)
    while length%2 == 0 and 2 < length:
        cur_mesh, cur_val, length = downsize_mesh(cur_mesh, cur_val, length)
        # print(f"build_full_matrix: {cur_val.__str__() = }, {cur_mesh.__str__() = }")
        matrices[length] = build_matrix(cur_mesh)
    return matrices

def main():
    mesh = np.array([0, 1/6, 1/3, 1/2, 2/3, 5/6, 1])
    matrix = build_matrix(mesh)
    print(matrix)

if __name__ == "__main__":
    main()