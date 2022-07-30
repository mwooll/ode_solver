# comparison

from time import perf_counter_ns
from types import FunctionType

import numpy as np
from bokeh.plotting import figure, save, show, output_file

from solver_main import initialization, make_iteration_matrices, \
    get_small_partitions, sparse_init
from other_solvers import Jacobi, direct_solver, Gauss_Seidel
from build_full_matrix import build_matrices


def sinus(X: np.array)              -> np.array:
    return np.sin(X*np.pi)

def constant(X: np.array or float)  -> np.array or float:
    if hasattr(X, "__iter__"):
        return np.ones(len(X))
    return 1

def special_a(X: np.array or float) -> np.array or float:
    if hasattr(X, "__iter__"):
        return np.array([special_a(x) for x in X])
    return 1 + 9*(0.5 < X)


def test_solver(solver:     FunctionType,
                to_test:    list,
                function_f: FunctionType, 
                function_a: FunctionType,
                interval:   list=[0, 1])        -> None:
    """short function to test and compare different algorithms for the same problem"""
    observations  = {}

    for i in to_test:
        start_time    = perf_counter_ns()
        partition = np.linspace(interval[0], interval[1], i)
        observations[i]  = [solver(partition=partition,
                                  f_func=function_f,
                                  a_func=function_a)[2]]
        stop_time     = perf_counter_ns()
        observations[i].append((stop_time - start_time)/10**(9))

    return observations

def make_comparisons(func_f:    FunctionType,
                     func_a:    FunctionType,
                     stop:      int,
                     solvers:   list,
                     start:     int=1,
                     interval:  list=[0, 1],
                     inc_mat:   str="yes",
                     names:     list=None)     -> (dict, list):
    """ """
    if names == None:
        names = ["multigrid method", "numpy inverse", "Jacobi", "Gauss-Seidel"]
    to_test = [2**k for k in range(1, stop +1)]

    speed   = {}
    iterations = {}
    for i in to_test:
        print(f"comparison: i = {i}")
        speed[i] = {}
        iterations[i] = {}
        large_part = np.linspace(interval[0], interval[1], i+1)
        partitions = get_small_partitions(large_part, i)
        matrices = None
        if inc_mat in ["no", "full"]:
            matrices = {"iter": make_iteration_matrices(partitions, func_a)}

        if inc_mat == "full":
            disturbance = build_matrices(partitions[i])
            matrices["iter"] = {k: matrices["iter"][k] + disturbance[k]
                                for k in matrices["iter"].keys()}

        for k, solver in enumerate(solvers):
            # print(f"comparison: solver = {names[k]}")
            start_time = perf_counter_ns()
            iter_speed = solver(partitions[i],
                                func_f,
                                func_a,
                                # matrices=matrices, 
                                epsilon=10**(-10))
            
            stop_time  = perf_counter_ns()
            speed[i][names[k]] = (stop_time - start_time)/10**(9)
            iterations[i][names[k]] = iter_speed[2]

        print(speed)

    print(f"comparison: {iterations = }")
    return speed, to_test, iterations

def plot_comparison(times:  dict,
                    tested: list,
                    name:   str,
                    speed:  list=None)    -> None:
    """dedicated function to plot the results from make_comparison"""
    
    """preparing the plot"""
    plot = figure(title="Berechnungszeit", plot_height=400, plot_width=800,
                  x_axis_type="log", x_axis_label="Anzahl Teilintervalle",
                  y_axis_type="log", y_axis_label="Berechnungszeit [s]",
                  toolbar_location=None, tools="")
    plot.xaxis.ticker.base = 2

    """plotting the times"""
    output_file(name)
    colours = ["blue", "red", "green", "purple"]
    print(times)
    solver_names = times[list(times.keys())[0]].keys()
    for colour, solver in zip(colours, solver_names):
        y = [times[k][solver] for k in times.keys()]
        plot.line(x=tested, y=y, line_width=2, 
                  legend_label=solver, line_color=colour)
        plot.circle(x=tested, y=y, fill_color="white", size=4, line_color=colour)

    if speed != None:
        line_style = ["dashed", "dotdash", "dotted", ""]
        for k, it in enumerate(speed):
            if it[1] == 0:
                label = f"{it[0]}"
            else:
                label = f"{it[0]}*h^(-{it[1]})"
            x_values = np.linspace(tested[0], tested[-1])
            y_values = [it[0]*x_val**it[1] for x_val in x_values]
            plot.line(x_values, y_values, color="black", 
                      line_dash=line_style[k],
                      legend_label=label)

    plot.legend.location = "top_left"
    save(plot)
    show(plot)
    return

def plot_iterations(iterations: dict,
                    tested:     list,
                    name:       str,
                    speed:      list=None) -> None:
    """dedicated function to plot the results from make_comparison"""
    
    """preparing the plot"""
    plot = figure(title="Iterationen", plot_height=400, plot_width=800,
                  x_axis_type="log", x_axis_label="Anzahl Teilintervalle",
                  y_axis_type="log", y_axis_label="Anzahl Iterationen",
                  toolbar_location=None, tools="")
    plot.xaxis[0].ticker.base = 2
    plot.yaxis[0].ticker.base = 10

    """plotting the times"""
    output_file(name)
    colours = ["blue", "red", "green", "purple"]
    solver_names = iterations[list(iterations.keys())[0]].keys()
    solver_names = [x for x in solver_names if not x == "numpy inverse"]
    for colour, solver in zip(colours, solver_names):
        y = [iterations[k][solver] for k in iterations.keys()]
        plot.line(x=tested, y=y, line_width=2, 
                  legend_label=solver, line_color=colour)
        plot.circle(x=tested, y=y, fill_color="white", size=4, line_color=colour)

    if speed != None:
        line_style = ["dashed", "dotdash", "dotted", ""]
        for k, it in enumerate(speed):
            if it[1] == 0:
                label = f"{it[0]}"
            else:
                label = f"{it[0]}*h^(-{it[1]})"
            x_values = np.linspace(tested[0], tested[-1])
            y_values = [it[0]*x_val**it[1] for x_val in x_values]
            plot.line(x_values, y_values, color="black", 
                      line_dash=line_style[k],
                      legend_label=label)

    plot.legend.location = "top_left"
    save(plot)
    show(plot)
    return

def compare_all_four()         -> None:
    """comparing all three methods"""
    function_f = sinus
    function_a = constant

    stop_at = 7
    solvers = [initialization, direct_solver, Jacobi, Gauss_Seidel]
    # names = ["full multigrid", "direct", "sparse multigrid"]
    result, tested, iterations = make_comparisons(function_f,
                                                  function_a,
                                                  stop_at,
                                                  solvers,
                                                  start=2,
                                                  # names=names,
                                                  inc_mat="no")
    file_name = f"experiments/times_k={stop_at}_M_D_J_G.html"
    iter_name = f"experiments/iterations_k={stop_at}_M_J_G.html"
    plot_comparison(result, tested, file_name)
    plot_iterations(iterations, tested, iter_name)
    return

def compare_mesh_direct(function_f, modi)       -> None:
    """comparing multi-grid and direct solver"""
    function_a = constant
    stop_at    = 11
    start      = 1

    for status in modi:
        print(f"\n\n{status = }")
        solvers = [initialization, direct_solver]
        faster, tested, it = make_comparisons(function_f,
                                             function_a,
                                             stop_at,
                                             solvers,
                                             start,
                                             inc_mat=status)
        file_name = f"experiments/times_k={stop_at}_m={status}_M_D.html"
        iter_name = f"experiments/iter_k={stop_at}_m={status}_M_D.html"
        plot_iterations(it, tested, iter_name)
        plot_comparison(faster, tested, file_name)
    with open(f"experiments/results_k={stop_at}.txt", "a") as file:
        file.write(f"tested: {tested}\n")
        file.write(f"faster: {faster}\n")
        file.write(f"iterations: {it}")
    return

def compare_GS_iterations():
    function_f = sinus
    function_a = constant
    stop_at = 4

    to_test = [2**i for i in range(1, stop_at)]

    results = {}
    for i in to_test:
        results[i] = {}
        partition = np.linspace(0, 1, i+1)
        for k in [0, 1, 10]:
            results[i][f"iterations = {k}"] = initialization(partition,
                                                             function_f,
                                                             function_a,
                                                             epsilon=10**(-14),
                                                             iterations=k)[2]
        print(results)
    file_name = "GS_iterations.html"
    plot_iterations(results, to_test, file_name)

def experiment_with_all_four():
    tested = [2**k for k in range(1, 8)]
    # iterations = {2: {'multigrid method': 1, 'numpy inverse': 1, 'Jacobi': 1, 'Gauss-Seidel': 1}, 4: {'multigrid method': 6, 'numpy inverse': 1, 'Jacobi': 172, 'Gauss-Seidel': 82}, 8: {'multigrid method': 7, 'numpy inverse': 1, 'Jacobi': 982, 'Gauss-Seidel': 455}, 16: {'multigrid method': 8, 'numpy inverse': 1, 'Jacobi': 4522, 'Gauss-Seidel': 2090}, 32: {'multigrid method': 8, 'numpy inverse': 1, 'Jacobi': 19288, 'Gauss-Seidel': 8916}, 64: {'multigrid method': 8, 'numpy inverse': 1, 'Jacobi': 79612, 'Gauss-Seidel': 36804}, 128: {'multigrid method': 8, 'numpy inverse': 1, 'Jacobi': 323862, 'Gauss-Seidel': 149555}}
    # iter_name = "experiments/iterations_k=7_M_J_G.html"
    # iter_speed = [[8, 0], [20, 2], [9, 2]]
    # plot_iterations(iterations, tested, iter_name, iter_speed)
    
    times = {2: {'multigrid method': 0.0075932, 'numpy inverse': 0.0031986, 'Jacobi': 0.005785, 'Gauss-Seidel': 0.0030844}, 4: {'multigrid method': 0.0127828, 'numpy inverse': 0.0154023, 'Jacobi': 0.0195183, 'Gauss-Seidel': 0.0147561}, 8: {'multigrid method': 0.0273431, 'numpy inverse': 0.0167295, 'Jacobi': 0.0737149, 'Gauss-Seidel': 0.0433316}, 16: {'multigrid method': 0.0294616, 'numpy inverse': 0.0990106, 'Jacobi': 0.2523875, 'Gauss-Seidel': 0.5076781}, 32: {'multigrid method': 0.1319198, 'numpy inverse': 0.1088319, 'Jacobi': 0.7148984, 'Gauss-Seidel': 8.1385528}, 64: {'multigrid method': 0.4904738, 'numpy inverse': 0.1257145, 'Jacobi': 2.443832, 'Gauss-Seidel': 124.663623}, 128: {'multigrid method': 1.0933194, 'numpy inverse': 0.2197691, 'Jacobi': 13.4916446, 'Gauss-Seidel': 1540.1156198}}
    conv_speed_all = [[10**(-2), 1], [10**(-2), 1], [10**(-3), 2], [10**(-5), 4]]
    file_name = "experiments/times_k=7_M_D_J_G.html"
    plot_comparison(times, tested, file_name, conv_speed_all)

def experiment_mg_direct():
    tested = [2**k for k in range(1, 12)]
    times = {2: {'multigrid method': 0.0035823, 'numpy inverse': 0.0026905}, 4: {'multigrid method': 0.0058464, 'numpy inverse': 0.0062581}, 8: {'multigrid method': 0.0189109, 'numpy inverse': 0.011512}, 16: {'multigrid method': 0.0384209, 'numpy inverse': 0.0224312}, 32: {'multigrid method': 0.1112418, 'numpy inverse': 0.0500268}, 64: {'multigrid method': 0.4078299, 'numpy inverse': 0.0983423}, 128: {'multigrid method': 1.4340164, 'numpy inverse': 0.2054411}, 256: {'multigrid method': 4.4188156, 'numpy inverse': 0.2914556}, 512: {'multigrid method': 15.540935, 'numpy inverse': 0.6245932}, 1024: {'multigrid method': 57.4162981, 'numpy inverse': 1.1542292}, 2048: {'multigrid method': 291.990309, 'numpy inverse': 2.8208472}}
    conv_speed_all = [[15*10**(-4), 1], [0.0002, 1.85]]
    file_name = "experiments/times_k=7_M_D_J_G.html"
    plot_comparison(times, tested, file_name, conv_speed_all)

def main():
    compare_mesh_direct(sinus, modi=["full"])
    # compare_all_four()

if __name__ == "__main__":
    # results = main()
    # experiment_mg_direct()
    experiment_with_all_four()
