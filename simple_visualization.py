# simple_visualization

from types import FunctionType

import numpy as np
import matplotlib.pyplot as plotter

from bokeh.plotting import figure, save, show, output_file
from bokeh.models import Span

from solver_main import initialization, hut_basis_vector#, sparse_init
from build_full_matrix import build_matrices



def special_f(X: float or np.array)     -> float or np.array:
    # epsilon was chosen to be 1
    return 2 + (1 - 2*X + 4*X**3 - 2*X**4)/12

def special_g(X: float or np.array)     -> float or np.array:
    return X*(1 - X)

def special_a(X: float or np.array, 
              k: float, x_0: float)     -> float or np.array:
    if hasattr(X, "__iter__"):
        return [1 if x < x_0 else k for x in X]
    return 1 if X < x_0 else k


def special_sin(X: float or np.array)   -> float or np.array:
    return np.sin(X*np.pi)

def scaled_a(X: float or np.array)      -> float or np.array:
    return np.ones_like(X)/np.pi**2

def constant(X: float or np.array)      -> float or np.array:
    if hasattr(X, "__iter__"):
        return np.ones(len(X))
    return 1


def plot_u(X:           float or np.array,
           partition:   np.array,
           solution:    np.array)       -> float or np.array:
    length  = len(partition)
    to_plot = np.zeros(len(X))

    for k in range(1, length-1):
        to_plot += solution[k-1]*hut_basis_vector(partition, X, k)
    return to_plot

def plot_it(theta_N:    np.array,
            func_f:     FunctionType,
            func_a:     FunctionType=None,
            func_g:     FunctionType=None,
            matrices:   dict=None)              -> None:
    """calculating the solution"""
    cal, err, it, res = initialization(theta_N, func_f, func_a)

    """post processing"""
    num_nodes   = len(theta_N)
    plot_points = 101
    to_evaluate = np.linspace(0, 1, plot_points)
    solution    = plot_u(to_evaluate, theta_N, cal)

    """plotting the solution"""
    plotter.plot(to_evaluate, func_f(to_evaluate), "b-", label="f")
    plotter.plot(to_evaluate, solution, "r-", label="u")
    
    if func_a:
        plotter.plot(to_evaluate, func_a(to_evaluate), "g-", label="a")

    plotter.title(f"solution of ode with {num_nodes} nodes")
    plotter.grid(True)
    plotter.xlabel("x axis")
    plotter.ylabel("y axis")
    plotter.legend()
    plotter.show()

    """comparing the computed solution to the algebraic solution, if given"""
    if func_g:
        """calculating the error and its integral"""
        exact    = func_g(to_evaluate)
        error    = abs(solution - exact)
        integral = [sum(error[:k+1])/plot_points for k in range(plot_points)]

        """plotting the computed solution to the algebraic solution"""
        plotter.plot(to_evaluate, exact,    "y-", label="g")
        plotter.plot(to_evaluate, solution, "r-", label="u")
        plotter.title("reference plot")
        plotter.grid(True)
        plotter.xlabel("x axis")
        plotter.ylabel("y axis")
        plotter.legend()
        plotter.show()

        """plotting the error and the integral of the error"""
        plotter.plot(to_evaluate, error, "y-", label="e")
        plotter.plot(to_evaluate, integral, "c-", label="E")
        plotter.title("error of solution")
        plotter.grid(True)
        plotter.xlabel("x axis")
        plotter.ylabel("y axis")
        plotter.legend()
        plotter.show()

def u_plot(to_evaluate, solution, title) -> figure:
    u_plot = figure(title=title, plot_height=500, plot_width=500,
                    toolbar_location=None, tools="")
    u_plot.line(to_evaluate, solution)
    return u_plot

def uncontinuous_a(theta_N: np.array,
                   func_f:  FunctionType,
                   solver:  FunctionType,
                   heights: list=[10], 
                   cut_off: list=[1/2],)            -> None:
    """calculating the solution"""
    heights = sorted(heights)
    plot_points = len(theta_N)
    to_evaluate = np.linspace(0, 1, plot_points)
    colours = ["blue", "green", "black", "red", "purple", "cyan"]
    for x_0 in cut_off:
        print(f"{x_0 = }")
        u_plot = figure(title="", plot_height=500, plot_width=500,
                        toolbar_location=None, tools="")
        v_line = Span(location=x_0, dimension='height',
                      line_color="black", line_width=1, line_dash="dotdash")
        u_plot.renderers.append(v_line)
        for k, h in enumerate(heights):
            print(f"{k = }, {h = }")
            if h == 1:
                function_a = None
            else:
                function_a = lambda X: special_a(X, h, x_0)
            
            cal = solver(theta_N, func_f, function_a, iterations=2)[0]
            plottable = plot_u(to_evaluate, theta_N, cal)
            u_plot.line(np.linspace(0, 1, plot_points), plottable,
                        legend_label=f"h={h}", line_color=colours[k])

        """plotting u on a nice bokeh plot"""
        show(u_plot)
        save(u_plot)

    return

def main_plotter(nodes_num)     -> None:
    i = nodes_num
    partition  = np.linspace(0, 1, 2**i + 1)
    # function_a = constant
    function_f = lambda x: special_sin(x)*np.pi**2
    function_g = special_sin
    k = [1,2,5,10,0.5,0.2]
    x_0 = [9/19]

    solver = initialization

    output_file(f"u_for_uncontinuous_a_x={x_0}_i={i}.html")
    uncontinuous_a(partition, function_f, solver, k, x_0)
    return

    matrices = build_matrices(partition)

    plot_it(partition, function_f, None, function_g, matrices)

if __name__ == "__main__":
    main_plotter(8)